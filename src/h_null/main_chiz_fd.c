/*
 * main_chiz_fd.c -- Longitudinal susceptibility from signed magnetization
 * finite differences.
 *
 * Computes
 *   chi_z(g,0;L) = d <Mz/L> / dh |_{h=0}
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <lapacke.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir((path), 0755)
#endif

#include "basis.h"
#include "diag.h"
#include "hamiltonian.h"
#include "lanczos.h"
#include "observables.h"

#define MAX_L_RUNS 32
#define MAX_G_POINTS 2048
#define ED_MAX_L 12
#define MAX_L_SUPPORTED 22

static const int DEFAULT_L[] = {4, 6, 8, 10, 12, 14, 16, 18, 20};
static const int N_DEFAULT_L = (int)(sizeof(DEFAULT_L) / sizeof(DEFAULT_L[0]));

typedef struct {
    int pbc;
    double dh;
    int smoke_max_g_points;
    int smoke_has_g_window;
    double smoke_g_min;
    double smoke_g_max;
    int use_fsync;
    int stop_after;
    int has_output_dir;
    char output_dir[512];
    int L_values[MAX_L_RUNS];
    int N_L;
} Params;

typedef struct {
    double mz_m2;
    double mz_m1;
    double mz_p1;
    double mz_p2;
    double chi_fd;
    double oddness1;
    double oddness2;
} ChizRow;

static int max_iter_for_L(int L)
{
    if (L <= 16) return 200;
    if (L <= 18) return 150;
    if (L <= 20) return 100;
    return 60;
}

static int method_code_for_L(int L)
{
    return (L <= ED_MAX_L) ? 0 : 1;
}

static double parse_double_checked(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (end == s || *end != '\0' || errno != 0 || !isfinite(v)) {
        fprintf(stderr, "[ising_chiz_fd] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_int_checked(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0) {
        fprintf(stderr, "[ising_chiz_fd] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static int parse_pbc(const char *s)
{
    if (s[0] == '0' && s[1] == '\0') return 0;
    if (s[0] == '1' && s[1] == '\0') return 1;
    fprintf(stderr, "[ising_chiz_fd] pbc must be 0 (OBC) or 1 (PBC)\n");
    exit(EXIT_FAILURE);
}

static int is_official_L(int L)
{
    for (int i = 0; i < N_DEFAULT_L; i++)
        if (DEFAULT_L[i] == L)
            return 1;
    if (L == 22)
        return 1;
    return 0;
}

static Params parse_args(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "\nUsage: %s <pbc> <dh> [L1 L2 ...]\n\n"
                "  pbc  : 1=PBC, 0=OBC\n"
                "  dh   : longitudinal-field finite-difference step\n"
                "  L... : optional official sizes; default 4 6 8 10 12 14 16 18 20\n"
                "         L=22 is supported only when requested explicitly.\n\n"
                "Options:\n"
                "  --max-g-points N : smoke-test only; use first N loaded g points\n\n"
                "  --g-window gmin gmax : smoke-test only; use loaded g with gmin <= g <= gmax\n\n"
                "  --stop-after N    : diagnostic only; stop after N completed points without smoke header\n\n"
                "  --output-dir DIR  : write outputs under DIR/{PBC,OBC} instead of data/h_null/chiz_fd/dh_<tag>\n\n"
                "  --no-fsync        : flush stdio only; default also fsyncs each data row\n\n"
                "Method is fixed by size: L<=12 full ED, L>=14 Lanczos.\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    Params p;
    p.pbc = parse_pbc(argv[1]);
    p.dh = parse_double_checked(argv[2], "dh");
    p.smoke_max_g_points = 0;
    p.smoke_has_g_window = 0;
    p.smoke_g_min = 0.0;
    p.smoke_g_max = 0.0;
    p.use_fsync = 1;
    p.stop_after = 0;
    p.has_output_dir = 0;
    p.output_dir[0] = '\0';
    p.N_L = 0;

    if (p.dh <= 0.0) {
        fprintf(stderr, "[ising_chiz_fd] dh must be positive\n");
        exit(EXIT_FAILURE);
    }

    if (argc == 3) {
        for (int i = 0; i < N_DEFAULT_L; i++)
            p.L_values[p.N_L++] = DEFAULT_L[i];
    } else {
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--max-g-points") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "[ising_chiz_fd] --max-g-points requires N\n");
                    exit(EXIT_FAILURE);
                }
                p.smoke_max_g_points = parse_int_checked(argv[++i], "--max-g-points");
                if (p.smoke_max_g_points <= 0 || p.smoke_max_g_points > MAX_G_POINTS) {
                    fprintf(stderr,
                            "[ising_chiz_fd] --max-g-points must be in [1,%d]\n",
                            MAX_G_POINTS);
                    exit(EXIT_FAILURE);
                }
                continue;
            }

            if (strcmp(argv[i], "--g-window") == 0) {
                if (i + 2 >= argc) {
                    fprintf(stderr, "[ising_chiz_fd] --g-window requires gmin gmax\n");
                    exit(EXIT_FAILURE);
                }
                p.smoke_g_min = parse_double_checked(argv[++i], "--g-window gmin");
                p.smoke_g_max = parse_double_checked(argv[++i], "--g-window gmax");
                if (p.smoke_g_min > p.smoke_g_max) {
                    fprintf(stderr, "[ising_chiz_fd] --g-window requires gmin <= gmax\n");
                    exit(EXIT_FAILURE);
                }
                p.smoke_has_g_window = 1;
                continue;
            }

            if (strcmp(argv[i], "--stop-after") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "[ising_chiz_fd] --stop-after requires N\n");
                    exit(EXIT_FAILURE);
                }
                p.stop_after = parse_int_checked(argv[++i], "--stop-after");
                if (p.stop_after <= 0) {
                    fprintf(stderr, "[ising_chiz_fd] --stop-after must be positive\n");
                    exit(EXIT_FAILURE);
                }
                continue;
            }

            if (strcmp(argv[i], "--output-dir") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "[ising_chiz_fd] --output-dir requires DIR\n");
                    exit(EXIT_FAILURE);
                }
                snprintf(p.output_dir, sizeof(p.output_dir), "%s", argv[++i]);
                if (p.output_dir[0] == '\0') {
                    fprintf(stderr, "[ising_chiz_fd] --output-dir cannot be empty\n");
                    exit(EXIT_FAILURE);
                }
                p.has_output_dir = 1;
                continue;
            }

            if (strcmp(argv[i], "--no-fsync") == 0) {
                p.use_fsync = 0;
                continue;
            }

            if (p.N_L >= MAX_L_RUNS) {
                fprintf(stderr, "[ising_chiz_fd] too many L values (max %d)\n", MAX_L_RUNS);
                exit(EXIT_FAILURE);
            }
            int L = parse_int_checked(argv[i], "L");
            if (!is_official_L(L)) {
                fprintf(stderr,
                        "[ising_chiz_fd] unsupported L=%d. Official pipeline sizes are "
                        "4 6 8 10 12 14 16 18 20, with explicit L=22 support.\n",
                        L);
                exit(EXIT_FAILURE);
            }
            p.L_values[p.N_L++] = L;
        }
    }

    if (p.N_L == 0) {
        for (int i = 0; i < N_DEFAULT_L; i++)
            p.L_values[p.N_L++] = DEFAULT_L[i];
    }

    return p;
}

static int ensure_dir(const char *path)
{
    if (MKDIR(path) == 0)
        return 0;
    if (errno == EEXIST)
        return 0;
    fprintf(stderr, "[ising_chiz_fd] cannot create directory %s\n", path);
    return -1;
}

static int ensure_chiz_bc_dirs(const char *out_dir)
{
    char bc_dir[512];
    snprintf(bc_dir, sizeof(bc_dir), "%s/PBC", out_dir);
    if (ensure_dir(bc_dir) != 0)
        return -1;
    snprintf(bc_dir, sizeof(bc_dir), "%s/OBC", out_dir);
    if (ensure_dir(bc_dir) != 0)
        return -1;
    return 0;
}

static void format_dh_tag(double dh, char *out, size_t out_sz)
{
    char sci[64];
    char mant[64];
    char *e = NULL;

    snprintf(sci, sizeof(sci), "%.12e", dh);
    e = strchr(sci, 'e');
    if (!e) {
        snprintf(out, out_sz, "dh");
        return;
    }

    *e = '\0';
    snprintf(mant, sizeof(mant), "%s", sci);

    size_t len = strlen(mant);
    while (len > 0 && mant[len - 1] == '0')
        mant[--len] = '\0';
    if (len > 0 && mant[len - 1] == '.')
        mant[--len] = '\0';
    for (size_t i = 0; mant[i] != '\0'; i++)
        if (mant[i] == '.')
            mant[i] = 'p';

    snprintf(out, out_sz, "dh_%se%s", mant, e + 1);
}

static int build_output_dir(double dh, char *out_dir, size_t out_sz)
{
    char tag[64];
    format_dh_tag(dh, tag, sizeof(tag));

    if (ensure_dir("data") != 0) return -1;
    if (ensure_dir("data/h_null") != 0) return -1;
    if (ensure_dir("data/h_null/chiz_fd") != 0) return -1;

    snprintf(out_dir, out_sz, "data/h_null/chiz_fd/%s", tag);
    if (ensure_dir(out_dir) != 0) return -1;
    if (ensure_chiz_bc_dirs(out_dir) != 0) return -1;

    return 0;
}

static int file_exists_readable(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;
    fclose(fp);
    return 1;
}

static int load_first_column(const char *path, double *g_arr, int max_pts)
{
    FILE *fp = fopen(path, "r");
    char line[4096];
    int n = 0;
    int line_no = 0;

    if (!fp) {
        fprintf(stderr, "[ising_chiz_fd] cannot open g-grid source %s\n", path);
        return -1;
    }

    while (fgets(line, sizeof(line), fp)) {
        char *s = line;
        char *end = NULL;
        double g;

        line_no++;
        while (isspace((unsigned char)*s))
            s++;
        if (*s == '\0' || *s == '#')
            continue;

        errno = 0;
        g = strtod(s, &end);
        if (end == s) {
            fprintf(stderr,
                    "[ising_chiz_fd] malformed g-grid row in %s:%d\n",
                    path, line_no);
            fclose(fp);
            return -1;
        }
        if (errno == ERANGE || !isfinite(g))
            continue;

        if (n >= max_pts) {
            fprintf(stderr,
                    "[ising_chiz_fd] too many g points in %s (max %d)\n",
                    path, max_pts);
            fclose(fp);
            return -1;
        }
        g_arr[n++] = g;
    }

    if (ferror(fp)) {
        fprintf(stderr, "[ising_chiz_fd] error while reading %s\n", path);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    if (n <= 0) {
        fprintf(stderr, "[ising_chiz_fd] no finite g points found in %s\n", path);
        return -1;
    }

    return n;
}

int load_existing_g_grid(int L, int pbc, int method_code,
                         double *g_arr, int max_pts,
                         char *source_path, size_t source_path_sz)
{
    char candidates[4][512];
    int n_candidates = 0;

    if (method_code != method_code_for_L(L)) {
        fprintf(stderr,
                "[ising_chiz_fd] internal method mismatch for L=%d: got %d, expected %d\n",
                L, method_code, method_code_for_L(L));
        return -1;
    }

    if (method_code == 0) {
        if (pbc) {
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/PBC/gap_L%02d.dat", L);
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/PBC/gapL%02d.dat", L);
        } else {
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/OBC/gap_obc_L%02d.dat", L);
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/OBC/gapobcL%02d.dat", L);
        }
    } else {
        if (pbc) {
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/PBC/gap_lz_L%02d.dat", L);
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/PBC/gaplzL%02d.dat", L);
        } else {
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/OBC/gap_lz_obc_L%02d.dat", L);
            snprintf(candidates[n_candidates++], sizeof(candidates[0]),
                     "data/h_null/observables/OBC/gaplzobcL%02d.dat", L);
        }
    }

    for (int i = 0; i < n_candidates; i++) {
        if (file_exists_readable(candidates[i])) {
            int n = load_first_column(candidates[i], g_arr, max_pts);
            if (n <= 0)
                return -1;
            snprintf(source_path, source_path_sz, "%s", candidates[i]);
            return n;
        }
    }

    fprintf(stderr,
            "[ising_chiz_fd] missing g-grid source for L=%d pbc=%d method_code=%d.\n",
            L, pbc, method_code);
    fprintf(stderr, "[ising_chiz_fd] tried:");
    for (int i = 0; i < n_candidates; i++)
        fprintf(stderr, " %s", candidates[i]);
    fprintf(stderr, "\n");
    return -1;
}

static void build_output_path(const char *out_dir, int pbc, int L,
                              char *path, size_t path_sz)
{
    if (pbc)
        snprintf(path, path_sz, "%s/PBC/chizfd_L%02d.dat", out_dir, L);
    else
        snprintf(path, path_sz, "%s/OBC/chizfd_obc_L%02d.dat", out_dir, L);
}

typedef enum {
    OUTPUT_NEW = 0,
    OUTPUT_APPEND = 1,
    OUTPUT_COMPLETE = 2,
    OUTPUT_RESTART = 3
} OutputStatus;

static const char *bc_label(int pbc)
{
    return pbc ? "PBC" : "OBC";
}

static int current_run_is_smoke(const Params *p)
{
    return p->smoke_max_g_points > 0 || p->smoke_has_g_window;
}

static int flush_data_file(FILE *fp, const char *path, int use_fsync)
{
    if (fflush(fp) != 0) {
        fprintf(stderr, "[ising_chiz_fd] fflush failed for %s: %s\n",
                path, strerror(errno));
        return -1;
    }

#ifndef _WIN32
    if (use_fsync) {
        int fd = fileno(fp);
        if (fd < 0) {
            fprintf(stderr, "[ising_chiz_fd] fileno failed for %s\n", path);
            return -1;
        }
        if (fsync(fd) != 0) {
            fprintf(stderr, "[ising_chiz_fd] fsync failed for %s: %s\n",
                    path, strerror(errno));
            return -1;
        }
    }
#else
    (void)use_fsync;
#endif

    return 0;
}

static int backup_existing_output(const char *path, char *backup_path, size_t backup_sz)
{
    time_t now = time(NULL);
    struct tm tm_now;
    char stamp[64];

#ifdef _WIN32
    tm_now = *localtime(&now);
#else
    if (localtime_r(&now, &tm_now) == NULL) {
        fprintf(stderr, "[ising_chiz_fd] localtime_r failed while backing up %s\n", path);
        return -1;
    }
#endif
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_now);
    snprintf(backup_path, backup_sz, "%s.bad_%s", path, stamp);

    if (rename(path, backup_path) != 0) {
        fprintf(stderr, "[ising_chiz_fd] cannot backup %s -> %s: %s\n",
                path, backup_path, strerror(errno));
        return -1;
    }
    return 0;
}

static int inspect_existing_output(const char *out_path,
                                   const double *g_grid,
                                   int n_g,
                                   int expected_method_code,
                                   int production_run,
                                   int *n_done,
                                   char *reason,
                                   size_t reason_sz)
{
    FILE *fp = fopen(out_path, "r");
    char line[4096];
    int row = 0;
    int line_no = 0;
    int has_smoke_header = 0;

    *n_done = 0;
    snprintf(reason, reason_sz, "new file");

    if (!fp) {
        if (errno == ENOENT)
            return OUTPUT_NEW;
        snprintf(reason, reason_sz, "cannot open existing file: %s", strerror(errno));
        return OUTPUT_RESTART;
    }

    while (fgets(line, sizeof(line), fp)) {
        char *s = line;
        double g, dh, mz_m2, mz_m1, mz_p1, mz_p2, chi_fd, oddness1, oddness2;
        int method_code;
        int used = 0;
        int has_newline = strchr(line, '\n') != NULL;

        line_no++;
        while (isspace((unsigned char)*s))
            s++;
        if (*s == '\0')
            continue;
        if (*s == '#') {
            if (strstr(s, "smoke_max_g_points") || strstr(s, "smoke_g_window"))
                has_smoke_header = 1;
            continue;
        }

        if (row >= n_g) {
            snprintf(reason, reason_sz,
                     "too many data rows (line %d exceeds n_g=%d)", line_no, n_g);
            fclose(fp);
            return OUTPUT_RESTART;
        }
        if (!has_newline) {
            snprintf(reason, reason_sz,
                     "data row without terminating newline at line %d", line_no);
            fclose(fp);
            return OUTPUT_RESTART;
        }

        int nf = sscanf(s,
                        "%lf %lf %d %lf %lf %lf %lf %lf %lf %lf %n",
                        &g, &dh, &method_code, &mz_m2, &mz_m1,
                        &mz_p1, &mz_p2, &chi_fd, &oddness1, &oddness2,
                        &used);
        if (nf != 10) {
            snprintf(reason, reason_sz, "malformed data row at line %d", line_no);
            fclose(fp);
            return OUTPUT_RESTART;
        }
        s += used;
        while (isspace((unsigned char)*s))
            s++;
        if (*s != '\0') {
            snprintf(reason, reason_sz, "extra tokens in data row at line %d", line_no);
            fclose(fp);
            return OUTPUT_RESTART;
        }

        if (fabs(g - g_grid[row]) > 1e-10) {
            snprintf(reason, reason_sz,
                     "g mismatch at row %d: have %.17g expected %.17g",
                     row + 1, g, g_grid[row]);
            fclose(fp);
            return OUTPUT_RESTART;
        }
        if (method_code != expected_method_code) {
            snprintf(reason, reason_sz,
                     "method_code mismatch at row %d: have %d expected %d",
                     row + 1, method_code, expected_method_code);
            fclose(fp);
            return OUTPUT_RESTART;
        }
        if (!isfinite(chi_fd) || chi_fd <= 0.0) {
            snprintf(reason, reason_sz,
                     "invalid chi_fd at row %d: %.17g", row + 1, chi_fd);
            fclose(fp);
            return OUTPUT_RESTART;
        }
        row++;
    }

    if (ferror(fp)) {
        snprintf(reason, reason_sz, "read error in existing file");
        fclose(fp);
        return OUTPUT_RESTART;
    }
    fclose(fp);

    if (production_run && has_smoke_header) {
        snprintf(reason, reason_sz, "existing file has smoke header");
        return OUTPUT_RESTART;
    }

    *n_done = row;
    if (row == n_g) {
        snprintf(reason, reason_sz, "already complete (%d/%d rows)", row, n_g);
        return OUTPUT_COMPLETE;
    }

    snprintf(reason, reason_sz, "valid partial file (%d/%d rows)", row, n_g);
    return OUTPUT_APPEND;
}

static int write_data_row(FILE *fp, const char *path, const Params *p,
                          double g, int method_code, const ChizRow *r)
{
    if (fprintf(fp,
                "%.8f %.12e %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                g, p->dh, method_code,
                r->mz_m2, r->mz_m1, r->mz_p1, r->mz_p2,
                r->chi_fd, r->oddness1, r->oddness2) < 0) {
        fprintf(stderr, "[ising_chiz_fd] write failed for %s\n", path);
        return -1;
    }
    return flush_data_file(fp, path, p->use_fsync);
}

static void print_progress(int L, int pbc, int ig, int n_g,
                           double g, double chi_fd, time_t t0)
{
    printf("[chi_z_fd] L=%d %s point %d/%d g=%.8f chi_fd=%.12e elapsed=%.1fs\n",
           L, bc_label(pbc), ig + 1, n_g, g, chi_fd,
           difftime(time(NULL), t0));
    fflush(stdout);
}

static int diagonalize_ground_state(double *Ham, long long dim, double *evec)
{
    lapack_int n = (lapack_int)dim;
    lapack_int m = 0;
    lapack_int isuppz[2];
    double eval[1];
    lapack_int info = LAPACKE_dsyevr(
        LAPACK_COL_MAJOR,
        'V',
        'I',
        'U',
        n,
        Ham,
        n,
        0.0,
        0.0,
        1,
        1,
        0.0,
        &m,
        eval,
        evec,
        n,
        isuppz
    );

    if (info != 0 || m != 1) {
        fprintf(stderr,
                "[ising_chiz_fd] LAPACK dsyevr ground-state solve failed: info=%d m=%d\n",
                (int)info, (int)m);
        return -1;
    }

    return 0;
}

static double mz_ed(const Basis *b, double g, double h,
                    double *Ham, double *evec)
{
    const long long dim = b->dim;
    memset(Ham, 0, (size_t)dim * (size_t)dim * sizeof(double));
    build_ham(b, g, h, Ham);
    if (diagonalize_ground_state(Ham, dim, evec) != 0)
        return NAN;

    return obs_mz_raw(b, evec);
}

static double mz_lanczos(const Basis *b, double g, double h,
                         int max_iter, unsigned long seed, double *evec)
{
    LanczosParams par = {
        .n_eig = 1,
        .max_iter = max_iter,
        .tol = 1e-10,
        .max_restarts = 80,
        .verbose = 0,
        .seed = seed,
    };
    double e0 = NAN;
    int ret = lanczos(b, g, h, &par, &e0, evec);
    if (ret != 0 && ret != -3) {
        fprintf(stderr,
                "[ising_chiz_fd] Lanczos failed: L=%d pbc=%d g=%.8f h=%+.6e ret=%d\n",
                b->L, b->pbc, g, h, ret);
        return NAN;
    }
    return obs_mz_raw(b, evec);
}

static ChizRow compute_row_ed(const Basis *b, double g, double dh,
                              double *Ham, double *evec)
{
    ChizRow r;
    r.mz_p1 = mz_ed(b, g,  dh,       Ham, evec);
    r.mz_p2 = mz_ed(b, g,  2.0 * dh, Ham, evec);
    /* Global spin-flip symmetry gives signed m_z(g,-h) = -m_z(g,+h). */
    r.mz_m1 = -r.mz_p1;
    r.mz_m2 = -r.mz_p2;
    r.chi_fd = (-r.mz_p2 + 8.0 * r.mz_p1 - 8.0 * r.mz_m1 + r.mz_m2) /
               (12.0 * dh);
    r.oddness1 = fabs(r.mz_p1 + r.mz_m1);
    r.oddness2 = fabs(r.mz_p2 + r.mz_m2);
    return r;
}

static ChizRow compute_row_lanczos(const Basis *b, double g, double dh,
                                   int max_iter, unsigned long seed,
                                   double *evec)
{
    ChizRow r;
    r.mz_p1 = mz_lanczos(b, g,  dh,       max_iter, seed, evec);
    r.mz_p2 = mz_lanczos(b, g,  2.0 * dh, max_iter, seed, evec);
    /* Global spin-flip symmetry gives signed m_z(g,-h) = -m_z(g,+h). */
    r.mz_m1 = -r.mz_p1;
    r.mz_m2 = -r.mz_p2;
    r.chi_fd = (-r.mz_p2 + 8.0 * r.mz_p1 - 8.0 * r.mz_m1 + r.mz_m2) /
               (12.0 * dh);
    r.oddness1 = fabs(r.mz_p1 + r.mz_m1);
    r.oddness2 = fabs(r.mz_p2 + r.mz_m2);
    return r;
}

static void warn_row(int L, int pbc, double g, const ChizRow *r)
{
    if (!isfinite(r->chi_fd)) {
        fprintf(stderr,
                "[WARN chiz_fd] non-finite chi_fd: L=%d pbc=%d g=%.8f\n",
                L, pbc, g);
    } else if (r->chi_fd < 0.0) {
        fprintf(stderr,
                "[WARN chiz_fd] negative chi_fd: L=%d pbc=%d g=%.8f chi=%.12e\n",
                L, pbc, g, r->chi_fd);
    }

    if (isfinite(r->oddness1) && r->oddness1 > 1e-6) {
        fprintf(stderr,
                "[WARN chiz_fd] oddness1>1e-6: L=%d pbc=%d g=%.8f odd1=%.12e\n",
                L, pbc, g, r->oddness1);
    }
    if (isfinite(r->oddness2) && r->oddness2 > 1e-6) {
        fprintf(stderr,
                "[WARN chiz_fd] oddness2>1e-6: L=%d pbc=%d g=%.8f odd2=%.12e\n",
                L, pbc, g, r->oddness2);
    }
}

static int write_header(FILE *fp, int L, int pbc, double dh, int method_code,
                        const char *g_grid_source, int n_g,
                        int smoke_max_g_points,
                        int smoke_has_g_window,
                        double smoke_g_min,
                        double smoke_g_max)
{
    if (fprintf(fp,
        "# chi_z finite-difference pipeline\n"
        "# dh = %.12e\n"
        "# L = %d\n"
        "# BC = %s\n"
        "# method_code = %d\n"
        "# chi_fd = d <Mz/L> / dh at h=0\n"
        "# stencil = [-m(+2dh)+8m(+dh)-8m(-dh)+m(-2dh)]/(12dh)\n"
        "# method_code = 0 ED for L<=12, 1 Lanczos for L>=14\n"
        "# g_grid_source = %s\n"
        "# n_g = %d\n",
        dh, L, bc_label(pbc), method_code, g_grid_source, n_g) < 0)
        return -1;

    if (smoke_max_g_points > 0 &&
        fprintf(fp, "# smoke_max_g_points = %d\n", smoke_max_g_points) < 0)
        return -1;

    if (smoke_has_g_window &&
        fprintf(fp, "# smoke_g_window = %.12g %.12g\n", smoke_g_min, smoke_g_max) < 0)
        return -1;

    return fprintf(fp,
        "# NOT psi_tilde, NOT psi_bar, NOT sqrt(mz_sq), NOT observables COL[\"chi_z\"]\n"
        "# columns:\n"
        "# g dh method_code mz_m2 mz_m1 mz_p1 mz_p2 chi_fd oddness1 oddness2\n") < 0
        ? -1 : 0;
}

static int run_L(const Params *p, int L, const char *out_dir)
{
    Basis b = basis_init(L, p->pbc);
    const long long dim = b.dim;
    const int method_code = method_code_for_L(L);
    double g_arr[MAX_G_POINTS];
    char g_grid_source[512];
    int N_g = load_existing_g_grid(L, p->pbc, method_code, g_arr, MAX_G_POINTS,
                                   g_grid_source, sizeof(g_grid_source));
    char path[512];
    char reason[512];
    int n_done = 0;
    OutputStatus status;
    FILE *fp = NULL;
    time_t t_start = time(NULL);

    if (N_g <= 0)
        return 1;
    int N_g_source = N_g;

    if (p->smoke_has_g_window) {
        int n_keep = 0;
        for (int ig = 0; ig < N_g; ig++) {
            double g = g_arr[ig];
            if (g >= p->smoke_g_min && g <= p->smoke_g_max)
                g_arr[n_keep++] = g;
        }
        N_g = n_keep;
        if (N_g <= 0) {
            fprintf(stderr,
                    "[ising_chiz_fd] --g-window %.12g %.12g selected no g points for L=%d pbc=%d\n",
                    p->smoke_g_min, p->smoke_g_max, L, p->pbc);
            return 1;
        }
    }

    if (p->smoke_max_g_points > 0 && p->smoke_max_g_points < N_g)
        N_g = p->smoke_max_g_points;

    build_output_path(out_dir, p->pbc, L, path, sizeof(path));

    status = inspect_existing_output(path, g_arr, N_g, method_code,
                                     !current_run_is_smoke(p),
                                     &n_done, reason, sizeof(reason));
    if (status == OUTPUT_COMPLETE) {
        printf("[chi_z_fd] L=%d %s already complete, skipping: %s (%s)\n",
               L, bc_label(p->pbc), path, reason);
        fflush(stdout);
        return 0;
    }
    if (status == OUTPUT_RESTART) {
        char backup[640];
        if (file_exists_readable(path)) {
            if (backup_existing_output(path, backup, sizeof(backup)) != 0)
                return 1;
            printf("[chi_z_fd] L=%d %s restarting output: %s; backup=%s\n",
                   L, bc_label(p->pbc), reason, backup);
        } else {
            printf("[chi_z_fd] L=%d %s restarting output: %s\n",
                   L, bc_label(p->pbc), reason);
        }
        fflush(stdout);
        n_done = 0;
    }

    fp = fopen(path, n_done > 0 ? "a" : "w");
    if (!fp) {
        fprintf(stderr, "[ising_chiz_fd] cannot open %s: %s\n",
                path, strerror(errno));
        return 1;
    }

    if (n_done == 0) {
        if (write_header(fp, L, p->pbc, p->dh, method_code,
                         g_grid_source, N_g, p->smoke_max_g_points,
                         p->smoke_has_g_window, p->smoke_g_min, p->smoke_g_max) != 0 ||
            flush_data_file(fp, path, p->use_fsync) != 0) {
            fclose(fp);
            return 1;
        }
    }

    printf("L=%2d pbc=%d method_code=%d (%s) g-pts=%d",
           L, p->pbc, method_code, method_code == 0 ? "ED" : "Lanczos",
           N_g);
    if (N_g != N_g_source)
        printf(" source-pts=%d", N_g_source);
    printf(" source=%s -> %s\n", g_grid_source, path);
    if (n_done > 0)
        printf("[chi_z_fd] L=%d %s resuming from point %d/%d\n",
               L, bc_label(p->pbc), n_done + 1, N_g);
    if (!p->use_fsync)
        printf("[chi_z_fd] L=%d %s fsync disabled by --no-fsync\n",
               L, bc_label(p->pbc));
    if (p->stop_after > 0)
        printf("[chi_z_fd] L=%d %s diagnostic stop-after=%d completed points in this invocation\n",
               L, bc_label(p->pbc), p->stop_after);
    fflush(stdout);

    if (method_code == 0) {
        double *Ham = calloc((size_t)dim * (size_t)dim, sizeof(double));
        double *evec = malloc((size_t)dim * sizeof(double));
        if (!Ham || !evec) {
            fprintf(stderr, "[ising_chiz_fd] ED allocation failed for L=%d\n", L);
            free(Ham);
            free(evec);
            fclose(fp);
            return 1;
        }

        int processed = 0;
        for (int ig = n_done; ig < N_g; ig++) {
            double g = g_arr[ig];
            ChizRow r = compute_row_ed(&b, g, p->dh, Ham, evec);
            warn_row(L, p->pbc, g, &r);
            if (write_data_row(fp, path, p, g, method_code, &r) != 0) {
                free(Ham);
                free(evec);
                fclose(fp);
                return 1;
            }
            print_progress(L, p->pbc, ig, N_g, g, r.chi_fd, t_start);
            processed++;
            if (p->stop_after > 0 && processed >= p->stop_after) {
                printf("[chi_z_fd] L=%d %s stop-after reached after %d points; partial file kept at %s\n",
                       L, bc_label(p->pbc), processed, path);
                fflush(stdout);
                break;
            }
        }

        free(Ham);
        free(evec);
    } else {
        int max_iter = max_iter_for_L(L);
        double *evec = malloc((size_t)dim * sizeof(double));
        if (!evec) {
            fprintf(stderr, "[ising_chiz_fd] Lanczos evec allocation failed for L=%d\n", L);
            fclose(fp);
            return 1;
        }

        int processed = 0;
        for (int ig = n_done; ig < N_g; ig++) {
            double g = g_arr[ig];
            unsigned long seed = 42UL + (unsigned long)L * 1000003UL +
                                 (unsigned long)ig * 7919UL;
            ChizRow r = compute_row_lanczos(&b, g, p->dh, max_iter, seed, evec);
            warn_row(L, p->pbc, g, &r);
            if (write_data_row(fp, path, p, g, method_code, &r) != 0) {
                free(evec);
                fclose(fp);
                return 1;
            }
            print_progress(L, p->pbc, ig, N_g, g, r.chi_fd, t_start);
            processed++;
            if (p->stop_after > 0 && processed >= p->stop_after) {
                printf("[chi_z_fd] L=%d %s stop-after reached after %d points; partial file kept at %s\n",
                       L, bc_label(p->pbc), processed, path);
                fflush(stdout);
                break;
            }
        }

        free(evec);
    }

    if (fclose(fp) != 0)
        return 1;

    return 0;
}

int main(int argc, char **argv)
{
    Params p = parse_args(argc, argv);
    char out_dir[512];
    time_t t0 = time(NULL);

    if (p.has_output_dir) {
        snprintf(out_dir, sizeof(out_dir), "%s", p.output_dir);
        if (ensure_dir(out_dir) != 0)
            return EXIT_FAILURE;
        if (ensure_chiz_bc_dirs(out_dir) != 0)
            return EXIT_FAILURE;
    } else {
        if (build_output_dir(p.dh, out_dir, sizeof(out_dir)) != 0)
            return EXIT_FAILURE;
    }

    printf("=====================================================\n");
    printf("|  chi_z finite-difference pipeline                 |\n");
    printf("=====================================================\n");
    printf(" pbc      = %s\n", p.pbc ? "PBC" : "OBC");
    printf(" dh       = %.12e\n", p.dh);
    printf(" out_dir  = %s\n", out_dir);
    printf(" fsync    = %s\n", p.use_fsync ? "enabled" : "disabled");
    printf(" g grid   = loaded from existing data/h_null/observables gap files\n");
    if (p.smoke_max_g_points > 0)
        printf(" smoke    = first %d g-points only\n", p.smoke_max_g_points);
    if (p.smoke_has_g_window)
        printf(" smoke    = g window [%.12g, %.12g]\n", p.smoke_g_min, p.smoke_g_max);
    if (p.stop_after > 0)
        printf(" stop     = after %d completed points per L\n", p.stop_after);
    printf(" method   = L<=12 ED (0), L>=14 Lanczos (1)\n");
    printf(" L values =");
    for (int i = 0; i < p.N_L; i++)
        printf(" %d", p.L_values[i]);
    printf("\n-----------------------------------------------------\n");
    fflush(stdout);

    for (int i = 0; i < p.N_L; i++) {
        int L = p.L_values[i];
        if (run_L(&p, L, out_dir) != 0)
            return EXIT_FAILURE;
    }

    printf("=====================================================\n");
    printf("Done in %.1f s\n", difftime(time(NULL), t0));
    fflush(stdout);
    return EXIT_SUCCESS;
}
