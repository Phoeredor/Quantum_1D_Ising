#define _POSIX_C_SOURCE 200809L

/*
 * Longitudinal-field generator for the 1D quantum Ising chain.
 *
 * Hamiltonian convention:
 *   H = - sum_j sigma^z_j sigma^z_{j+1}
 *       - g * sum_j sigma^x_j
 *       - h * sum_j sigma^z_j
 *
 * Magnetization convention:
 *   mz = <sum_j sigma^z_j>/L is the longitudinal order-parameter
 *        magnetization coupled to h.
 *   mx = <sum_j sigma^x_j>/L is the transverse magnetization coupled to g.
 *   Both are computed directly from the ground-state eigenvector.  Any
 *   Hellmann-Feynman derivative check belongs in validation scripts, not in
 *   this production generator.
 *
 * Usage:
 *   ./ising_hfield <g> <pbc> [options]
 *
 * Raw output:
 *   PBC: data/h_field/hfield_raw/hfield_pbc_g<G>_L<LL>.dat
 *   OBC: data/h_field/hfield_raw/hfield_obc_g<G>_L<LL>.dat
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "basis.h"
#include "diag.h"
#include "hamiltonian.h"
#include "lanczos.h"
#include "observables.h"

#define ED_L_MAX 12
#define MAX_L_VALUE 24
#define MAX_L_RUNS 32
#define MAX_GRID_POINTS 4096
#define MAX_PATH_LEN 1024
#define MAX_COMMAND_LEN 2048
#define PRIMARY_TOL 1e-10

typedef enum {
    MODE_AUTO = 0,
    MODE_CQT,
    MODE_FOQT
} RunMode;

typedef struct {
    double g;
    int pbc;
    RunMode requested_mode;
    RunMode mode;
    int L_values[MAX_L_RUNS];
    int N_L;
    int yh_set;
    double yh;
    int xmax_set;
    double xmax;
    double dx_near;
    double dx_mid;
    double dx_far;
    unsigned long seed;
    bool overwrite;
    bool resume;
    bool force_unlock;
    bool max_ram_set;
    double max_ram_gb;
    int stop_after;
    char command[MAX_COMMAND_LEN];
} Params;

typedef struct {
    double h;
    double scale_x;
    double kappa;
} GridPoint;

typedef struct {
    double evals[4];
    double mz;
    double abs_mz;
    double mx;
    double resid[4];
    int method_code;
    int lanczos_ret;
} SolveResult;

typedef struct {
    double h;
    double scale_x;
    double kappa;
    double evals[4];
    double delta_h;
    double delta0_h0;
    double mz;
    double abs_mz;
    double mx;
    int method_code;
    double resid[4];
} DataRow;

typedef struct {
    bool held;
    char path[MAX_PATH_LEN];
} LockFile;

typedef struct {
    int completed_valid_rows;
    int skipped_rows_this_run;
    int newly_computed_rows_this_run;
    double min_delta_h;
    double max_delta_h;
    double min_mz;
    double max_mz;
} RunStats;

static const int DEFAULT_L[] = {4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
static const int N_DEFAULT_L = (int)(sizeof(DEFAULT_L) / sizeof(DEFAULT_L[0]));

static const double ZONE_X1[] = {0.0, 1.5, 5.0};
static const double ZONE_X2[] = {1.5, 5.0, 0.0}; /* third endpoint is xmax */
static const int N_ZONES = 3;

static const char *EXPECTED_COLUMNS =
    "h scale_x kappa E0 E1 E2 E3 delta_h delta0_h0 "
    "mz abs_mz mx method_code resid0 resid1 resid2 resid3";

#ifndef _WIN32
static volatile sig_atomic_t g_lock_held = 0;
static char g_active_lock_path[MAX_PATH_LEN];
#endif

static double parse_double(const char *s, const char *name);
static int parse_int(const char *s, const char *name);
static unsigned long parse_ulong(const char *s, const char *name);
static int parse_pbc(const char *s);
static Params parse_args(int argc, char **argv);
static void build_command_line(int argc, char **argv, char *buf, size_t n);
static void resolve_mode_and_defaults(Params *P);
static const char *mode_name(RunMode mode);
static int ensure_output_dirs(void);
static bool path_exists(const char *path);
static void make_g_tag(double g, char *buf, size_t n);
static void build_output_path(const Params *P, int L, char *path, size_t n);
static void build_checkpoint_path(const char *path, char *ckpt, size_t n);
static void build_lock_path(const char *path, char *lock_path, size_t n);
static int precheck_outputs(const Params *P);
static int build_variable_grid(double xmax, double dx_near, double dx_mid,
                               double dx_far, double *x_arr, int max_pts,
                               int *n_out);
static int build_h_grid(const Params *P, int L, double delta0_h0,
                        double m0, GridPoint *grid, int max_pts,
                        int *n_out);
static int compare_double(const void *a, const void *b);
static int solve_low_energy(const Basis *b, double g, double h,
                            unsigned long seed, SolveResult *out);
static int solve_ed(const Basis *b, double g, double h, SolveResult *out);
static int solve_lanczos(const Basis *b, double g, double h,
                         unsigned long seed, SolveResult *out);
static int lanczos_max_iter(int L);
static double compute_delta0_h0(const Basis *b, double g, unsigned long seed,
                                int *method_code);
static int write_one_L(const Params *P, int L, int *new_rows_total);
static int write_header(FILE *fp, const Params *P, int L, int N_h,
                        double delta0_h0, double m0, const char *method);
static int write_new_output_header(const char *path, const Params *P, int L,
                                   int N_h, double delta0_h0, double m0,
                                   const char *method);
static int append_data_row(const char *path, const GridPoint *gp,
                           const SolveResult *r, double delta_h,
                           double delta0_h0);
static int write_checkpoint(const char *path, const Params *P, int L,
                            int total_points, double delta0_h0, double m0,
                            const RunStats *stats,
                            double last_primary, double last_h,
                            double run_t0, const char *primary_name,
                            double estimated_remaining_seconds);
static int fsync_stream(FILE *fp, const char *path);
static int acquire_lock(const char *output_path, const Params *P,
                        LockFile *lock);
static int release_lock(LockFile *lock);
static int remove_existing_outputs(const char *path);
static int verify_header_compatible(const char *path, const Params *P, int L,
                                    int N_h, double delta0_h0, double m0,
                                    const char *method);
static int parse_existing_rows(const char *path, const Params *P,
                               const GridPoint *grid, int N_h,
                               bool *completed, RunStats *stats);
static int truncate_file_at(const char *path, long offset);
static bool parse_data_row(const char *line, DataRow *row);
static bool parse_double_field(const char **p, double *out);
static bool parse_int_field(const char **p, int *out);
static int match_grid_index(const Params *P, const GridPoint *grid, int N_h,
                            double primary);
static double primary_value(const Params *P, const GridPoint *gp);
static const char *primary_name(const Params *P);
static void init_run_stats(RunStats *stats);
static void update_ranges(RunStats *stats, double delta_h, double mz);
static bool ranges_valid(const RunStats *stats);
static void format_range(double min_v, double max_v, char *buf, size_t n);
static void format_timestamp(time_t t, char *buf, size_t n);
static void get_hostname_string(char *buf, size_t n);
static double now_seconds(void);
static double estimate_peak_memory_gb(int L);
static void print_allocation_failure(const char *where, const Basis *b,
                                     double g, double h);
static bool double_close(double a, double b, double abs_tol, double rel_tol);
static bool header_has_obsolete_field(const char *line);
#ifndef _WIN32
static void handle_signal(int sig);
static int install_signal_handlers(void);
#endif

int main(int argc, char **argv)
{
    Params P = parse_args(argc, argv);
    resolve_mode_and_defaults(&P);

#ifndef _WIN32
    if (install_signal_handlers() != EXIT_SUCCESS)
        return EXIT_FAILURE;
#endif

    if (ensure_output_dirs() != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (precheck_outputs(&P) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    printf("=====================================================\n");
    printf("|  1D QUANTUM ISING - LONGITUDINAL FIELD GENERATOR  |\n");
    printf("=====================================================\n");
    printf(" convention = H=-sum zz - g sum sx - h sum sz\n");
    printf(" g          = %.17g\n", P.g);
    printf(" pbc        = %s\n", P.pbc ? "PBC" : "OBC");
    printf(" mode       = %s\n", mode_name(P.mode));
    printf(" yh         = ");
    if (P.mode == MODE_CQT)
        printf("%.17g\n", P.yh);
    else
        printf("NaN\n");
    printf(" xmax       = %.17g\n", P.xmax);
    printf(" dx         = near %.17g  mid %.17g  far %.17g\n",
           P.dx_near, P.dx_mid, P.dx_far);
    printf(" seed       = %lu\n", P.seed);
    printf(" overwrite  = %s\n", P.overwrite ? "yes" : "no");
    printf(" resume     = %s\n", P.resume ? "yes" : "no");
    if (P.max_ram_set)
        printf(" max RAM    = %.3f GiB\n", P.max_ram_gb);
    if (P.stop_after >= 0)
        printf(" stop-after = %d newly computed rows\n", P.stop_after);
    printf(" L values   =");
    for (int i = 0; i < P.N_L; i++)
        printf(" %d", P.L_values[i]);
    printf("\n");
    printf("-----------------------------------------------------\n");
    fflush(stdout);

    int new_rows_total = 0;
    for (int i = 0; i < P.N_L; i++) {
        if (write_one_L(&P, P.L_values[i], &new_rows_total) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }

    printf(">> h-field generation completed.\n");
    fflush(stdout);
    return EXIT_SUCCESS;
}

static double parse_double(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (!s[0] || *end || errno || !isfinite(v)) {
        fprintf(stderr, "[ERROR] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_int(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || errno || v < -2147483647L || v > 2147483647L) {
        fprintf(stderr, "[ERROR] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static unsigned long parse_ulong(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (!s[0] || *end || errno) {
        fprintf(stderr, "[ERROR] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_pbc(const char *s)
{
    if (strcmp(s, "0") == 0)
        return 0;
    if (strcmp(s, "1") == 0)
        return 1;
    fprintf(stderr, "[ERROR] <pbc> must be 0 (OBC) or 1 (PBC)\n");
    exit(EXIT_FAILURE);
}

static Params parse_args(int argc, char **argv)
{
    Params P;
    memset(&P, 0, sizeof(P));
    P.requested_mode = MODE_AUTO;
    P.mode = MODE_AUTO;
    P.dx_near = 0.02;
    P.dx_mid = 0.10;
    P.dx_far = 0.50;
    P.seed = 42UL;
    P.stop_after = -1;
    build_command_line(argc, argv, P.command, sizeof(P.command));

    if (argc < 3) {
        fprintf(stderr,
                "\nUsage: %s <g> <pbc> [options]\n\n"
                "Required:\n"
                "  g      transverse field in H=-sum zz - g sum sx - h sum sz\n"
                "  pbc    1=PBC, 0=OBC\n\n"
                "Options:\n"
                "  --mode cqt|foqt|auto\n"
                "  --L L1 L2 ...\n"
                "  --yh YH_VALUE        required for CQT\n"
                "  --xmax XMAX\n"
                "  --dx-near DX_NEAR\n"
                "  --dx-mid DX_MID\n"
                "  --dx-far DX_FAR\n"
                "  --seed SEED\n"
                "  --overwrite\n"
                "  --resume\n"
                "  --force-unlock\n"
                "  --max-ram-gb VALUE\n"
                "  --stop-after N\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    P.g = parse_double(argv[1], "g");
    P.pbc = parse_pbc(argv[2]);

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --mode requires cqt, foqt, or auto\n");
                exit(EXIT_FAILURE);
            }
            if (strcmp(argv[i], "cqt") == 0)
                P.requested_mode = MODE_CQT;
            else if (strcmp(argv[i], "foqt") == 0)
                P.requested_mode = MODE_FOQT;
            else if (strcmp(argv[i], "auto") == 0)
                P.requested_mode = MODE_AUTO;
            else {
                fprintf(stderr, "[ERROR] invalid --mode '%s'\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--L") == 0) {
            int n_before = P.N_L;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                if (P.N_L >= MAX_L_RUNS) {
                    fprintf(stderr, "[ERROR] too many L values (max %d)\n",
                            MAX_L_RUNS);
                    exit(EXIT_FAILURE);
                }
                int L = parse_int(argv[++i], "L");
                if (L < 2 || L > MAX_L_VALUE) {
                    fprintf(stderr,
                            "[ERROR] L=%d out of supported range [2,%d]\n",
                            L, MAX_L_VALUE);
                    exit(EXIT_FAILURE);
                }
                P.L_values[P.N_L++] = L;
            }
            if (P.N_L == n_before) {
                fprintf(stderr, "[ERROR] --L requires at least one L value\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--yh") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --yh requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.yh = parse_double(argv[i], "yh");
            P.yh_set = 1;
        } else if (strcmp(argv[i], "--xmax") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --xmax requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.xmax = parse_double(argv[i], "xmax");
            P.xmax_set = 1;
        } else if (strcmp(argv[i], "--dx-near") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --dx-near requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.dx_near = parse_double(argv[i], "dx-near");
        } else if (strcmp(argv[i], "--dx-mid") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --dx-mid requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.dx_mid = parse_double(argv[i], "dx-mid");
        } else if (strcmp(argv[i], "--dx-far") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --dx-far requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.dx_far = parse_double(argv[i], "dx-far");
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --seed requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.seed = parse_ulong(argv[i], "seed");
        } else if (strcmp(argv[i], "--max-ram-gb") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --max-ram-gb requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.max_ram_gb = parse_double(argv[i], "max-ram-gb");
            P.max_ram_set = true;
        } else if (strcmp(argv[i], "--stop-after") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --stop-after requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.stop_after = parse_int(argv[i], "stop-after");
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            P.overwrite = true;
        } else if (strcmp(argv[i], "--resume") == 0) {
            P.resume = true;
        } else if (strcmp(argv[i], "--force-unlock") == 0) {
            P.force_unlock = true;
        } else {
            fprintf(stderr, "[ERROR] unknown option '%s'\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    if (P.N_L == 0) {
        for (int i = 0; i < N_DEFAULT_L; i++)
            P.L_values[P.N_L++] = DEFAULT_L[i];
    }

    return P;
}

static void build_command_line(int argc, char **argv, char *buf, size_t n)
{
    size_t used = 0;
    if (n == 0)
        return;
    buf[0] = '\0';
    for (int i = 0; i < argc; i++) {
        int written = snprintf(buf + used, n - used, "%s%s",
                               (i == 0) ? "" : " ", argv[i]);
        if (written < 0)
            break;
        if ((size_t)written >= n - used) {
            buf[n - 1] = '\0';
            break;
        }
        used += (size_t)written;
    }
}

static void resolve_mode_and_defaults(Params *P)
{
    if (P->overwrite && P->resume) {
        fprintf(stderr, "[ERROR] --overwrite and --resume are mutually exclusive\n");
        exit(EXIT_FAILURE);
    }
    if (P->dx_near <= 0.0 || P->dx_mid <= 0.0 || P->dx_far <= 0.0) {
        fprintf(stderr, "[ERROR] grid spacings must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (P->max_ram_set && P->max_ram_gb <= 0.0) {
        fprintf(stderr, "[ERROR] --max-ram-gb must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (P->stop_after < -1) {
        fprintf(stderr, "[ERROR] --stop-after must be nonnegative\n");
        exit(EXIT_FAILURE);
    }

    if (P->requested_mode == MODE_AUTO) {
        if (fabs(P->g - 1.0) < 0.05) {
            if (!P->yh_set) {
                fprintf(stderr,
                        "[ERROR] auto selected the CQT region (|g-1|<0.05), "
                        "but --yh is missing. The Python driver must pass the "
                        "BC-specific y_h from data/h_null/fss/fss_constants.json.\n");
                exit(EXIT_FAILURE);
            }
            P->mode = MODE_CQT;
        } else if (P->g < 1.0) {
            P->mode = MODE_FOQT;
        } else {
            fprintf(stderr,
                    "[ERROR] auto mode has no valid h-field scaling for g=%.17g: "
                    "FOQT is not used for g>=1, and this point is not close "
                    "enough to the critical point for CQT.\n",
                    P->g);
            exit(EXIT_FAILURE);
        }
    } else {
        P->mode = P->requested_mode;
    }

    if (P->mode == MODE_CQT && !P->yh_set) {
        fprintf(stderr,
                "[ERROR] --mode cqt requires --yh. Do not use a hardcoded "
                "y_h; the Python driver must pass the BC-specific y_h from "
                "data/h_null/fss/fss_constants.json.\n");
        exit(EXIT_FAILURE);
    }
    if (P->mode == MODE_FOQT && P->yh_set) {
        fprintf(stderr,
                "[WARN] --yh is ignored in FOQT mode; kappa sets the grid.\n");
    }
    if (P->mode == MODE_FOQT && P->g >= 1.0) {
        fprintf(stderr, "[ERROR] FOQT mode is valid only for g < 1\n");
        exit(EXIT_FAILURE);
    }

    if (!P->xmax_set)
        P->xmax = (P->mode == MODE_CQT) ? 12.0 : 5.0;
    if (P->xmax < 0.0) {
        fprintf(stderr, "[ERROR] --xmax must be nonnegative\n");
        exit(EXIT_FAILURE);
    }
}

static const char *mode_name(RunMode mode)
{
    switch (mode) {
    case MODE_CQT:
        return "cqt";
    case MODE_FOQT:
        return "foqt";
    default:
        return "auto";
    }
}

static int ensure_output_dirs(void)
{
#ifndef _WIN32
    if (mkdir("data", 0775) != 0 && errno != EEXIST) {
        perror("[ERROR] mkdir data");
        return EXIT_FAILURE;
    }
    if (mkdir("data/h_field", 0775) != 0 && errno != EEXIST) {
        perror("[ERROR] mkdir data/h_field");
        return EXIT_FAILURE;
    }
    if (mkdir("data/h_field/hfield_raw", 0775) != 0 && errno != EEXIST) {
        perror("[ERROR] mkdir data/h_field/hfield_raw");
        return EXIT_FAILURE;
    }
    if (mkdir("data/h_field/hfield_raw/.locks", 0775) != 0 && errno != EEXIST) {
        perror("[ERROR] mkdir data/h_field/hfield_raw/.locks");
        return EXIT_FAILURE;
    }
#endif
    return EXIT_SUCCESS;
}

static bool path_exists(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return false;
    fclose(fp);
    return true;
}

static void make_g_tag(double g, char *buf, size_t n)
{
    char raw[96];
    snprintf(raw, sizeof(raw), "g%.12g", g);
    size_t j = 0;
    for (size_t i = 0; raw[i] != '\0' && j + 1 < n; i++) {
        if (raw[i] == '+') {
            continue;
        } else if (raw[i] == '-') {
            buf[j++] = 'm';
        } else {
            buf[j++] = raw[i];
        }
    }
    if (n > 0)
        buf[j] = '\0';
}

static void build_output_path(const Params *P, int L, char *path, size_t n)
{
    char tag[64];
    make_g_tag(P->g, tag, sizeof(tag));
    snprintf(path, n, "data/h_field/hfield_raw/hfield_%s_%s_L%02d.dat",
             P->pbc ? "pbc" : "obc", tag, L);
}

static void build_checkpoint_path(const char *path, char *ckpt, size_t n)
{
    size_t path_len = strlen(path);
    const char *suffix = ".ckpt";
    size_t suffix_len = strlen(suffix);
    if (path_len + suffix_len + 1 > n) {
        if (n > 0)
            ckpt[0] = '\0';
        return;
    }
    memcpy(ckpt, path, path_len);
    memcpy(ckpt + path_len, suffix, suffix_len + 1);
}

static void build_lock_path(const char *path, char *lock_path, size_t n)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *prefix = "data/h_field/hfield_raw/.locks/";
    const char *suffix = ".lock";
    size_t prefix_len = strlen(prefix);
    size_t base_len = strlen(base);
    size_t suffix_len = strlen(suffix);
    if (prefix_len + base_len + suffix_len + 1 > n) {
        if (n > 0)
            lock_path[0] = '\0';
        return;
    }
    memcpy(lock_path, prefix, prefix_len);
    memcpy(lock_path + prefix_len, base, base_len);
    memcpy(lock_path + prefix_len + base_len, suffix, suffix_len + 1);
}

static int precheck_outputs(const Params *P)
{
    if (P->overwrite || P->resume)
        return EXIT_SUCCESS;

    for (int i = 0; i < P->N_L; i++) {
        char path[MAX_PATH_LEN];
        build_output_path(P, P->L_values[i], path, sizeof(path));
        if (path_exists(path)) {
            fprintf(stderr,
                    "[ERROR] output file already exists: %s\n"
                    "        Pass --overwrite to replace it or --resume to continue it.\n",
                    path);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int build_variable_grid(double xmax, double dx_near, double dx_mid,
                               double dx_far, double *x_arr, int max_pts,
                               int *n_out)
{
    double tmp[MAX_GRID_POINTS];
    const double dxs[] = {dx_near, dx_mid, dx_far};
    int n = 0;

    for (int iz = 0; iz < N_ZONES; iz++) {
        double x1 = ZONE_X1[iz];
        double x2 = (iz == 2) ? xmax : ZONE_X2[iz];
        double dx = dxs[iz];
        if (x1 > xmax)
            continue;
        if (x2 > xmax)
            x2 = xmax;
        if (x2 < x1)
            continue;

        for (double x = x1; x <= x2 + 0.5 * dx; x += dx) {
            double x_use = x;
            if (x_use > x2)
                x_use = x2;
            double xs[2] = {x_use, -x_use};
            int nx = (fabs(x_use) < 1e-14) ? 1 : 2;
            for (int s = 0; s < nx; s++) {
                bool dup = false;
                for (int k = 0; k < n; k++) {
                    if (fabs(tmp[k] - xs[s]) < 1e-10) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    if (n >= max_pts) {
                        fprintf(stderr, "[ERROR] h-grid exceeds %d points\n",
                                max_pts);
                        return EXIT_FAILURE;
                    }
                    tmp[n++] = xs[s];
                }
            }
            if (fabs(x_use - x2) < 1e-14)
                break;
        }
    }

    qsort(tmp, (size_t)n, sizeof(double), compare_double);
    memcpy(x_arr, tmp, (size_t)n * sizeof(double));
    *n_out = n;
    return EXIT_SUCCESS;
}

static int build_h_grid(const Params *P, int L, double delta0_h0,
                        double m0, GridPoint *grid, int max_pts,
                        int *n_out)
{
    double x_arr[MAX_GRID_POINTS];
    int n_x = 0;
    if (build_variable_grid(P->xmax, P->dx_near, P->dx_mid, P->dx_far,
                            x_arr, max_pts, &n_x) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (P->mode == MODE_CQT) {
        double Lyh = pow((double)L, P->yh);
        for (int i = 0; i < n_x; i++) {
            grid[i].h = x_arr[i] / Lyh;
            grid[i].scale_x = x_arr[i];
            grid[i].kappa = NAN;
        }
    } else {
        if (!(m0 > 0.0) || !(delta0_h0 > 0.0)) {
            fprintf(stderr,
                    "[ERROR] invalid FOQT scale for L=%d: m0=%.17e "
                    "delta0_h0=%.17e\n",
                    L, m0, delta0_h0);
            return EXIT_FAILURE;
        }
        double h_scale = delta0_h0 / (2.0 * m0 * (double)L);
        for (int i = 0; i < n_x; i++) {
            grid[i].h = x_arr[i] * h_scale;
            grid[i].scale_x = NAN;
            grid[i].kappa = x_arr[i];
        }
    }

    *n_out = n_x;
    return EXIT_SUCCESS;
}

static int solve_low_energy(const Basis *b, double g, double h,
                            unsigned long seed, SolveResult *out)
{
    if (b->L <= ED_L_MAX)
        return solve_ed(b, g, h, out);
    return solve_lanczos(b, g, h, seed, out);
}

static int solve_ed(const Basis *b, double g, double h, SolveResult *out)
{
    long long dim = b->dim;
    double *Ham = calloc((size_t)dim * dim, sizeof(double));
    double *eig = malloc((size_t)dim * sizeof(double));
    if (!Ham || !eig) {
        print_allocation_failure("ED", b, g, h);
        free(Ham);
        free(eig);
        return EXIT_FAILURE;
    }

    build_ham(b, g, h, Ham);
    if (diagonalize(Ham, dim, eig) != EXIT_SUCCESS) {
        free(Ham);
        free(eig);
        return EXIT_FAILURE;
    }

    for (int k = 0; k < 4; k++)
        out->evals[k] = (k < dim) ? eig[k] : NAN;
    out->mz = obs_mz_raw(b, Ham);
    out->abs_mz = obs_psi_bar(b, Ham);
    out->mx = obs_mx(b, Ham);
    for (int k = 0; k < 4; k++)
        out->resid[k] = NAN;
    out->method_code = 0;
    out->lanczos_ret = EXIT_SUCCESS;

    free(Ham);
    free(eig);
    return EXIT_SUCCESS;
}

static int solve_lanczos(const Basis *b, double g, double h,
                         unsigned long seed, SolveResult *out)
{
    static bool residual_warning_printed = false;
    long long dim = b->dim;
    double evals[4] = {NAN, NAN, NAN, NAN};
    double *evecs = malloc((size_t)4 * dim * sizeof(double));
    if (!evecs) {
        print_allocation_failure("Lanczos eigenvectors", b, g, h);
        return EXIT_FAILURE;
    }

    LanczosParams par = {
        .n_eig = 4,
        .max_iter = lanczos_max_iter(b->L),
        .tol = 1e-10,
        .max_restarts = 80,
        .verbose = 0,
        .seed = seed,
    };

    int ret = lanczos(b, g, h, &par, evals, evecs);
    if (ret != EXIT_SUCCESS && ret != -3) {
        fprintf(stderr,
                "[ERROR] Lanczos failed for L=%d g=%.17g h=%.17e ret=%d\n",
                b->L, g, h, ret);
        free(evecs);
        return EXIT_FAILURE;
    }
    if (ret == -3) {
        fprintf(stderr,
                "[WARN] Lanczos did not fully converge for L=%d g=%.17g "
                "h=%.17e; writing available Ritz values.\n",
                b->L, g, h);
    }
    if (!residual_warning_printed) {
        fprintf(stderr,
                "[WARN] Lanczos residuals are not exposed by the current API; "
                "writing resid0..resid3 as NaN.\n");
        residual_warning_printed = true;
    }

    for (int k = 0; k < 4; k++) {
        out->evals[k] = evals[k];
        out->resid[k] = NAN;
    }
    out->mz = obs_mz_raw(b, evecs);
    out->abs_mz = obs_psi_bar(b, evecs);
    out->mx = obs_mx(b, evecs);
    out->method_code = 1;
    out->lanczos_ret = ret;

    free(evecs);
    return EXIT_SUCCESS;
}

static int lanczos_max_iter(int L)
{
    if (L <= 16)
        return 200;
    if (L <= 18)
        return 150;
    if (L <= 20)
        return 100;
    return 60;
}

static double compute_delta0_h0(const Basis *b, double g, unsigned long seed,
                                int *method_code)
{
    SolveResult r;
    memset(&r, 0, sizeof(r));
    if (solve_low_energy(b, g, 0.0, seed, &r) != EXIT_SUCCESS)
        return NAN;
    if (method_code)
        *method_code = r.method_code;
    return r.evals[1] - r.evals[0];
}

static int write_one_L(const Params *P, int L, int *new_rows_total)
{
    int status = EXIT_FAILURE;
    LockFile lock;
    memset(&lock, 0, sizeof(lock));

    char path[MAX_PATH_LEN];
    build_output_path(P, L, path, sizeof(path));
    if (acquire_lock(path, P, &lock) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (P->overwrite) {
        if (remove_existing_outputs(path) != EXIT_SUCCESS)
            goto cleanup;
    } else if (!P->resume && path_exists(path)) {
        fprintf(stderr,
                "[ERROR] output file already exists after lock acquisition: %s\n"
                "        Pass --overwrite to replace it or --resume to continue it.\n",
                path);
        goto cleanup;
    }

    Basis b = basis_init(L, P->pbc);
    long long dim = b.dim;
    const char *method = (L <= ED_L_MAX) ? "ED" : "Lanczos";
    double mem_gb = estimate_peak_memory_gb(L);
    if (P->max_ram_set && mem_gb > P->max_ram_gb) {
        fprintf(stderr,
                "[ERROR] memory guard rejected L=%d dim=%lld method=%s: "
                "estimated peak %.3f GiB exceeds --max-ram-gb %.3f\n",
                L, dim, method, mem_gb, P->max_ram_gb);
        goto cleanup;
    }

    int delta_method = -1;
    unsigned long seed_base = P->seed + (unsigned long)L * 104729UL;
    double delta0_h0 = compute_delta0_h0(&b, P->g, seed_base + 17UL,
                                         &delta_method);
    if (!isfinite(delta0_h0) || delta0_h0 < 0.0) {
        fprintf(stderr, "[ERROR] failed to compute delta0_h0 for L=%d\n", L);
        goto cleanup;
    }

    double m0 = (P->mode == MODE_FOQT) ? pow(1.0 - P->g * P->g, 0.125) : NAN;
    GridPoint grid[MAX_GRID_POINTS];
    int N_h = 0;
    if (build_h_grid(P, L, delta0_h0, m0, grid, MAX_GRID_POINTS,
                     &N_h) != EXIT_SUCCESS)
        goto cleanup;
    if (N_h <= 0) {
        fprintf(stderr, "[ERROR] empty h-grid for L=%d\n", L);
        goto cleanup;
    }

    bool completed[MAX_GRID_POINTS];
    for (int i = 0; i < N_h; i++)
        completed[i] = false;

    RunStats stats;
    init_run_stats(&stats);

    bool existing = path_exists(path);
    if (P->resume && existing) {
        if (verify_header_compatible(path, P, L, N_h, delta0_h0, m0,
                                     method) != EXIT_SUCCESS)
            goto cleanup;
        if (parse_existing_rows(path, P, grid, N_h, completed,
                                &stats) != EXIT_SUCCESS)
            goto cleanup;
        stats.skipped_rows_this_run = stats.completed_valid_rows;
        printf("resume L=%d: found %d completed points, %d missing points\n",
               L, stats.completed_valid_rows,
               N_h - stats.completed_valid_rows);
    } else {
        if (P->resume && !existing) {
            printf("resume L=%d: no existing output, creating %s\n", L, path);
        }
        if (write_new_output_header(path, P, L, N_h, delta0_h0, m0,
                                    method) != EXIT_SUCCESS)
            goto cleanup;
    }
    fflush(stdout);

    double min_h = grid[0].h, max_h = grid[0].h;
    double min_s = primary_value(P, &grid[0]);
    double max_s = min_s;
    for (int i = 1; i < N_h; i++) {
        double s = primary_value(P, &grid[i]);
        if (grid[i].h < min_h) min_h = grid[i].h;
        if (grid[i].h > max_h) max_h = grid[i].h;
        if (s < min_s) min_s = s;
        if (s > max_s) max_s = s;
    }

    double run_t0 = now_seconds();
    double last_primary = NAN;
    double last_h = NAN;

    for (int ih = 0; ih < N_h; ih++) {
        if (completed[ih])
            continue;

        double t0 = now_seconds();
        SolveResult r;
        memset(&r, 0, sizeof(r));
        unsigned long seed = seed_base + (unsigned long)ih * 8191UL + 101UL;
        if (solve_low_energy(&b, P->g, grid[ih].h, seed, &r) != EXIT_SUCCESS)
            goto cleanup;

        double delta_h = r.evals[1] - r.evals[0];
        if (append_data_row(path, &grid[ih], &r, delta_h,
                            delta0_h0) != EXIT_SUCCESS)
            goto cleanup;

        completed[ih] = true;
        stats.completed_valid_rows++;
        stats.newly_computed_rows_this_run++;
        (*new_rows_total)++;
        update_ranges(&stats, delta_h, r.mz);

        last_primary = primary_value(P, &grid[ih]);
        last_h = grid[ih].h;
        double dt = now_seconds() - t0;
        double elapsed = now_seconds() - run_t0;
        double avg = elapsed / (double)stats.newly_computed_rows_this_run;
        int remaining = N_h - stats.completed_valid_rows;
        double eta = avg * (double)remaining;

        if (write_checkpoint(path, P, L, N_h, delta0_h0, m0, &stats,
                             last_primary, last_h, run_t0, primary_name(P),
                             eta) != EXIT_SUCCESS)
            goto cleanup;

        printf("L=%d point %d/%d h=%.17e %s=%.17e dt=%.2fs avg=%.2fs "
               "eta=%.0fs skipped=%d\n",
               L, ih + 1, N_h, grid[ih].h, primary_name(P), last_primary,
               dt, avg, eta, stats.skipped_rows_this_run);
        fflush(stdout);

        if (P->stop_after >= 0 && *new_rows_total >= P->stop_after) {
            fprintf(stderr,
                    "[ising_hfield] --stop-after %d reached after writing %d new rows; "
                    "exiting with failure for resume testing.\n",
                    P->stop_after, *new_rows_total);
            write_checkpoint(path, P, L, N_h, delta0_h0, m0, &stats,
                             last_primary, last_h, run_t0, primary_name(P),
                             eta);
            status = EXIT_FAILURE;
            goto cleanup;
        }
    }

    if (write_checkpoint(path, P, L, N_h, delta0_h0, m0, &stats,
                         last_primary, last_h, run_t0, primary_name(P),
                         0.0) != EXIT_SUCCESS)
        goto cleanup;

    char h_range[96], s_range[96], d_range[96], mz_range[96];
    format_range(min_h, max_h, h_range, sizeof(h_range));
    format_range(min_s, max_s, s_range, sizeof(s_range));
    if (ranges_valid(&stats)) {
        format_range(stats.min_delta_h, stats.max_delta_h,
                     d_range, sizeof(d_range));
        format_range(stats.min_mz, stats.max_mz, mz_range, sizeof(mz_range));
    } else {
        snprintf(d_range, sizeof(d_range), "[NaN, NaN]");
        snprintf(mz_range, sizeof(mz_range), "[NaN, NaN]");
    }

    printf("L=%2d dim=%9lld method=%s scaling-pts=%d\n",
           L, dim, method, N_h);
    printf("  h range          %s\n", h_range);
    printf("  %s range     %s\n",
           (P->mode == MODE_CQT) ? "scale_x" : "kappa  ", s_range);
    printf("  delta0_h0        %.17e\n", delta0_h0);
    printf("  delta_h range    %s\n", d_range);
    printf("  mz range         %s\n", mz_range);
    printf("  total points     %d\n", N_h);
    printf("  completed before %d\n", stats.skipped_rows_this_run);
    printf("  newly computed   %d\n", stats.newly_computed_rows_this_run);
    printf("  skipped          %d\n", stats.skipped_rows_this_run);
    printf("  output           %s\n", path);
    fflush(stdout);

    status = EXIT_SUCCESS;

cleanup:
    if (release_lock(&lock) != EXIT_SUCCESS)
        status = EXIT_FAILURE;
    return status;
}

static int write_header(FILE *fp, const Params *P, int L, int N_h,
                        double delta0_h0, double m0, const char *method)
{
    time_t now = time(NULL);
    int rc = fprintf(fp,
            "# 1D Quantum Ising -- longitudinal h-field generator\n"
            "# Generated: %s"
            "# Hamiltonian convention: H = -sum_j sigma^z_j sigma^z_{j+1}"
            " - g sum_j sigma^x_j - h sum_j sigma^z_j\n"
            "# L = %d\n"
            "# g = %.17g\n"
            "# pbc = %d\n"
            "# mode = %s\n"
            "# yh = %.17g\n"
            "# grid_type = scaling_adaptive\n"
            "# scaling_variable = %s\n"
            "# %s = [-%.17g, %.17g]\n"
            "# xmax = %.17g\n"
            "# dx_near = %.17g\n"
            "# dx_mid = %.17g\n"
            "# dx_far = %.17g\n"
            "# N_h = %d\n"
            "# delta0_h0 = %.17e\n"
            "# m0 = %.17e\n"
            "# method = %s\n"
            "# method_code: 0=ED, 1=Lanczos\n"
            "# residuals: Lanczos API does not expose them here; NaN unless "
            "available in a future implementation\n"
            "# seed_policy = base seed %lu; per-L and per-h deterministic offsets\n"
            "# scale_x = h*L^yh in CQT mode, NaN in FOQT mode\n"
            "# kappa = 2*m0*h*L/delta0_h0 in FOQT mode, NaN in CQT mode\n"
            "# mz = <sum_j sigma^z_j>/L, signed longitudinal magnetization "
            "coupled to h\n"
            "# abs_mz = <|sum_j sigma^z_j|>/L, nonnegative diagnostic\n"
            "# mx = <sum_j sigma^x_j>/L, transverse magnetization coupled to g\n"
            "# magnetization_method = mz, abs_mz, and mx are computed directly "
            "from the ground-state eigenvector; Hellmann-Feynman derivatives "
            "are validation diagnostics only\n"
            "# Columns: %s\n"
            "#\n",
            ctime(&now), L, P->g, P->pbc, mode_name(P->mode),
            (P->mode == MODE_CQT) ? P->yh : NAN,
            (P->mode == MODE_CQT) ? "x_h = h * L^y_h" :
                                     "kappa = 2*m0*h*L/delta0_h0",
            (P->mode == MODE_CQT) ? "scale_x_range" : "kappa_range",
            P->xmax, P->xmax,
            P->xmax, P->dx_near, P->dx_mid, P->dx_far,
            N_h, delta0_h0, m0, method, P->seed, EXPECTED_COLUMNS);
    return (rc < 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}

static int write_new_output_header(const char *path, const Params *P, int L,
                                   int N_h, double delta0_h0, double m0,
                                   const char *method)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot create %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (write_header(fp, P, L, N_h, delta0_h0, m0,
                     method) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] failed to write header to %s\n", path);
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, path) != EXIT_SUCCESS) {
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int append_data_row(const char *path, const GridPoint *gp,
                           const SolveResult *r, double delta_h,
                           double delta0_h0)
{
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot append to %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    int rc = fprintf(fp,
            "%.17e %.17e %.17e "
            "%.17e %.17e %.17e %.17e "
            "%.17e %.17e %.17e %.17e %.17e %d "
            "%.17e %.17e %.17e %.17e\n",
            gp->h, gp->scale_x, gp->kappa,
            r->evals[0], r->evals[1], r->evals[2], r->evals[3],
            delta_h, delta0_h0, r->mz, r->abs_mz, r->mx, r->method_code,
            r->resid[0], r->resid[1], r->resid[2], r->resid[3]);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] fprintf failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, path) != EXIT_SUCCESS) {
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int write_checkpoint(const char *path, const Params *P, int L,
                            int total_points, double delta0_h0, double m0,
                            const RunStats *stats,
                            double last_primary, double last_h,
                            double run_t0, const char *primary_name_value,
                            double estimated_remaining_seconds)
{
    char ckpt[MAX_PATH_LEN], tmp[MAX_PATH_LEN + 8];
    char timestamp[64], hostname[128];
    build_checkpoint_path(path, ckpt, sizeof(ckpt));
    if (ckpt[0] == '\0') {
        fprintf(stderr, "[ERROR] checkpoint path too long for %s\n", path);
        return EXIT_FAILURE;
    }
    snprintf(tmp, sizeof(tmp), "%s.tmp", ckpt);
    format_timestamp(time(NULL), timestamp, sizeof(timestamp));
    get_hostname_string(hostname, sizeof(hostname));

    FILE *fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot write checkpoint %s: %s\n",
                tmp, strerror(errno));
        return EXIT_FAILURE;
    }
    double elapsed = now_seconds() - run_t0;
    double avg = (stats->newly_computed_rows_this_run > 0)
                     ? elapsed / (double)stats->newly_computed_rows_this_run
                     : NAN;
    int rc = fprintf(fp,
            "output = %s\n"
            "L = %d\n"
            "g = %.17g\n"
            "pbc = %d\n"
            "mode = %s\n"
            "yh = %.17g\n"
            "xmax = %.17g\n"
            "dx_near = %.17g\n"
            "dx_mid = %.17g\n"
            "dx_far = %.17g\n"
            "delta0_h0 = %.17e\n"
            "m0 = %.17e\n"
            "total_points = %d\n"
            "completed_valid_rows = %d\n"
            "newly_computed_rows_this_run = %d\n"
            "skipped_rows_this_run = %d\n"
            "last_primary_name = %s\n"
            "last_primary_variable = %.17e\n"
            "last_h = %.17e\n"
            "avg_seconds_per_new_point = %.17g\n"
            "elapsed_seconds_this_run = %.17g\n"
            "estimated_remaining_seconds = %.17g\n"
            "timestamp = %s\n"
            "pid = %ld\n"
            "hostname = %s\n"
            "command = %s\n",
            path, L, P->g, P->pbc, mode_name(P->mode),
            (P->mode == MODE_CQT) ? P->yh : NAN,
            P->xmax, P->dx_near, P->dx_mid, P->dx_far,
            delta0_h0, m0,
            total_points, stats->completed_valid_rows,
            stats->newly_computed_rows_this_run,
            stats->skipped_rows_this_run,
            primary_name_value, last_primary, last_h, avg, elapsed,
            estimated_remaining_seconds, timestamp,
#ifndef _WIN32
            (long)getpid(),
#else
            0L,
#endif
            hostname, P->command);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] fprintf failed for checkpoint %s\n", tmp);
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for checkpoint %s: %s\n",
                tmp, strerror(errno));
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, tmp) != EXIT_SUCCESS) {
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for checkpoint %s: %s\n",
                tmp, strerror(errno));
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (rename(tmp, ckpt) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] rename checkpoint %s -> %s failed: %s\n",
                tmp, ckpt, strerror(errno));
        remove(tmp);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int fsync_stream(FILE *fp, const char *path)
{
#ifndef _WIN32
    int fd = fileno(fp);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] fileno failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (fsync(fd) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fsync failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
#else
    (void)fp;
    (void)path;
#endif
    return EXIT_SUCCESS;
}

static int acquire_lock(const char *output_path, const Params *P,
                        LockFile *lock)
{
    build_lock_path(output_path, lock->path, sizeof(lock->path));
    lock->held = false;
    if (lock->path[0] == '\0') {
        fprintf(stderr, "[ERROR] lock path too long for %s\n", output_path);
        return EXIT_FAILURE;
    }

    if (path_exists(lock->path)) {
        if (!P->force_unlock) {
            fprintf(stderr,
                    "[ERROR] lock file exists: %s\n"
                    "        Another process may be writing %s.\n"
                    "        Use --force-unlock only if you are sure the lock is stale.\n",
                    lock->path, output_path);
            return EXIT_FAILURE;
        }
        fprintf(stderr,
                "[WARN] --force-unlock removing existing lock %s. Contents:\n",
                lock->path);
        FILE *old = fopen(lock->path, "r");
        if (old) {
            char line[1024];
            while (fgets(line, sizeof(line), old))
                fputs(line, stderr);
            fclose(old);
        }
        if (remove(lock->path) != EXIT_SUCCESS && errno != ENOENT) {
            fprintf(stderr, "[ERROR] cannot remove stale lock %s: %s\n",
                    lock->path, strerror(errno));
            return EXIT_FAILURE;
        }
    }

#ifndef _WIN32
    int fd = open(lock->path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        if (errno == EEXIST) {
            fprintf(stderr,
                    "[ERROR] failed to acquire lock %s: already exists.\n"
                    "        Use --force-unlock only if you are sure it is stale.\n",
                    lock->path);
        } else {
            fprintf(stderr, "[ERROR] failed to acquire lock %s: %s\n",
                    lock->path, strerror(errno));
        }
        return EXIT_FAILURE;
    }
    FILE *fp = fdopen(fd, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] fdopen failed for lock %s: %s\n",
                lock->path, strerror(errno));
        close(fd);
        remove(lock->path);
        return EXIT_FAILURE;
    }
#else
    fprintf(stderr, "[WARN] POSIX atomic locks unavailable; using conservative lock fallback.\n");
    FILE *fp = fopen(lock->path, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] failed to create lock %s: %s\n",
                lock->path, strerror(errno));
        return EXIT_FAILURE;
    }
#endif

    char timestamp[64], hostname[128];
    format_timestamp(time(NULL), timestamp, sizeof(timestamp));
    get_hostname_string(hostname, sizeof(hostname));
    int rc = fprintf(fp,
            "output = %s\n"
            "pid = %ld\n"
            "hostname = %s\n"
            "timestamp = %s\n"
            "command = %s\n",
            output_path,
#ifndef _WIN32
            (long)getpid(),
#else
            0L,
#endif
            hostname, timestamp, P->command);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] failed while writing lock %s\n", lock->path);
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for lock %s: %s\n",
                lock->path, strerror(errno));
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, lock->path) != EXIT_SUCCESS) {
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for lock %s: %s\n",
                lock->path, strerror(errno));
        remove(lock->path);
        return EXIT_FAILURE;
    }

    lock->held = true;
#ifndef _WIN32
    strncpy(g_active_lock_path, lock->path, sizeof(g_active_lock_path) - 1);
    g_active_lock_path[sizeof(g_active_lock_path) - 1] = '\0';
    g_lock_held = 1;
#endif
    return EXIT_SUCCESS;
}

static int release_lock(LockFile *lock)
{
    if (!lock->held)
        return EXIT_SUCCESS;
    if (remove(lock->path) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] failed to remove lock %s: %s\n",
                lock->path, strerror(errno));
        return EXIT_FAILURE;
    }
    lock->held = false;
#ifndef _WIN32
    g_lock_held = 0;
    g_active_lock_path[0] = '\0';
#endif
    return EXIT_SUCCESS;
}

static int remove_existing_outputs(const char *path)
{
    char ckpt[MAX_PATH_LEN];
    build_checkpoint_path(path, ckpt, sizeof(ckpt));
    if (remove(path) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] cannot remove %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (remove(ckpt) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] cannot remove %s: %s\n", ckpt, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int verify_header_compatible(const char *path, const Params *P, int L,
                                    int N_h, double delta0_h0, double m0,
                                    const char *method)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot read %s for resume: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }

    bool have_L = false, have_g = false, have_pbc = false, have_mode = false;
    bool have_yh = false, have_grid = false, have_scaling = false;
    bool have_xmax = false, have_dx_near = false, have_dx_mid = false;
    bool have_dx_far = false, have_N_h = false, have_delta = false;
    bool have_m0 = false, have_method = false, have_columns = false;
    char line[2048];
    int line_no = 0;

    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        if (line[0] != '#')
            break;
        if (header_has_obsolete_field(line)) {
            fprintf(stderr, "[ERROR] incompatible obsolete header field in %s:%d: %s",
                    path, line_no, line);
            fclose(fp);
            return EXIT_FAILURE;
        }

        int iv = 0;
        double dv = 0.0;
        char sv[128];
        if (sscanf(line, "# L = %d", &iv) == 1) {
            have_L = true;
            if (iv != L) {
                fprintf(stderr, "[ERROR] resume L mismatch: file=%d run=%d\n", iv, L);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# g = %lf", &dv) == 1) {
            have_g = true;
            if (!double_close(dv, P->g, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume g mismatch: file=%.17g run=%.17g\n",
                        dv, P->g);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# pbc = %d", &iv) == 1) {
            have_pbc = true;
            if (iv != P->pbc) {
                fprintf(stderr, "[ERROR] resume pbc mismatch: file=%d run=%d\n",
                        iv, P->pbc);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# mode = %127s", sv) == 1) {
            have_mode = true;
            if (strcmp(sv, mode_name(P->mode)) != 0) {
                fprintf(stderr, "[ERROR] resume mode mismatch: file=%s run=%s\n",
                        sv, mode_name(P->mode));
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# yh = %lf", &dv) == 1) {
            have_yh = true;
            if (P->mode == MODE_CQT &&
                !double_close(dv, P->yh, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume yh mismatch: file=%.17g run=%.17g\n",
                        dv, P->yh);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# grid_type = %127s", sv) == 1) {
            have_grid = true;
            if (strcmp(sv, "scaling_adaptive") != 0) {
                fprintf(stderr, "[ERROR] resume grid_type mismatch: %s\n", sv);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (strncmp(line, "# scaling_variable =", 20) == 0) {
            have_scaling = true;
        } else if (sscanf(line, "# xmax = %lf", &dv) == 1) {
            have_xmax = true;
            if (!double_close(dv, P->xmax, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume xmax mismatch: file=%.17g run=%.17g\n",
                        dv, P->xmax);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# dx_near = %lf", &dv) == 1) {
            have_dx_near = true;
            if (!double_close(dv, P->dx_near, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume dx_near mismatch: file=%.17g run=%.17g\n",
                        dv, P->dx_near);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# dx_mid = %lf", &dv) == 1) {
            have_dx_mid = true;
            if (!double_close(dv, P->dx_mid, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume dx_mid mismatch: file=%.17g run=%.17g\n",
                        dv, P->dx_mid);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# dx_far = %lf", &dv) == 1) {
            have_dx_far = true;
            if (!double_close(dv, P->dx_far, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume dx_far mismatch: file=%.17g run=%.17g\n",
                        dv, P->dx_far);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# N_h = %d", &iv) == 1) {
            have_N_h = true;
            if (iv != N_h) {
                fprintf(stderr, "[ERROR] resume N_h mismatch: file=%d run=%d\n",
                        iv, N_h);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# delta0_h0 = %lf", &dv) == 1) {
            have_delta = true;
            if (!double_close(dv, delta0_h0, 1e-10, 1e-10)) {
                fprintf(stderr,
                        "[ERROR] resume delta0_h0 mismatch: file=%.17e run=%.17e\n",
                        dv, delta0_h0);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# m0 = %lf", &dv) == 1) {
            have_m0 = true;
            if (P->mode == MODE_FOQT && !double_close(dv, m0, 1e-13, 1e-13)) {
                fprintf(stderr, "[ERROR] resume m0 mismatch: file=%.17e run=%.17e\n",
                        dv, m0);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (sscanf(line, "# method = %127s", sv) == 1) {
            have_method = true;
            if (strcmp(sv, method) != 0) {
                fprintf(stderr, "[ERROR] resume method mismatch: file=%s run=%s\n",
                        sv, method);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } else if (strncmp(line, "# Columns:", 10) == 0) {
            have_columns = (strstr(line, EXPECTED_COLUMNS) != NULL);
            if (!have_columns) {
                fprintf(stderr, "[ERROR] resume column schema mismatch in %s\n", path);
                fclose(fp);
                return EXIT_FAILURE;
            }
        }
    }
    fclose(fp);

    if (!(have_L && have_g && have_pbc && have_mode && have_yh &&
          have_grid && have_scaling && have_xmax && have_dx_near &&
          have_dx_mid && have_dx_far && have_N_h && have_delta &&
          have_m0 && have_method && have_columns)) {
        fprintf(stderr,
                "[ERROR] resume header in %s is missing required metadata\n",
                path);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int parse_existing_rows(const char *path, const Params *P,
                               const GridPoint *grid, int N_h,
                               bool *completed, RunStats *stats)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot read %s for resume rows: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }

    char line[4096];
    int line_no = 0;
    bool pending_invalid = false;
    int pending_invalid_line = 0;
    long pending_invalid_offset = -1;

    for (;;) {
        long line_start = ftell(fp);
        if (!fgets(line, sizeof(line), fp))
            break;
        line_no++;
        char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;

        DataRow row;
        bool valid = parse_data_row(p, &row);
        int idx = -1;
        if (valid) {
            double primary = (P->mode == MODE_CQT) ? row.scale_x : row.kappa;
            idx = match_grid_index(P, grid, N_h, primary);
            if (idx < 0)
                valid = false;
        }

        if (!valid) {
            if (pending_invalid) {
                fprintf(stderr,
                        "[ERROR] more than one corrupted trailing row while "
                        "resuming %s (first at line %d, another at line %d)\n",
                        path, pending_invalid_line, line_no);
                fclose(fp);
                return EXIT_FAILURE;
            }
            pending_invalid = true;
            pending_invalid_line = line_no;
            pending_invalid_offset = line_start;
            continue;
        }

        if (pending_invalid) {
            fprintf(stderr,
                    "[ERROR] corrupted row at %s:%d is not the final line\n",
                    path, pending_invalid_line);
            fclose(fp);
            return EXIT_FAILURE;
        }

        if (completed[idx]) {
            fprintf(stderr,
                    "[ERROR] duplicate completed primary variable in %s at line %d\n",
                    path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }

        completed[idx] = true;
        stats->completed_valid_rows++;
        update_ranges(stats, row.delta_h, row.mz);
    }

    if (pending_invalid) {
        fprintf(stderr,
                "[WARN] ignoring corrupted final row in %s at line %d during resume\n",
                path, pending_invalid_line);
        fclose(fp);
        if (pending_invalid_offset >= 0 &&
            truncate_file_at(path, pending_invalid_offset) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        return EXIT_SUCCESS;
    }

    fclose(fp);
    return EXIT_SUCCESS;
}

static int truncate_file_at(const char *path, long offset)
{
    if (offset < 0)
        return EXIT_FAILURE;
#ifndef _WIN32
    if (truncate(path, (off_t)offset) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] failed to truncate corrupted tail from %s: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
#else
    fprintf(stderr,
            "[WARN] cannot truncate corrupted final row on this platform: %s\n",
            path);
    return EXIT_SUCCESS;
#endif
}

static bool parse_data_row(const char *line, DataRow *row)
{
    const char *p = line;
    if (!parse_double_field(&p, &row->h)) return false;
    if (!parse_double_field(&p, &row->scale_x)) return false;
    if (!parse_double_field(&p, &row->kappa)) return false;
    for (int k = 0; k < 4; k++)
        if (!parse_double_field(&p, &row->evals[k])) return false;
    if (!parse_double_field(&p, &row->delta_h)) return false;
    if (!parse_double_field(&p, &row->delta0_h0)) return false;
    if (!parse_double_field(&p, &row->mz)) return false;
    if (!parse_double_field(&p, &row->abs_mz)) return false;
    if (!parse_double_field(&p, &row->mx)) return false;
    if (!parse_int_field(&p, &row->method_code)) return false;
    for (int k = 0; k < 4; k++)
        if (!parse_double_field(&p, &row->resid[k])) return false;
    while (isspace((unsigned char)*p))
        p++;
    return *p == '\0';
}

static bool parse_double_field(const char **p, double *out)
{
    while (isspace((unsigned char)**p))
        (*p)++;
    if (**p == '\0')
        return false;
    errno = 0;
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p || errno)
        return false;
    *out = v;
    *p = end;
    return true;
}

static bool parse_int_field(const char **p, int *out)
{
    while (isspace((unsigned char)**p))
        (*p)++;
    if (**p == '\0')
        return false;
    errno = 0;
    char *end = NULL;
    long v = strtol(*p, &end, 10);
    if (end == *p || errno || v < -2147483647L || v > 2147483647L)
        return false;
    *out = (int)v;
    *p = end;
    return true;
}

static int match_grid_index(const Params *P, const GridPoint *grid, int N_h,
                            double primary)
{
    for (int i = 0; i < N_h; i++) {
        if (fabs(primary_value(P, &grid[i]) - primary) <= PRIMARY_TOL)
            return i;
    }
    return -1;
}

static double primary_value(const Params *P, const GridPoint *gp)
{
    return (P->mode == MODE_CQT) ? gp->scale_x : gp->kappa;
}

static const char *primary_name(const Params *P)
{
    return (P->mode == MODE_CQT) ? "scale_x" : "kappa";
}

static void init_run_stats(RunStats *stats)
{
    stats->completed_valid_rows = 0;
    stats->skipped_rows_this_run = 0;
    stats->newly_computed_rows_this_run = 0;
    stats->min_delta_h = HUGE_VAL;
    stats->max_delta_h = -HUGE_VAL;
    stats->min_mz = HUGE_VAL;
    stats->max_mz = -HUGE_VAL;
}

static void update_ranges(RunStats *stats, double delta_h, double mz)
{
    if (delta_h < stats->min_delta_h) stats->min_delta_h = delta_h;
    if (delta_h > stats->max_delta_h) stats->max_delta_h = delta_h;
    if (mz < stats->min_mz) stats->min_mz = mz;
    if (mz > stats->max_mz) stats->max_mz = mz;
}

static bool ranges_valid(const RunStats *stats)
{
    return stats->completed_valid_rows > 0 &&
           stats->min_delta_h != HUGE_VAL &&
           stats->max_delta_h != -HUGE_VAL;
}

static void format_range(double min_v, double max_v, char *buf, size_t n)
{
    snprintf(buf, n, "[%.17e, %.17e]", min_v, max_v);
}

static void format_timestamp(time_t t, char *buf, size_t n)
{
    struct tm *tm_info = localtime(&t);
    if (!tm_info) {
        snprintf(buf, n, "unknown");
        return;
    }
    strftime(buf, n, "%Y-%m-%d %H:%M:%S %Z", tm_info);
}

static void get_hostname_string(char *buf, size_t n)
{
#ifndef _WIN32
    if (gethostname(buf, n) != EXIT_SUCCESS) {
        snprintf(buf, n, "unknown");
        return;
    }
    buf[n - 1] = '\0';
#else
    snprintf(buf, n, "unknown");
#endif
}

static double now_seconds(void)
{
    return (double)time(NULL);
}

static double estimate_peak_memory_gb(int L)
{
    double dim = (double)(1LL << L);
    double bytes = 0.0;
    if (L <= ED_L_MAX) {
        double matrix = dim * dim * sizeof(double);
        double eig = dim * sizeof(double);
        bytes = 3.0 * matrix + eig + 64.0 * dim * sizeof(double);
    } else {
        int D = lanczos_max_iter(L);
        bytes = ((double)D + 12.0) * dim * sizeof(double)
              + 4.0 * dim * sizeof(double)
              + 4.0 * (double)D * (double)D * sizeof(double);
    }
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_allocation_failure(const char *where, const Basis *b,
                                     double g, double h)
{
    fprintf(stderr,
            "[ERROR] allocation failed in %s: L=%d dim=%lld g=%.17g "
            "h=%.17e mode_method=%s estimated_peak=%.3f GiB\n",
            where, b->L, b->dim, g, h,
            (b->L <= ED_L_MAX) ? "ED" : "Lanczos",
            estimate_peak_memory_gb(b->L));
}

static bool double_close(double a, double b, double abs_tol, double rel_tol)
{
    if (isnan(a) && isnan(b))
        return true;
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return fabs(a - b) <= abs_tol + rel_tol * scale;
}

static bool header_has_obsolete_field(const char *line)
{
    static const char *bad[] = {
        "h_window",
        "dh_window",
        "hmax",
        "h_core",
        "dh_core",
        "dh_outer",
        "physical_adaptive",
        "union grid",
        "physical h-grid",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        if (strstr(line, bad[i]))
            return true;
    }
    return false;
}

#ifndef _WIN32
static void handle_signal(int sig)
{
    (void)sig;
    if (g_lock_held && g_active_lock_path[0] != '\0')
        unlink(g_active_lock_path);
    _Exit(EXIT_FAILURE);
}

static int install_signal_handlers(void)
{
    if (signal(SIGINT, handle_signal) == SIG_ERR)
        return EXIT_FAILURE;
    if (signal(SIGTERM, handle_signal) == SIG_ERR)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
#endif
