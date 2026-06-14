/* ============================================================
 * main_static.c
 * Full exact diagonalization: static observables of the 1D
 * quantum Ising chain.
 *
 * Hamiltonian:
 *   H = - sum_j sigma^z_j sigma^z_{j+1}
 *       - h * sum_j sigma^z_j
 *       - g   * sum_j sigma^x_j
 *
 * Usage:
 *   ./ising_static [--resume] <h> <pbc> [L1 L2 ...] [--resume]
 *
 * Output files:
 *   PBC (pbc=1):  data/h_null/observables/PBC/gap_L<LL>.dat
 *                 data/h_null/observables/PBC/obs_L<LL>.dat
 *   OBC (pbc=0):  data/h_null/observables/OBC/gap_obc_L<LL>.dat
 *                 data/h_null/observables/OBC/obs_obc_L<LL>.dat
 *
 * System sizes:
 *   PBC: L in {4, 6, 8, 10, 12}
 *   OBC: L in {4, 6, 8, 10, 12}
 * ============================================================ */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir((path), 0755)
#endif

#include "basis.h"
#include "diag.h"
#include "hamiltonian.h"
#include "observables.h"

/* ============================================================
 *  CONFIGURATION
 * ============================================================ */

static const int L_PBC[] = {4, 6, 8, 10, 12};
static const int N_L_PBC = (int)(sizeof(L_PBC) / sizeof(L_PBC[0]));

static const int L_OBC[] = {4, 6, 8, 10, 12};
static const int N_L_OBC = (int)(sizeof(L_OBC) / sizeof(L_OBC[0]));

/* Hard upper bound: 2^14 = 16384 */
#define MAX_DIM 16384
#define MAX_L_RUNS 32
#define RESUME_G_TOL 1e-8

/* 5-point centered stencil step for chi_x = -(1/L) d2E0/dg2 */
#define CHI_PERP_DG 1e-3

/* ============================================================
 *  ADAPTIVE g-GRID
 * ============================================================ */
#define G_C        1.0
#define NU         1.0
#define G_MIN_PHYS 0.4
#define G_MAX_PHYS 1.6

static const double ZONE_X1[] = {0.0, 1.5, 5.0};
static const double ZONE_X2[] = {1.5, 5.0, 12.0};
static const double ZONE_DX[] = {0.02, 0.10, 0.50};
static const int N_ZONES = 3;

static int build_g_grid(int L, double *g_arr, int max_pts)
{
    double tmp[2048];
    int n = 0;

    for (int iz = 0; iz < N_ZONES; iz++) {
        double x = ZONE_X1[iz];
        while (x <= ZONE_X2[iz] + ZONE_DX[iz] * 0.5) {
            double xs[2] = {x, -x};
            int nx = (x == 0.0) ? 1 : 2;
            for (int s = 0; s < nx; s++) {
                double g = G_C + xs[s] / pow((double)L, 1.0 / NU);
                g = round(g * 1e8) / 1e8;
                if (g < G_MIN_PHYS || g > G_MAX_PHYS) continue;
                int dup = 0;
                for (int k = 0; k < n; k++)
                    if (fabs(tmp[k] - g) < 1e-9) { dup = 1; break; }
                if (!dup && n < max_pts) tmp[n++] = g;
            }
            x += ZONE_DX[iz];
        }
    }

    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (tmp[j] < tmp[i]) {
                double t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t;
            }

    memcpy(g_arr, tmp, n * sizeof(double));
    return n;
}

/* ============================================================
 *  STRUCTS & PROTOTYPES
 * ============================================================ */

typedef struct {
    double h;
    int pbc;
    int resume;
    int L_values[MAX_L_RUNS];
    int N_L;
} SimParams;

typedef struct {
    double total_units;
    double done_units;
    double measured_units;
    time_t run_t0;
} Progress;

static SimParams parse_args(int argc, char **argv);
static double parse_double(const char *s, const char *name);
static int parse_pbc(const char *s);
static int parse_L_value(const char *s);

static void write_gap_header(FILE *fp, int L, double h, int pbc, int N_g);
static void write_obs_header(FILE *fp, int L, double h, int pbc, int N_g);
static int write_gap_file(const char *path, int L, double h, int pbc,
                          int N_g_total, const double *g_arr, int N_rows,
                          const double (*evals_arr)[4]);
static int write_obs_file(const char *path, int L, double h, int pbc,
                          int N_g_total, const double *g_arr, int N_rows,
                          const double (*obs_arr)[8]);
static int load_gap_file(const char *path, const char *header_info,
                         const double *g_arr, int N_g,
                         double (*evals_arr)[4]);
static int load_obs_file(const char *path, const char *header_info,
                         const double *g_arr, int N_g,
                         double (*obs_arr)[8]);
static double compute_chi_perp_ed(const Basis *b, double g, double h,
                                  double dg);

static const char *observables_dir(int pbc);
static int ensure_dir(const char *path);
static int ensure_output_dirs(int pbc);
static void build_output_paths(int pbc, int L, char *gap_path,
                               size_t gap_path_sz, char *obs_path,
                               size_t obs_path_sz);
static int replace_file(const char *tmp, const char *path);
static int file_header_matches(FILE *fp, const char *header_info);
static long detect_cpu_count(void);
static double detect_available_ram_gib(void);
static double estimate_peak_mem_gib(int L);
static double work_units_per_g(int L);
static void format_duration(double seconds, char *buf, size_t n);
static void print_progress(int L, int done_g, int N_g,
                           const Progress *progress, double last_seconds);

static int run_static(const SimParams *P, int L, Progress *progress);

/* ============================================================
 *  MAIN
 * ============================================================ */

int main(int argc, char **argv)
{
    SimParams P = parse_args(argc, argv);

    int total_g_points = 0;
    int valid_L_count = 0;
    double total_units = 0.0;

    for (int i = 0; i < P.N_L; i++) {
        int L = P.L_values[i];
        long long dim = (long long)1 << L;
        if (dim > MAX_DIM)
            continue;

        double g_tmp[2048];
        int N_g = build_g_grid(L, g_tmp, 2048);
        total_g_points += N_g;
        total_units += (double)N_g * work_units_per_g(L);
        valid_L_count++;
    }

    if (valid_L_count == 0) {
        fprintf(stderr, "[ERROR] no runnable L values with dim <= MAX_DIM=%d\n",
                MAX_DIM);
        return EXIT_FAILURE;
    }

    printf("=====================================================\n");
    printf("|   1D QUANTUM ISING - EXACT DIAGONALIZATION       |\n");
    printf("=====================================================\n");
    printf(" g grid   = adaptive [%.1f, %.1f], gc=%.1f, nu=%.1f\n",
           G_MIN_PHYS, G_MAX_PHYS, G_C, NU);
    printf(" h   = %.4f\n", P.h);
    printf(" pbc      = %s\n", P.pbc ? "PBC" : "OBC");
    printf(" resume   = %s\n", P.resume ? "on" : "off");
    printf(" chi_z    = finite-difference data in data/h_null/chiz_fd/dh_5e-04/<BC>/chizfd*.dat\n");
    printf(" chi_x    = 5-pt E0 stencil, dg=%.0e\n", CHI_PERP_DG);
    printf(" calls/g  = 6 dense ED calls (1 spectrum + 5 chi_x; chi_z disabled)\n");
    printf(" L values =");
    for (int i = 0; i < P.N_L; i++) printf(" %d", P.L_values[i]);
    printf("\n total g  = %d\n", total_g_points);

    long cpu_count = detect_cpu_count();
    double ram_avail = detect_available_ram_gib();
    if (cpu_count > 0)
        printf(" cpu      = %ld visible cores\n", cpu_count);
    if (ram_avail > 0.0)
        printf(" RAM      = %.2f GiB available now\n", ram_avail);
    printf(" ETA      = calibrated after the first completed new g-point\n");

    printf(" plan     =\n");
    for (int i = 0; i < P.N_L; i++) {
        int L = P.L_values[i];
        long long dim = (long long)1 << L;
        if (dim > MAX_DIM) {
            printf("   L=%2d  dim=%9lld  skipped: dim > MAX_DIM=%d\n",
                   L, dim, MAX_DIM);
            continue;
        }

        double g_tmp[2048];
        int N_g = build_g_grid(L, g_tmp, 2048);
        printf("   L=%2d  dim=%9lld  g-pts=%3d  peak_RAM/proc~%.2f GiB\n",
               L, dim, N_g, estimate_peak_mem_gib(L));
    }
    printf("-----------------------------------------------------\n");
    fflush(stdout);

    Progress progress = {
        .total_units = total_units,
        .done_units = 0.0,
        .measured_units = 0.0,
        .run_t0 = time(NULL),
    };

    for (int iL = 0; iL < P.N_L; iL++) {
        int L = P.L_values[iL];
        time_t t0 = time(NULL);

        int s1 = run_static(&P, L, &progress);

        double elapsed = difftime(time(NULL), t0);
        printf("  L=%2d  (dim=%5lld)  %.2f s\n", L, (long long)1 << L, elapsed);
        fflush(stdout);

        if (s1) {
            fprintf(stderr, "[ERROR] Failed for L=%d\n", L);
            return EXIT_FAILURE;
        }
    }

    printf("=====================================================\n");
    printf(">> All runs completed.\n");
    fflush(stdout);
    return EXIT_SUCCESS;
}

/* ============================================================
 *  HISTOGRAM HELPER
 * ============================================================ */

static void maybe_write_histogram(const Basis *b, const double *psi0,
                                   double g, int L)
{
    static const double G_HIST[] = {0.5, 1.0, 1.5};
    static const int N_HIST = 3;

    for (int ih = 0; ih < N_HIST; ih++) {
        if (fabs(g - G_HIST[ih]) > 1e-6)
            continue;

        double *hist = calloc((size_t)(L + 1), sizeof(double));
        if (!hist) {
            fprintf(stderr, "[WARN] histogram alloc failed\n");
            return;
        }

        obs_pm_histogram(b, psi0, hist);

        if (ensure_output_dirs(b->pbc) != 0) {
            free(hist);
            return;
        }

        char fname[256];
        snprintf(fname, sizeof(fname),
                 "%s/hist_M_L%02d_g%d.dat", observables_dir(b->pbc), L, ih);

        FILE *fp = fopen(fname, "w");
        if (fp) {
            time_t now = time(NULL);
            fprintf(fp,
                    "# 1D Quantum Ising -- P(M) ground-state histogram\n"
                    "# Generated: %s"
                    "# L=%d  g=%.6f  pbc=%d\n"
                    "# hist[k] = sum_{|ii|=k} |psi0[ii]|^2\n"
                    "# Columns: M_physical  P(M)   [M = 2k - L]\n"
                    "#\n",
                    ctime(&now), L, g, b->pbc);

            for (int k = 0; k <= L; k++)
                fprintf(fp, "%4d  %.12e\n", 2 * k - L, hist[k]);

            fclose(fp);
        }
        free(hist);
        return;
    }
}

/* ============================================================
 *  STATIC SWEEP
 * ============================================================ */

static int run_static(const SimParams *P, int L, Progress *progress)
{
    Basis b = basis_init(L, P->pbc);
    long long dim = b.dim;

    if (dim > MAX_DIM) {
        fprintf(stderr, "[WARN] L=%d dim=%lld > MAX_DIM=%d: skipping.\n",
            L, dim, MAX_DIM);
        return 0;
    }

    if (ensure_output_dirs(P->pbc) != 0)
        return 1;

    char fn_gap[256], fn_obs[256];
    build_output_paths(P->pbc, L, fn_gap, sizeof(fn_gap), fn_obs,
                       sizeof(fn_obs));

    double g_arr[2048];
    int N_g = build_g_grid(L, g_arr, 2048);

    char header[256];
    snprintf(header, sizeof(header),
             "L=%d  h=%.6f  pbc=%d  grid=adaptive  N_g=%d",
             L, P->h, P->pbc, N_g);

    printf("L=%2d  (dim=%9lld, %d g-pts)\n", L, dim, N_g);
    fflush(stdout);

    double (*evals_all)[4] =
        (double (*)[4])malloc((size_t)N_g * 4 * sizeof(double));
    double (*obs_all)[8] =
        (double (*)[8])malloc((size_t)N_g * 8 * sizeof(double));
    if (!evals_all || !obs_all) {
        fprintf(stderr, "[ERROR] Storage allocation failed for L=%d\n", L);
        free(evals_all);
        free(obs_all);
        return 1;
    }

    for (int ig = 0; ig < N_g; ig++) {
        for (int k = 0; k < 4; k++)
            evals_all[ig][k] = NAN;
        for (int k = 0; k < 8; k++)
            obs_all[ig][k] = NAN;
    }

    double units_g = work_units_per_g(L);
    int start_ig = 0;

    if (P->resume) {
        int gap_done = load_gap_file(fn_gap, header, g_arr, N_g, evals_all);
        int obs_done = load_obs_file(fn_obs, header, g_arr, N_g, obs_all);
        start_ig = (gap_done < obs_done) ? gap_done : obs_done;

        if (gap_done != obs_done) {
            fprintf(stderr,
                    "[main_static] resume mismatch L=%d: gap=%d obs=%d; "
                    "using %d safe rows\n",
                    L, gap_done, obs_done, start_ig);
            fflush(stderr);

            if (start_ig > 0) {
                if (write_gap_file(fn_gap, L, P->h, P->pbc, N_g, g_arr,
                                   start_ig, evals_all) != 0 ||
                    write_obs_file(fn_obs, L, P->h, P->pbc, N_g, g_arr,
                                   start_ig, obs_all) != 0) {
                    fprintf(stderr,
                            "[WARN] failed to truncate resume files for L=%d\n",
                            L);
                    fflush(stderr);
                }
            }
        }

        if (start_ig > 0) {
            printf("  resume: loaded %d/%d completed g-points\n",
                   start_ig, N_g);
            fflush(stdout);
        }
    }

    if (start_ig == 0) {
        if (write_gap_file(fn_gap, L, P->h, P->pbc, N_g, g_arr, 0,
                           evals_all) != 0 ||
            write_obs_file(fn_obs, L, P->h, P->pbc, N_g, g_arr, 0,
                           obs_all) != 0) {
            fprintf(stderr, "[ERROR] Cannot initialize output files for L=%d\n",
                    L);
            free(evals_all);
            free(obs_all);
            return 1;
        }
    }

    if (start_ig >= N_g) {
        if (write_obs_file(fn_obs, L, P->h, P->pbc, N_g, g_arr, N_g,
                           obs_all) != 0) {
            fprintf(stderr,
                    "[ERROR] Cannot sanitize completed obs file for L=%d\n",
                    L);
            free(evals_all);
            free(obs_all);
            return 1;
        }
        progress->done_units += (double)N_g * units_g;
        printf("  resume: L=%d already complete; sanitized obs chi_z column to 0.0\n",
               L);
        fflush(stdout);
        print_progress(L, N_g, N_g, progress, 0.0);
        free(evals_all);
        free(obs_all);
        return 0;
    }

    progress->done_units += (double)start_ig * units_g;
    if (start_ig > 0)
        print_progress(L, start_ig, N_g, progress, 0.0);

    double *Ham = calloc((size_t)dim * dim, sizeof(double));
    double *eig = malloc((size_t)dim * sizeof(double));
    if (!Ham || !eig) {
        fprintf(stderr, "[ERROR] Allocation failed for L=%d\n", L);
        free(Ham);
        free(eig);
        free(evals_all);
        free(obs_all);
        return 1;
    }

    for (int ig = start_ig; ig < N_g; ig++) {
        time_t tg0 = time(NULL);
        double g = g_arr[ig];

        memset(Ham, 0, (size_t)dim * dim * sizeof(double));
        build_ham(&b, g, P->h, Ham);

        if (diagonalize(Ham, dim, eig) != 0) {
            fprintf(stderr, "[ERROR] Diag failed at L=%d g=%.6f\n", L, g);
            free(Ham);
            free(eig);
            free(evals_all);
            free(obs_all);
            return 1;
        }

        /* Ham[:,0] stores psi0 after diagonalize(). */
        const double *psi0 = Ham;

        /* --- Gap file --- */
        evals_all[ig][0] = eig[0];
        evals_all[ig][1] = eig[1];
        evals_all[ig][2] = eig[2];
        evals_all[ig][3] = eig[3];

        /* --- Observables --- */
        double Mx = obs_mx(&b, psi0);
        double mz2 = obs_mz_sq(&b, psi0);
        double mz4 = obs_mz4(&b, psi0);
        double chiz = 0.0;
        double binder = obs_binder(mz2, mz4);
        double chi_perp = compute_chi_perp_ed(&b, g, P->h, CHI_PERP_DG);

        /*
         * psi_tilde = |<gs_even | Mz/L | gs_odd>|
         * For h=0 H commutes with P = prod sigma^x_j, so LAPACK's
         * full ED already returns P-eigenstates.  Ham[:,0] and Ham[:,1]
         * are the ground states of the even and odd sectors respectively
         * (checked by comparing the lowest even/odd sector energies).
         * No additional sector diagonalisation needed.
         */
        double psi_t = 0.0;
        if (P->h == 0.0)
            psi_t = obs_psi_tilde(&b, Ham, Ham + dim, P->h);

        double psi_b = obs_psi_bar(&b, psi0);

        obs_all[ig][0] = Mx;
        obs_all[ig][1] = mz2;
        obs_all[ig][2] = mz4;
        obs_all[ig][3] = binder;
        obs_all[ig][4] = chiz;
        obs_all[ig][5] = psi_t;
        obs_all[ig][6] = chi_perp;
        obs_all[ig][7] = psi_b;

        if (write_gap_file(fn_gap, L, P->h, P->pbc, N_g, g_arr, ig + 1,
                           evals_all) != 0) {
            fprintf(stderr, "  [WARN] checkpoint gap write failed for L=%d ig=%d\n",
                    L, ig);
            fflush(stderr);
        }

        if (write_obs_file(fn_obs, L, P->h, P->pbc, N_g, g_arr, ig + 1,
                           obs_all) != 0) {
            fprintf(stderr, "  [WARN] checkpoint obs write failed for L=%d ig=%d\n",
                    L, ig);
            fflush(stderr);
        }

        /* Histogram at selected g values - after checkpointing main data. */
        maybe_write_histogram(&b, psi0, g, L);

        time_t tg1 = time(NULL);
        progress->done_units += units_g;
        progress->measured_units += units_g;
        print_progress(L, ig + 1, N_g, progress, difftime(tg1, tg0));
    }

    printf("  -> %s\n", fn_gap);
    printf("  -> %s\n", fn_obs);
    fflush(stdout);

    free(Ham);
    free(eig);
    free(evals_all);
    free(obs_all);
    return 0;
}

/* ============================================================
 *  FILE HEADERS
 * ============================================================ */

static void write_gap_header(FILE *fp, int L, double h, int pbc, int N_g)
{
    time_t now = time(NULL);
    fprintf(fp,
            "# 1D Quantum Ising -- Full ED -- Gap and Energy Levels\n"
            "# Generated: %s"
            "# L=%d  h=%.6f  pbc=%d  grid=adaptive  N_g=%d\n"
            "# g in [%.4f, %.4f]  (x-adaptive, nu=%.1f, gc=%.1f)\n"
            "# Columns: g  E0  E1  E2  E3  gap=(E1-E0)  gap*L  E0/L\n#\n",
            ctime(&now), L, h, pbc, N_g,
            G_MIN_PHYS, G_MAX_PHYS, NU, G_C);
}

static void write_obs_header(FILE *fp, int L, double h, int pbc, int N_g)
{
    time_t now = time(NULL);
    fprintf(fp,
            "# 1D Quantum Ising -- Full ED -- Observables\n"
            "# Generated: %s"
            "# L=%d  h=%.6f  pbc=%d  grid=adaptive  N_g=%d\n"
            "# g in [%.4f, %.4f]  (x-adaptive, nu=%.1f, gc=%.1f)\n"
            "# Mx      = (1/L) sum_j <psi0|sigma^x_j|psi0>\n"
            "# mz_sq   = <psi0|(Mz/L)^2|psi0>\n"
            "# mz      = sqrt(mz_sq)\n"
            "# chi_z   = 0.0  [finite-difference data: data/h_null/chiz_fd/dh_5e-04/<BC>/chizfd*.dat]\n"
            "# mz4     = <psi0|(Mz/L)^4|psi0>\n"
            "# psi_t   = |<gs_even|Mz/L|gs_odd>|  (h=0 only, else 0)\n"
            "# psi_b = sum_ii |psi0[ii]|^2 * |sz_total(ii,L)| / L\n"
            "# binder  = mz4/mz_sq^2  (->3 PM, ->1 FM)\n"
            "# chi_x = -(1/L) d2E0/dg2  [5-pt stencil, dg=%.0e]\n"
            "# g_chi_x = g * chi_x  (quantum specific heat at T=0)\n"
            "# Columns: g  Mx  mz_sq  mz  chi_z  mz4  psi_tilde  psi_bar  binder"
            "  chi_x  g_chi_x\n",
            ctime(&now), L, h, pbc, N_g,
            G_MIN_PHYS, G_MAX_PHYS, NU, G_C, CHI_PERP_DG);
}

static int replace_file(const char *tmp, const char *path)
{
#ifdef _WIN32
    remove(path);
#endif
    return rename(tmp, path);
}

static const char *observables_dir(int pbc)
{
    return pbc ? "data/h_null/observables/PBC" : "data/h_null/observables/OBC";
}

static int ensure_dir(const char *path)
{
    if (MKDIR(path) == 0)
        return 0;
    if (errno == EEXIST)
        return 0;
    fprintf(stderr, "[main_static] cannot create directory %s\n", path);
    return -1;
}

static int ensure_output_dirs(int pbc)
{
    if (ensure_dir("data") != 0) return -1;
    if (ensure_dir("data/h_null") != 0) return -1;
    if (ensure_dir("data/h_null/observables") != 0) return -1;
    if (ensure_dir(observables_dir(pbc)) != 0) return -1;
    return 0;
}

static void build_output_paths(int pbc, int L, char *gap_path,
                               size_t gap_path_sz, char *obs_path,
                               size_t obs_path_sz)
{
    if (pbc) {
        snprintf(gap_path, gap_path_sz, "%s/gap_L%02d.dat",
                 observables_dir(pbc), L);
        snprintf(obs_path, obs_path_sz, "%s/obs_L%02d.dat",
                 observables_dir(pbc), L);
    } else {
        snprintf(gap_path, gap_path_sz, "%s/gap_obc_L%02d.dat",
                 observables_dir(pbc), L);
        snprintf(obs_path, obs_path_sz, "%s/obs_obc_L%02d.dat",
                 observables_dir(pbc), L);
    }
}

static int file_header_matches(FILE *fp, const char *header_info)
{
    char line[1024];

    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '#')
            break;
        if (strstr(line, header_info))
            return 1;
    }

    return 0;
}

static int load_gap_file(const char *path, const char *header_info,
                         const double *g_arr, int N_g,
                         double (*evals_arr)[4])
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;

    if (!file_header_matches(fp, header_info)) {
        fclose(fp);
        return 0;
    }

    rewind(fp);
    char line[1024];
    int n = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        double g, e0, e1, e2, e3;
        if (sscanf(line, "%lf %lf %lf %lf %lf", &g, &e0, &e1, &e2, &e3) != 5)
            break;
        if (n >= N_g || fabs(g - g_arr[n]) > RESUME_G_TOL)
            break;

        evals_arr[n][0] = e0;
        evals_arr[n][1] = e1;
        evals_arr[n][2] = e2;
        evals_arr[n][3] = e3;
        n++;
    }

    fclose(fp);
    return n;
}

static int load_obs_file(const char *path, const char *header_info,
                         const double *g_arr, int N_g,
                         double (*obs_arr)[8])
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;

    if (!file_header_matches(fp, header_info)) {
        fclose(fp);
        return 0;
    }

    rewind(fp);
    char line[1024];
    int n = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        double g, mx, mz_sq, mz, chi_z, mz4, psi_t, psi_bar, binder;
        double chi_perp, g_chi_perp;
        int nr = sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                        &g, &mx, &mz_sq, &mz, &chi_z, &mz4, &psi_t, &psi_bar,
                        &binder, &chi_perp, &g_chi_perp);
        if (nr != 11)
            break;
        if (n >= N_g || fabs(g - g_arr[n]) > RESUME_G_TOL)
            break;

        obs_arr[n][0] = mx;
        obs_arr[n][1] = mz_sq;
        obs_arr[n][2] = mz4;
        obs_arr[n][3] = binder;
        (void)chi_z;
        obs_arr[n][4] = 0.0;
        obs_arr[n][5] = psi_t;
        obs_arr[n][6] = chi_perp;
        obs_arr[n][7] = psi_bar;
        n++;
    }

    fclose(fp);
    return n;
}

static int write_gap_file(const char *path, int L, double h, int pbc,
                          int N_g_total, const double *g_arr, int N_rows,
                          const double (*evals_arr)[4])
{
    char tmp[520];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    FILE *fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "[main_static] cannot open %s\n", tmp);
        return -1;
    }

    write_gap_header(fp, L, h, pbc, N_g_total);

    for (int ig = 0; ig < N_rows; ig++) {
        double gap = evals_arr[ig][1] - evals_arr[ig][0];
        fprintf(fp,
                "%.8f  %.12f  %.12f  %.12f  %.12f"
                "  %.12f  %.12f  %.12f\n",
                g_arr[ig], evals_arr[ig][0], evals_arr[ig][1],
                evals_arr[ig][2], evals_arr[ig][3],
                gap, gap * L, evals_arr[ig][0] / L);
    }

    if (fclose(fp) != 0) {
        remove(tmp);
        return -1;
    }
    if (replace_file(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static int write_obs_file(const char *path, int L, double h, int pbc,
                          int N_g_total, const double *g_arr, int N_rows,
                          const double (*obs_arr)[8])
{
    char tmp[520];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    FILE *fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "[main_static] cannot open %s\n", tmp);
        return -1;
    }

    write_obs_header(fp, L, h, pbc, N_g_total);

    for (int ig = 0; ig < N_rows; ig++) {
        double mz_sq = obs_arr[ig][1];
        double mz = sqrt(fabs(mz_sq));
        double chi_perp = obs_arr[ig][6];
        double g_chi_perp = g_arr[ig] * chi_perp;

        fprintf(fp,
                "%.8f  %.12f  %.12f  %.12f  %.12f"
                "  %.12f  %.12f  %.12f  %.12f  %.12f  %.12f\n",
                g_arr[ig], obs_arr[ig][0], mz_sq, mz, obs_arr[ig][4],
                obs_arr[ig][2], obs_arr[ig][5], obs_arr[ig][7],
                obs_arr[ig][3], chi_perp, g_chi_perp);
    }

    if (fclose(fp) != 0) {
        remove(tmp);
        return -1;
    }
    if (replace_file(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static long detect_cpu_count(void)
{
#if !defined(_WIN32) && defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : -1;
#else
    return -1;
#endif
}

static double detect_available_ram_gib(void)
{
#if !defined(_WIN32) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return -1.0;
    return (double)pages * (double)page_size /
           (1024.0 * 1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

static double estimate_peak_mem_gib(int L)
{
    double dim = (double)(1LL << L);
    double matrix_bytes = dim * dim * sizeof(double);
    double dsyevd_work_bytes =
        (1.0 + 6.0 * dim + 2.0 * dim * dim) * sizeof(double);
    double peak_bytes = 2.0 * matrix_bytes + dsyevd_work_bytes;
    return peak_bytes / (1024.0 * 1024.0 * 1024.0);
}

static double work_units_per_g(int L)
{
    double dim = (double)(1LL << L);
    return 6.0 * dim * dim * dim;
}

static void format_duration(double seconds, char *buf, size_t n)
{
    if (!isfinite(seconds) || seconds < 0.0) {
        snprintf(buf, n, "--:--:--");
        return;
    }

    long long s = (long long)(seconds + 0.5);
    long long h = s / 3600;
    long long m = (s % 3600) / 60;
    long long sec = s % 60;
    snprintf(buf, n, "%02lld:%02lld:%02lld", h, m, sec);
}

static void print_progress(int L, int done_g, int N_g,
                           const Progress *progress, double last_seconds)
{
    double pct_L = (N_g > 0) ? (100.0 * (double)done_g / (double)N_g) : 100.0;
    double pct_tot = (progress->total_units > 0.0)
                         ? (100.0 * progress->done_units /
                            progress->total_units)
                         : 100.0;
    double eta_seconds = NAN;
    double elapsed_seconds = difftime(time(NULL), progress->run_t0);

    if (progress->measured_units > 0.0 &&
        progress->total_units > progress->done_units) {
        double rate = progress->measured_units / fmax(elapsed_seconds, 1e-9);
        eta_seconds = (progress->total_units - progress->done_units) / rate;
    }

    char elapsed_buf[32], eta_buf[32], last_buf[32];
    format_duration(elapsed_seconds, elapsed_buf, sizeof(elapsed_buf));
    format_duration(eta_seconds, eta_buf, sizeof(eta_buf));
    format_duration(last_seconds, last_buf, sizeof(last_buf));

    printf("  progress L=%2d g=%3d/%3d  %6.2f%% L  %6.2f%% total"
           "  last=%s  elapsed=%s  ETA=%s\n",
           L, done_g, N_g, pct_L, pct_tot, last_buf, elapsed_buf, eta_buf);
    fflush(stdout);
}

static int compute_e0_ed(const Basis *b, double g, double h, double *e0)
{
    const long long dim = b->dim;
    double *Ham = calloc((size_t)dim * dim, sizeof(double));
    double *eig = malloc((size_t)dim * sizeof(double));
    if (!Ham || !eig) {
        free(Ham);
        free(eig);
        return -1;
    }

    build_ham(b, g, h, Ham);
    if (diagonalize(Ham, dim, eig) != 0) {
        free(Ham);
        free(eig);
        return -1;
    }

    *e0 = eig[0];
    free(Ham);
    free(eig);
    return 0;
}

static double compute_chi_perp_ed(const Basis *b, double g, double h,
                                  double dg)
{
    double em2, em1, e0, ep1, ep2;

    if (compute_e0_ed(b, g - 2.0 * dg, h, &em2) != 0) return 0.0 / 0.0;
    if (compute_e0_ed(b, g - dg, h, &em1) != 0) return 0.0 / 0.0;
    if (compute_e0_ed(b, g, h, &e0) != 0) return 0.0 / 0.0;
    if (compute_e0_ed(b, g + dg, h, &ep1) != 0) return 0.0 / 0.0;
    if (compute_e0_ed(b, g + 2.0 * dg, h, &ep2) != 0) return 0.0 / 0.0;

    double d2 = (-ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2) /
                (12.0 * dg * dg);
    return -d2 / b->L;
}

/* ============================================================
 *  ARGUMENT PARSING
 * ============================================================ */

static double parse_double(const char *s, const char *name)
{
    char *end; errno = 0;
    double v = strtod(s, &end);
    if (*end || errno || !isfinite(v)) {
        fprintf(stderr, "[ERROR] Invalid float <%s> = '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_pbc(const char *s)
{
    if (s[0] == '0' && !s[1]) return 0;
    if (s[0] == '1' && !s[1]) return 1;
    fprintf(stderr, "[ERROR] <pbc> must be 0 (OBC) or 1 (PBC)\n");
    exit(EXIT_FAILURE);
}

static int parse_L_value(const char *s)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (*end || errno || v < 1 || v > 28) {
        fprintf(stderr, "[ERROR] Invalid L value = '%s' (expected 1..28)\n", s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static SimParams parse_args(int argc, char **argv)
{
    SimParams P;
    P.resume = 0;
    P.N_L = 0;

    int arg0 = 1;
    if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
        P.resume = 1;
        arg0 = 2;
    }

    if (argc - arg0 < 2) {
        fprintf(stderr,
                "\nUsage: %s [--resume] <h> <pbc> [L1 L2 ...] [--resume]\n\n"
                "  h    longitudinal field (usually 0.0)\n"
                "  pbc    boundary conditions (0=OBC, 1=PBC)\n"
                "  L1..   optional system sizes (default from main_static.c)\n\n"
                "Examples:\n"
                "  %s 0.0 1              # PBC default L values\n"
                "  %s --resume 0.0 0     # OBC resume default L values\n"
                "  %s --resume 0.0 1 14  # PBC resume only L=14\n\n"
                "g-grid: adaptive x=(g-gc)*L^(1/nu), gc=%.1f, nu=%.1f\n"
                "        g in [%.1f, %.1f]\n\n",
                argv[0], argv[0], argv[0], argv[0],
                G_C, NU, G_MIN_PHYS, G_MAX_PHYS);
        exit(EXIT_FAILURE);
    }

    P.h = parse_double(argv[arg0], "h");
    P.pbc = parse_pbc(argv[arg0 + 1]);

    for (int i = arg0 + 2; i < argc; i++) {
        if (strcmp(argv[i], "--resume") == 0) {
            P.resume = 1;
            continue;
        }
        if (P.N_L >= MAX_L_RUNS) {
            fprintf(stderr, "[ERROR] too many L values (max %d)\n", MAX_L_RUNS);
            exit(EXIT_FAILURE);
        }
        P.L_values[P.N_L++] = parse_L_value(argv[i]);
    }

    if (P.N_L == 0) {
        const int *defaults = P.pbc ? L_PBC : L_OBC;
        int n_defaults = P.pbc ? N_L_PBC : N_L_OBC;
        for (int i = 0; i < n_defaults; i++)
            P.L_values[P.N_L++] = defaults[i];
    }

    return P;
}
