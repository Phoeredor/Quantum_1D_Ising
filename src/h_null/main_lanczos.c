/*
 * Lanczos-based gap and observable analysis for large L.
 * Computes the low-energy spectrum and ground-state observables of the
 * 1D quantum Ising chain using the Lanczos algorithm, enabling FSS
 * analysis at L = 14-24 (default run: 14-22).
 *
 * Features:
 *   - Adaptive g-grid (same scaling-aware sampling as main_static.c)
 *   - chi_z column kept as 0.0; finite-difference chi_z comes from main_chiz_fd.c
 *   - All observables at each g-point in a single pass
 *
 * Usage:
 *   ./ising_lanczos [--resume] h PBC [L1 L2 ...]
 *
 *   h       : longitudinal field (usually 0)
 *   PBC     : 1 = periodic, 0 = open
 *   L1 L2.. : system sizes to run (default: 14 16 18 20 22)
 *
 * Output files (data/h_null/observables/<BC>/):
 *   gap_lz_L<LL>.dat      (PBC)  |  gap_lz_obc_L<LL>.dat  (OBC)
 *   obs_lz_L<LL>.dat      (PBC)  |  obs_lz_obc_L<LL>.dat  (OBC)
 *
 * gap columns : g  E0  E1  E2  gap=E1-E0  gap*L  E0/L
 * obs columns : g  Mx  mz_sq  mz  chi_z  mz4  psi_tilde  psi_bar
 *               binder  chi_x  g_chi_x
 *
 * Hamiltonian convention:
 *   H = -sum_j s^z_j s^z_{j+1} - h*sum_j s^z_j - g*sum_j s^x_j
 */

#include <math.h>
#include <errno.h>
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
#include "basis_sector.h"
#include "hamiltonian_sector.h"
#include "lanczos.h"
#include "observables.h"

#define MAX_L_LZ 24
#define MAX_DIM_LZ (1LL << MAX_L_LZ) /* 16 777 216 */

static const int DEFAULT_L[] = {14, 16, 18, 20, 22};
static const int N_DEFAULT_L = 5;

/* ============================================================
 *  ADAPTIVE g-GRID (same as main_static.c)
 * ============================================================ */
#define G_C 1.0
#define NU 1.0
#define G_MIN_PHYS 0.4
#define G_MAX_PHYS 1.6

static const double ZONE_X1[] = {0.0, 1.5, 5.0};
static const double ZONE_X2[] = {1.5, 5.0, 12.0};
static const double ZONE_DX[] = {0.02, 0.10, 0.50};
static const int N_ZONES = 3;

static int build_g_grid(int L, double *g_arr, int max_pts) {
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
        if (g < G_MIN_PHYS || g > G_MAX_PHYS)
          continue;
        int dup = 0;
        for (int k = 0; k < n; k++)
          if (fabs(tmp[k] - g) < 1e-9) {
            dup = 1;
            break;
          }
        if (!dup && n < max_pts)
          tmp[n++] = g;
      }
      x += ZONE_DX[iz];
    }
  }

  for (int i = 0; i < n - 1; i++)
    for (int j = i + 1; j < n; j++)
      if (tmp[j] < tmp[i]) {
        double t = tmp[i];
        tmp[i] = tmp[j];
        tmp[j] = t;
      }

  memcpy(g_arr, tmp, (size_t)n * sizeof(double));
  return n;
}

/* 5-point centered stencil step for chi_x = -(1/L) d2E0/dg2 */
#define CHI_PERP_DG 1e-3

static double lz_E0(const Basis *b, double g, double h, int max_iter,
                    unsigned long seed, double *work_buf) {
  LanczosParams par = {
      .n_eig = 1,
      .max_iter = max_iter,
      .tol = 1e-10,
      .max_restarts = 80,
      .verbose = 0,
      .seed = seed,
  };

  double e0;
  int ret = lanczos(b, g, h, &par, &e0, work_buf);
  if (ret != 0 && ret != -3) {
    fprintf(stderr, "[chi_x] lanczos failed at g=%.8f h=%.8f (ret=%d)\n",
            g, h, ret);
    return 0.0 / 0.0;
  }
  return e0;
}

static double compute_chi_perp_lz(const Basis *b, double g, double h,
                                  int max_iter, unsigned long seed,
                                  double *work_buf) {
  const double dg = CHI_PERP_DG;

  /* Same seed across stencil points to keep Lanczos noise correlated. */
  unsigned long fixed_seed = seed;

  double em2 = lz_E0(b, g - 2.0 * dg, h, max_iter, fixed_seed, work_buf);
  double em1 = lz_E0(b, g - dg, h, max_iter, fixed_seed, work_buf);
  double e0 = lz_E0(b, g, h, max_iter, fixed_seed, work_buf);
  double ep1 = lz_E0(b, g + dg, h, max_iter, fixed_seed, work_buf);
  double ep2 = lz_E0(b, g + 2.0 * dg, h, max_iter, fixed_seed, work_buf);

  if (!isfinite(em2) || !isfinite(em1) || !isfinite(e0) || !isfinite(ep1) ||
      !isfinite(ep2))
    return 0.0 / 0.0;

  double d2 = (-ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2) /
              (12.0 * dg * dg);
  return -d2 / b->L;
}

static int replace_file(const char *tmp, const char *path) {
#ifdef _WIN32
  remove(path);
#endif
  return rename(tmp, path);
}

static const char *observables_dir(int pbc) {
  return pbc ? "data/h_null/observables/PBC" : "data/h_null/observables/OBC";
}

static int ensure_dir(const char *path) {
  if (MKDIR(path) == 0)
    return 0;
  if (errno == EEXIST)
    return 0;
  fprintf(stderr, "[main_lanczos] cannot create directory %s\n", path);
  return -1;
}

static int ensure_output_dirs(int pbc) {
  if (ensure_dir("data") != 0) return -1;
  if (ensure_dir("data/h_null") != 0) return -1;
  if (ensure_dir("data/h_null/observables") != 0) return -1;
  if (ensure_dir(observables_dir(pbc)) != 0) return -1;
  return 0;
}

/* ------------------------------------------------------------------ */
/* write_gap_file: g E0 E1 E2 gap gap*L E0/L                          */
/* ------------------------------------------------------------------ */
static int write_gap_file(const char *path, const char *header_info,
                          const double *g_arr, int N_g,
                          const double **evals_arr, /* [N_g][3] */
                          int L) {
  char tmp[520];
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);

  FILE *fp = fopen(tmp, "w");
  if (!fp) {
    fprintf(stderr, "[main_lanczos] cannot open %s\n", tmp);
    return -1;
  }

  time_t now = time(NULL);
  fprintf(fp,
          "# 1D Quantum Ising - Lanczos low-energy spectrum\n"
          "# %s\n"
          "# Generated: %s"
      "# Columns: g  E0  E1  E2  gap=E1-E0  gap*L  E0/L\n"
          "#\n",
          header_info, ctime(&now));

  for (int ig = 0; ig < N_g; ig++) {
    const double *ev = evals_arr[ig];
    double gap = ev[1] - ev[0];
    fprintf(fp,
            "%.10f  %+.12f  %+.12f  %+.12f  %.12f  %.12f  %.12f\n",
            g_arr[ig], ev[0], ev[1], ev[2], gap, gap * L, ev[0] / L);
  }

  fclose(fp);
  if (replace_file(tmp, path) != 0) {
    remove(tmp);
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* write_obs_file: g Mx mz_sq mz chi_z mz4 psi_tilde binder chi_x g*chi_x */
/* ------------------------------------------------------------------ */
static int write_obs_file(
    const char *path, const char *header_info, const double *g_arr, int N_g,
  const double (*obs)[8]) /* [N_g][8]: Mx,mz_sq,mz4,binder,chi_z,psi_t,chi_perp,psi_bar */
{
  char tmp[520];
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);

  FILE *fp = fopen(tmp, "w");
  if (!fp) {
    fprintf(stderr, "[main_lanczos] cannot open %s\n", tmp);
    return -1;
  }

  time_t now = time(NULL);
  fprintf(fp,
          "# 1D Quantum Ising - Lanczos Observables\n"
          "# %s\n"
          "# Generated: %s"
          "# Mx      = (1/L) sum_j <psi0|sigma^x_j|psi0>\n"
          "# mz_sq   = <psi0|(Mz/L)^2|psi0>\n"
          "# mz      = sqrt(mz_sq)\n"
          "# chi_z   = 0.0  [finite-difference data: data/h_null/chiz_fd/dh_5e-04/<BC>/chizfd*.dat]\n"
          "# mz4     = <psi0|(Mz/L)^4|psi0>\n"
          "# psi_t   = |<gs_even|Mz/L|gs_odd>|  (sector Lanczos, h=0; NaN "
          "otherwise)\n"
          "# psi_b = sum_ii |psi0[ii]|^2 * |sz_total(ii,L)| / L\n"
          "# binder  = mz4/mz_sq^2  (->3 PM, ->1 FM)\n"
          "# chi_x = -(1/L) d2E0/dg2  [5-pt stencil, dg=%.0e]\n"
          "# g_chi_x = g * chi_x  (quantum specific heat at T=0)\n"
          "# Columns: g  Mx  mz_sq  mz  chi_z  mz4  psi_tilde  psi_bar  binder"
          "  chi_x  g_chi_x\n"
          "#\n",
          header_info, ctime(&now), CHI_PERP_DG);

  for (int ig = 0; ig < N_g; ig++) {
    double mz_sq = obs[ig][1];
    double mz = sqrt(fabs(mz_sq));
    double chi_z = obs[ig][4];
      double chi_perp = obs[ig][6];
      double g_chi_perp = g_arr[ig] * chi_perp;

    fprintf(fp,
            "%.8f  %.12f  %.12f  %.12f  %.12f"
      "  %.12f  %.12f  %.12f  %.12f  %.12f  %.12f\n",
            g_arr[ig], obs[ig][0], /* Mx         */
            mz_sq,                 /* mz_sq      */
            mz,                    /* mz         */
            chi_z,                 /* chi_z      */
            obs[ig][2],            /* mz4        */
            obs[ig][5],            /* psi_tilde  */
      obs[ig][7],            /* psi_bar    */
        obs[ig][3],            /* binder     */
        chi_perp,              /* chi_perp   */
        g_chi_perp);           /* g*chi_perp */
  }

  fclose(fp);
  if (replace_file(tmp, path) != 0) {
    remove(tmp);
    return -1;
  }
  return 0;
}

/* ============================================================
 *  CHECKPOINT / RESUME / PROGRESS HELPERS
 * ============================================================ */
#define RESUME_G_TOL 5e-7

static int max_iter_for_L(int L) {
  if (L <= 16)
    return 200;
  if (L <= 18)
    return 150;
  if (L <= 20)
    return 100;
  return 60;
}

static int lanczos_calls_per_g(double h) {
  return (h == 0.0) ? 8 : 6;
}

static double work_units_per_g(int L, int max_iter, double h) {
  double dim = (double)(1LL << L);
  double full_calls = 6.0; /* spectrum + chi_x(5) */
  double sector_equiv = (h == 0.0) ? 1.0 : 0.0; /* 2 sector calls at dim/2 */
  return (double)max_iter * dim * (full_calls + sector_equiv);
}

static double estimate_peak_mem_gib(int L, int max_iter) {
  double dim = (double)(1LL << L);
  double bytes = ((double)max_iter + 6.0) * dim * sizeof(double);
  return bytes / (1024.0 * 1024.0 * 1024.0);
}

static void format_duration(double seconds, char *buf, size_t n) {
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

static long detect_cpu_count(void) {
#if !defined(_WIN32) && defined(_SC_NPROCESSORS_ONLN)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? n : -1;
#else
  return -1;
#endif
}

static double detect_available_ram_gib(void) {
#if !defined(_WIN32) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
  long pages = sysconf(_SC_AVPHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return -1.0;
  return (double)pages * (double)page_size / (1024.0 * 1024.0 * 1024.0);
#else
  return -1.0;
#endif
}

static void build_output_paths(int pbc, int L, char *gap_path,
                               size_t gap_path_sz, char *obs_path,
                               size_t obs_path_sz) {
  if (pbc) {
    snprintf(gap_path, gap_path_sz, "%s/gap_lz_L%02d.dat",
             observables_dir(pbc), L);
    snprintf(obs_path, obs_path_sz, "%s/obs_lz_L%02d.dat",
             observables_dir(pbc), L);
  } else {
    snprintf(gap_path, gap_path_sz, "%s/gap_lz_obc_L%02d.dat",
             observables_dir(pbc), L);
    snprintf(obs_path, obs_path_sz, "%s/obs_lz_obc_L%02d.dat",
             observables_dir(pbc), L);
  }
}

static int file_header_matches(FILE *fp, const char *header_info) {
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
                         const double *g_arr, int N_g, double **evals_arr) {
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

    double g, e0, e1, e2;
    if (sscanf(line, "%lf %lf %lf %lf", &g, &e0, &e1, &e2) != 4)
      break;
    if (n >= N_g || fabs(g - g_arr[n]) > RESUME_G_TOL)
      break;

    evals_arr[n][0] = e0;
    evals_arr[n][1] = e1;
    evals_arr[n][2] = e2;
    n++;
  }

  fclose(fp);
  return n;
}

static int load_obs_file(const char *path, const char *header_info,
                         const double *g_arr, int N_g, double (*obs)[8]) {
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

    obs[n][0] = mx;
    obs[n][1] = mz_sq;
    obs[n][2] = mz4;
    obs[n][3] = binder;
    (void)chi_z;
    obs[n][4] = 0.0;
    obs[n][5] = psi_t;
    obs[n][6] = chi_perp;
    obs[n][7] = psi_bar;
    n++;
  }

  fclose(fp);
  return n;
}

static void print_progress(int L, int done_g, int N_g, double done_units,
                           double total_units, double elapsed_seconds,
                           double measured_units, double last_seconds) {
  double pct_L = (N_g > 0) ? (100.0 * (double)done_g / (double)N_g) : 100.0;
  double pct_tot =
      (total_units > 0.0) ? (100.0 * done_units / total_units) : 100.0;
  double eta_seconds = NAN;

  if (measured_units > 0.0 && total_units > done_units) {
    double rate = measured_units / fmax(elapsed_seconds, 1e-9);
    eta_seconds = (total_units - done_units) / rate;
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

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
  int resume = 0;
  int arg0 = 1;

  if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
    resume = 1;
    arg0 = 2;
  }

  if (argc - arg0 < 2) {
    fprintf(stderr,
            "Usage: %s [--resume] h PBC [L1 L2 ...] [--resume]\n"
            "  h       : longitudinal field (usually 0)\n"
            "  PBC     : 1=periodic, 0=open\n"
            "  L1 L2.. : system sizes (default: 14 16 18 20 22)\n\n"
            "g-grid: adaptive x=(g-gc)*L^(1/nu), gc=%.1f, nu=%.1f\n"
            "        g in [%.1f, %.1f]\n",
            argv[0], G_C, NU, G_MIN_PHYS, G_MAX_PHYS);
    return EXIT_FAILURE;
  }

  double h = atof(argv[arg0]);
  int pbc = atoi(argv[arg0 + 1]);

  /* Parse optional L values */
  int L_list[16];
  int n_L = 0;
  for (int i = arg0 + 2; i < argc && n_L < 16; i++) {
    if (strcmp(argv[i], "--resume") == 0) {
      resume = 1;
      continue;
    }
    L_list[n_L++] = atoi(argv[i]);
  }

  if (n_L == 0) {
    for (int i = 0; i < N_DEFAULT_L; i++)
      L_list[n_L++] = DEFAULT_L[i];
  }

  int total_g_points = 0;
  int valid_L_count = 0;
  double total_units = 0.0;

  for (int i = 0; i < n_L; i++) {
    int L = L_list[i];
    if (L < 2 || L > MAX_L_LZ)
      continue;

    double g_tmp[2048];
    int N_g = build_g_grid(L, g_tmp, 2048);
    int max_iter_L = max_iter_for_L(L);

    total_g_points += N_g;
    total_units += (double)N_g * work_units_per_g(L, max_iter_L, h);
    valid_L_count++;
  }

  if (valid_L_count == 0) {
    fprintf(stderr, "[main_lanczos] no valid L in requested range [2,%d]\n",
            MAX_L_LZ);
    return EXIT_FAILURE;
  }

  /* Print run info */
  printf("=====================================================\n");
  printf("|  1D QUANTUM ISING - LANCZOS EXACT DIAGONALIZATION |\n");
  printf("=====================================================\n");
  printf(" g grid   = adaptive [%.1f, %.1f], gc=%.1f, nu=%.1f\n", G_MIN_PHYS,
         G_MAX_PHYS, G_C, NU);
  printf(" h   = %.4f\n", h);
  printf(" pbc      = %s\n", pbc ? "PBC" : "OBC");
  printf(" resume   = %s\n", resume ? "on" : "off");
  printf(" chi_z    = finite-difference data in data/h_null/chiz_fd/dh_5e-04/<BC>/chizfd*.dat\n");
  printf(" chi_x    = 5-pt E0 stencil, dg=%.0e\n", CHI_PERP_DG);
  printf(" calls/g  = %d Lanczos calls (%s sector calls)\n",
         lanczos_calls_per_g(h), (h == 0.0) ? "with" : "without");
  printf(" L values =");
  for (int i = 0; i < n_L; i++)
    printf(" %d", L_list[i]);
  printf("\n total g  = %d\n", total_g_points);

  long cpu_count = detect_cpu_count();
  double ram_avail = detect_available_ram_gib();
  if (cpu_count > 0)
    printf(" cpu      = %ld visible cores\n", cpu_count);
  if (ram_avail > 0.0)
    printf(" RAM      = %.2f GiB available now\n", ram_avail);
  printf(" ETA      = calibrated after the first completed new g-point\n");

  printf(" plan     =\n");
  for (int i = 0; i < n_L; i++) {
    int L = L_list[i];
    if (L < 2 || L > MAX_L_LZ)
      continue;

    double g_tmp[2048];
    int N_g = build_g_grid(L, g_tmp, 2048);
    int max_iter_L = max_iter_for_L(L);
    long long dim = 1LL << L;
    printf("   L=%2d  dim=%9lld  g-pts=%3d  max_iter=%3d"
           "  peak_RAM/proc~%.2f GiB\n",
           L, dim, N_g, max_iter_L, estimate_peak_mem_gib(L, max_iter_L));
  }
  printf("-----------------------------------------------------\n");
  fflush(stdout);

  time_t run_t0 = time(NULL);
  double done_units = 0.0;
  double measured_units = 0.0;

  if (ensure_output_dirs(pbc) != 0)
    return EXIT_FAILURE;

  /* ---- Loop over system sizes ---- */
  for (int iL = 0; iL < n_L; iL++) {
    int L = L_list[iL];

    if (L < 2 || L > MAX_L_LZ) {
      fprintf(stderr, "[main_lanczos] L=%d out of range [2,%d], skipping\n", L,
              MAX_L_LZ);
      continue;
    }

    long long dim = 1LL << L;

    /* Build adaptive g-grid for this L */
    double g_arr[2048];
    int N_g = build_g_grid(L, g_arr, 2048);

    /* L-adaptive Krylov dimension.
     * V requires (max_iter+1)*dim*8 bytes:
     *   L=22, iter=60: 61 *  4194304 * 8 = 1.91 GiB
     *   L=24, iter=60: 61 * 16777216 * 8 = 7.63 GiB             */
    int max_iter_L = max_iter_for_L(L);

    LanczosParams par = {
      .n_eig = 3,
        .max_iter = max_iter_L,
        .tol = 1e-10,
        .max_restarts = 80,
        .verbose = 0,
        .seed = 42UL,
    };

    char gap_path[512], obs_path[512], header[256];
    build_output_paths(pbc, L, gap_path, sizeof(gap_path), obs_path,
                       sizeof(obs_path));

    snprintf(header, sizeof(header),
             "L=%d  pbc=%d  h=%.6f  algorithm=Lanczos  "
             "max_iter=%d  tol=%.1e",
             L, pbc, h, par.max_iter, par.tol);

    printf("L=%2d  (dim=%9lld, %d g-pts, max_iter=%d)\n", L, dim, N_g,
           max_iter_L);
    fflush(stdout);

    /* Eigenvalue storage */
    double **evals_all = (double **)malloc((size_t)N_g * sizeof(double *));
    if (!evals_all) {
      fprintf(stderr, "\n[main_lanczos] malloc evals_all failed\n");
      continue;
    }
    for (int ig = 0; ig < N_g; ig++) {
      evals_all[ig] = (double *)malloc(3 * sizeof(double));
      if (!evals_all[ig]) {
        for (int j = 0; j < ig; j++)
          free(evals_all[j]);
        free(evals_all);
        evals_all = NULL;
        break;
      }
      for (int k = 0; k < 3; k++)
        evals_all[ig][k] = NAN;
    }
    if (!evals_all)
      continue;

    /* Observable storage: N_g rows x 8 columns
     * [0]=Mx  [1]=mz_sq  [2]=mz4  [3]=binder
     * [4]=chi_z  [5]=psi_tilde  [6]=chi_perp  [7]=psi_bar */
    double (*obs_all)[8] =
      (double (*)[8])malloc((size_t)N_g * 8 * sizeof(double));
    if (!obs_all) {
      fprintf(stderr, "\n[main_lanczos] malloc obs_all failed\n");
      for (int ig = 0; ig < N_g; ig++)
        free(evals_all[ig]);
      free(evals_all);
      continue;
    }
    for (int ig = 0; ig < N_g; ig++)
      for (int k = 0; k < 8; k++)
        obs_all[ig][k] = NAN;

    double units_g = work_units_per_g(L, max_iter_L, h);
    int start_ig = 0;

    if (resume) {
      int gap_done = load_gap_file(gap_path, header, g_arr, N_g, evals_all);
      int obs_done = load_obs_file(obs_path, header, g_arr, N_g, obs_all);
      start_ig = (gap_done < obs_done) ? gap_done : obs_done;

      if (gap_done != obs_done) {
        fprintf(stderr,
                "[main_lanczos] resume mismatch L=%d: gap=%d obs=%d; "
                "using %d safe rows\n",
                L, gap_done, obs_done, start_ig);
        fflush(stderr);

        if (start_ig > 0) {
          write_gap_file(gap_path, header, g_arr, start_ig,
                         (const double **)evals_all, L);
          write_obs_file(obs_path, header, g_arr, start_ig,
                         (const double (*)[8])obs_all);
        }
      }

      if (start_ig > 0) {
        printf("  resume: loaded %d/%d completed g-points\n", start_ig, N_g);
        fflush(stdout);
      }
    }

    if (start_ig >= N_g) {
      if (write_obs_file(obs_path, header, g_arr, N_g,
                         (const double (*)[8])obs_all) != 0) {
        fprintf(stderr,
                "[main_lanczos] cannot sanitize completed obs file for L=%d\n",
                L);
        for (int ig = 0; ig < N_g; ig++)
          free(evals_all[ig]);
        free(evals_all);
        free(obs_all);
        return EXIT_FAILURE;
      }
      done_units += (double)N_g * units_g;
      printf("  resume: L=%d already complete; sanitized obs chi_z column to 0.0\n",
             L);
      fflush(stdout);
      print_progress(L, N_g, N_g, done_units, total_units,
                     difftime(time(NULL), run_t0), measured_units, 0.0);

      for (int ig = 0; ig < N_g; ig++)
        free(evals_all[ig]);
      free(evals_all);
      free(obs_all);
      continue;
    }

    done_units += (double)start_ig * units_g;
    if (start_ig > 0)
      print_progress(L, start_ig, N_g, done_units, total_units,
                     difftime(time(NULL), run_t0), measured_units, 0.0);

    /* Basis */
    Basis b = basis_init(L, pbc);

    /* Eigenvector buffer: 3 vectors, reused across g points.
     * RAM: 3 * dim * 8 bytes  (L=22: 96 MB, L=24: 384 MB)       */
    double *evecs = (double *)malloc((size_t)3 * dim * sizeof(double));
    if (!evecs) {
      fprintf(stderr,
              "\n[main_lanczos] malloc evecs failed "
              "(L=%d, %.2f GB needed)\n",
              L, 3.0 * (double)dim * 8.0 / 1e9);
      for (int ig = 0; ig < N_g; ig++)
        free(evals_all[ig]);
      free(evals_all);
      free(obs_all);
      continue;
    }

    time_t t0 = time(NULL);

    /* ---- Loop over g values ---- */
    for (int ig = start_ig; ig < N_g; ig++) {
      time_t tg0 = time(NULL);
      double g = g_arr[ig];

      /* Vary seed per g to avoid systematic starting-vector bias */
      par.seed = 42UL + (unsigned long)ig * 7919UL;

      int ret = lanczos(&b, g, h, &par, evals_all[ig], evecs);

      if (ret != 0 && ret != -3) {
        /* Hard failure: fill NaN so the file row is flagged */
        for (int k = 0; k < 3; k++)
          evals_all[ig][k] = NAN;
        for (int k = 0; k < 8; k++)
          obs_all[ig][k] = NAN;
        obs_all[ig][4] = 0.0;
      } else {
        /* ret==0 (converged) or ret==-3 (partial, accept anyway) */
        double mx = obs_mx(&b, evecs);
        double mz2 = obs_mz_sq(&b, evecs);
        double mz4 = obs_mz4(&b, evecs);
        double U = obs_binder(mz2, mz4);

        double chi_z = 0.0;
        double chi_perp = compute_chi_perp_lz(
          &b, g, h, max_iter_L, 42UL + (unsigned long)ig * 7919UL, evecs);

        obs_all[ig][0] = mx;
        obs_all[ig][1] = mz2;
        obs_all[ig][2] = mz4;
        obs_all[ig][3] = U;
        obs_all[ig][4] = chi_z;

        /* psi_tilde via sector Lanczos at h = 0. */
        double psi_t = NAN; /* NaN default */
        if (h == 0.0) {
          long long sdim = 1LL << (L - 1);
          double *gs_even = (double *)malloc((size_t)sdim * sizeof(double));
          double *gs_odd = (double *)malloc((size_t)sdim * sizeof(double));
          if (gs_even && gs_odd) {
            BasisSector bs_even = basis_sector_init(L, pbc, 0);
            BasisSector bs_odd = basis_sector_init(L, pbc, 1);

            LanczosParams spar = {
                .n_eig = 1,
                .max_iter = max_iter_L,
                .tol = 1e-10,
                .max_restarts = 80,
                .verbose = 0,
                .seed = 42UL + (unsigned long)ig * 7919UL,
            };

            double e0_even, e0_odd;
            int r1 = lanczos_sector(&bs_even, g, &spar, &e0_even, gs_even);
            int r2 = lanczos_sector(&bs_odd, g, &spar, &e0_odd, gs_odd);

            if ((r1 == 0 || r1 == -3) && (r2 == 0 || r2 == -3))
              psi_t = obs_psi_tilde_sector(L, gs_even, gs_odd);
          }
          free(gs_even);
          free(gs_odd);
        }
        obs_all[ig][5] = psi_t;
        obs_all[ig][6] = chi_perp;
        obs_all[ig][7] = obs_psi_bar(&b, evecs);
      }

      if (write_gap_file(gap_path, header, g_arr, ig + 1,
                         (const double **)evals_all, L) != 0) {
        fprintf(stderr, "  [WARN] checkpoint gap write failed for L=%d ig=%d\n",
                L, ig);
        fflush(stderr);
      }

      if (write_obs_file(obs_path, header, g_arr, ig + 1,
                         (const double (*)[8])obs_all) != 0) {
        fprintf(stderr, "  [WARN] checkpoint obs write failed for L=%d ig=%d\n",
                L, ig);
        fflush(stderr);
      }

      time_t tg1 = time(NULL);
      done_units += units_g;
      measured_units += units_g;
      print_progress(L, ig + 1, N_g, done_units, total_units,
                     difftime(tg1, run_t0), measured_units,
                     difftime(tg1, tg0));
    }

    time_t t1 = time(NULL);
    printf("  L=%d completed in %.1f s\n", L, difftime(t1, t0));
    printf("  -> %s\n", gap_path);
    printf("  -> %s\n", obs_path);
    fflush(stdout);

    /* Free per-L buffers */
    free(evecs);
    for (int ig = 0; ig < N_g; ig++)
      free(evals_all[ig]);
    free(evals_all);
    free(obs_all);

  } /* end loop over L */

  printf("=====================================================\n");
  printf(">> Lanczos runs completed.\n");
  fflush(stdout);
  return EXIT_SUCCESS;
}
