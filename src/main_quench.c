/**
 * @file main_quench.c
 * @brief Full-spectrum ED quench dynamics of the 1D quantum Ising chain.
 *
 * Hamiltonian:
 *   H = - sum_j sigma^z_j sigma^z_{j+1}
 *       - g sum_j sigma^x_j
 *       - h sum_j sigma^z_j
 *
 * The ED benchmark workflow is:
 *   1. diagonalize H_i and save its ground state psi0;
 *   2. release/reuse the initial dense matrix;
 *   3. diagonalize H_f and keep the final eigenvectors;
 *   4. evolve psi0 in the H_f eigenbasis.
 *
 * Supported interfaces:
 *   Basic ED benchmark:
 *     ./ising_quench <L> <g_i> <g_f> <h_i> <h_f> <t_max> <N_t> <bc>
 *
 *   CQT dynamic FSS:
 *     ./ising_quench cqt <L> <bc> <g_pc> <beta_over_nu> <nu> <z> <y_h> <theta_max> <N_theta>
 *     ./ising_quench cqt <L> <bc> <beta_over_nu> <nu> <y_h> <theta_max> <N_theta>
 *       (short form assumes g_pc=1 and z=1)
 *
 *   FOQT h-quench dynamic scaling:
 *     ./ising_quench foqt <L> <bc> <g> <kappa0> <kappa1> <theta_max> <N_theta>
 *     ./ising_quench foqt <L> <bc> <g> <theta_max> <N_theta>
 *       (second form uses kappa0=+1, kappa1=-1)
 *
 *   Loschmidt echo CQT longitudinal quench:
 *     ./ising_quench loschmidt <L> <bc> <g_pc> <Phi> <y_h> <z> <theta_max> <N_theta>
 *     ./ising_quench loschmidt <L> <bc> <Phi> <y_h> <theta_max> <N_theta>
 *       (short form assumes g_pc=1 and z=1)
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "basis.h"
#include "diag.h"
#include "hamiltonian.h"

#ifndef MAX_DIM
#define MAX_DIM 4096
#endif

typedef enum {
    MODE_LEGACY = 0,
    MODE_CQT,
    MODE_FOQT,
    MODE_LOSCHMIDT
} QuenchMode;

typedef struct {
    double Mx;
    double Czz;
    double Mz2;
    double Mz_raw;
    double norm2;
} Observables;

typedef struct {
    int L;
    int pbc;
    int N_t;
    QuenchMode mode;

    double g_i;
    double g_f;
    double h_i;
    double h_f;
    double t_max;
    double theta_max;

    double g_pc;
    double beta_over_nu;
    double nu;
    double z;
    double y_h;

    double g_fixed;
    double kappa0;
    double kappa1;
    double Delta0;
    double m0;

    double Phi_i;
    double Phi_f;
    double delta_w;
} QuenchParams;

static QuenchParams parse_args(int argc, char **argv);
static void print_usage(const char *prog);
static double parse_double(const char *s, const char *name);
static int parse_int_min(const char *s, const char *name, int min_value);
static int parse_L(const char *s);
static int parse_bc(const char *s);
static const char *bc_name(int pbc);
static const char *bc_label(int pbc);
static const char *mode_name(QuenchMode mode);
static int ensure_output_dirs(void);

static int prepare_foqt_params(QuenchParams *Q);
static int compute_delta0(int L, int pbc, double g, double *Delta0);
static int run_quench(const QuenchParams *Q);
static Observables measure_observables(const Basis *b,
                                       const double *psi_re,
                                       const double *psi_im);
static void output_path(const QuenchParams *Q, char *fname, size_t n);
static void write_header(FILE *fp,
                         const QuenchParams *Q,
                         double E0_i,
                         double E0_f,
                         double overlap_norm2,
                         double E_quench);
static int is_mode(const char *s, const char *mode);

int main(int argc, char **argv)
{
    QuenchParams Q = parse_args(argc, argv);

    if (ensure_output_dirs() != 0)
        return EXIT_FAILURE;

    if (Q.mode == MODE_FOQT && prepare_foqt_params(&Q) != 0)
        return EXIT_FAILURE;

    printf("===================================================\n");
    printf("|   1D QUANTUM ISING - QUENCH DYNAMICS           |\n");
    printf("===================================================\n");
    printf(" mode  = %s\n", mode_name(Q.mode));
    printf(" L     = %d  (dim = %lld, MAX_DIM = %d)\n",
           Q.L, 1LL << Q.L, MAX_DIM);
    printf(" bc    = %s\n", bc_label(Q.pbc));
    printf(" g_i   = %.17g  g_f = %.17g\n", Q.g_i, Q.g_f);
    printf(" h_i   = %.17g  h_f = %.17g\n", Q.h_i, Q.h_f);
    if (Q.mode == MODE_CQT) {
        printf(" theta = t/L^z, theta_max = %.17g, N_theta = %d\n",
               Q.theta_max, Q.N_t);
        printf(" g_pc = %.17g  beta/nu = %.17g  nu = %.17g  z = %.17g  y_h = %.17g\n",
               Q.g_pc, Q.beta_over_nu, Q.nu, Q.z, Q.y_h);
    } else if (Q.mode == MODE_FOQT) {
        printf(" theta = Delta0*t, theta_max = %.12g, N_theta = %d\n",
               Q.theta_max, Q.N_t);
        printf(" Delta0 = %.15e  m0 = %.15e  kappa: %.6g -> %.6g\n",
               Q.Delta0, Q.m0, Q.kappa0, Q.kappa1);
    } else if (Q.mode == MODE_LOSCHMIDT) {
        printf(" theta = t/L^z, theta_max = %.17g, N_theta = %d\n",
               Q.theta_max, Q.N_t);
        printf(" g_pc = %.17g  Phi_i = %.17g  Phi_f = %.17g  delta_w = %.17g  z = %.17g  y_h = %.17g\n",
               Q.g_pc, Q.Phi_i, Q.Phi_f, Q.delta_w, Q.z, Q.y_h);
    } else {
        printf(" t_max = %.12g  N_t = %d\n", Q.t_max, Q.N_t);
    }
    printf("---------------------------------------------------\n");

    clock_t t0 = clock();
    int status = run_quench(&Q);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf(" Elapsed: %.2f s\n", elapsed);

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}

static int run_quench(const QuenchParams *Q)
{
    Basis b = basis_init(Q->L, Q->pbc);
    const long long dim = b.dim;
    const int L = Q->L;

    if (dim > MAX_DIM) {
        fprintf(stderr,
                "[ERROR] dim=%lld > MAX_DIM=%d. Rebuild with "
                "make quench QUENCH_MAX_DIM=<dim> if you have enough RAM.\n",
                dim, MAX_DIM);
        return 1;
    }

    const size_t dim_sz = (size_t)dim;
    const size_t mat_elems = dim_sz * dim_sz;
    const size_t mat_bytes = mat_elems * sizeof(double);

    double *Ham    = calloc(mat_elems, sizeof(double));
    double *eig    = malloc(dim_sz * sizeof(double));
    double *eig_f  = malloc(dim_sz * sizeof(double));
    double *psi0   = malloc(dim_sz * sizeof(double));
    double *cn     = malloc(dim_sz * sizeof(double));
    double *psi_re = malloc(dim_sz * sizeof(double));
    double *psi_im = malloc(dim_sz * sizeof(double));

    if (!Ham || !eig || !eig_f || !psi0 || !cn || !psi_re || !psi_im) {
        fprintf(stderr,
                "[ERROR] Allocation failed. Dense matrix alone is %.1f MB.\n",
                (double)mat_bytes / (1024.0 * 1024.0));
        goto cleanup_fail;
    }

    build_ham(&b, Q->g_i, Q->h_i, Ham);
    if (diagonalize(Ham, dim, eig) != 0)
        goto cleanup_fail;

    const double E0_i = eig[0];
    memcpy(psi0, Ham, dim_sz * sizeof(double));

    memset(Ham, 0, mat_bytes);
    build_ham(&b, Q->g_f, Q->h_f, Ham);
    if (diagonalize(Ham, dim, eig_f) != 0)
        goto cleanup_fail;

    const double E0_f = eig_f[0];

    for (long long n = 0; n < dim; n++) {
        double s = 0.0;
        const double *un = Ham + (size_t)n * dim_sz;
        for (long long ii = 0; ii < dim; ii++)
            s += un[ii] * psi0[ii];
        cn[n] = s;
    }

    double overlap_norm2 = 0.0;
    for (long long n = 0; n < dim; n++)
        overlap_norm2 += cn[n] * cn[n];

    if (fabs(overlap_norm2 - 1.0) > 1e-10) {
        fprintf(stderr,
                "[WARN] sum_n |c_n|^2 = %.15f (dev = %+.3e)\n",
                overlap_norm2, overlap_norm2 - 1.0);
    }

    double E_quench = 0.0;
    for (long long n = 0; n < dim; n++)
        E_quench += cn[n] * cn[n] * eig_f[n];
    E_quench /= (double)L;

    char fname[256];
    output_path(Q, fname, sizeof(fname));

    FILE *fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open %s: %s\n", fname, strerror(errno));
        goto cleanup_fail;
    }

    write_header(fp, Q, E0_i, E0_f, overlap_norm2, E_quench);

    const double dt = (Q->mode == MODE_LEGACY)
        ? Q->t_max / (double)(Q->N_t - 1)
        : 0.0;
    const double dtheta = (Q->mode == MODE_LEGACY)
        ? 0.0
        : Q->theta_max / (double)(Q->N_t - 1);
    const double Lz = pow((double)L, Q->z);

    double max_norm_dev = 0.0;
    double max_E_dev = 0.0;
    double max_loschmidt_excess = 0.0;
    double loschmidt_t0_dev = 0.0;

    for (int it = 0; it < Q->N_t; it++) {
        double theta = 0.0;
        double t = 0.0;

        if (Q->mode == MODE_CQT || Q->mode == MODE_LOSCHMIDT) {
            theta = it * dtheta;
            t = theta * Lz;
        } else if (Q->mode == MODE_FOQT) {
            theta = it * dtheta;
            t = theta / Q->Delta0;
        } else {
            t = it * dt;
        }

        memset(psi_re, 0, dim_sz * sizeof(double));
        memset(psi_im, 0, dim_sz * sizeof(double));

        double A_re = 0.0;
        double A_im = 0.0;

        for (long long n = 0; n < dim; n++) {
            const double phase = eig_f[n] * t;
            const double phase_re =  cn[n] * cos(phase);
            const double phase_im = -cn[n] * sin(phase);
            const double *un = Ham + (size_t)n * dim_sz;
            const double cn2 = cn[n] * cn[n];
            const double echo_phase = (eig_f[n] - E0_f) * t;

            A_re += cn2 * cos(echo_phase);
            A_im -= cn2 * sin(echo_phase);

            for (long long ii = 0; ii < dim; ii++) {
                psi_re[ii] += phase_re * un[ii];
                psi_im[ii] += phase_im * un[ii];
            }
        }

        Observables obs = measure_observables(&b, psi_re, psi_im);
        const double norm_dev = fabs(obs.norm2 - 1.0);
        if (norm_dev > max_norm_dev)
            max_norm_dev = norm_dev;

        const double E_t = E_quench;
        const double E_dev = fabs(E_t - E_quench);
        if (E_dev > max_E_dev)
            max_E_dev = E_dev;

        const double loschmidt_echo = A_re * A_re + A_im * A_im;
        const double Q_echo = -log(fmax(loschmidt_echo, 1e-300));
        if (it == 0)
            loschmidt_t0_dev = fabs(loschmidt_echo - 1.0);
        if (loschmidt_echo > 1.0) {
            const double excess = loschmidt_echo - 1.0;
            if (excess > max_loschmidt_excess)
                max_loschmidt_excess = excess;
        }

        if (Q->mode == MODE_CQT) {
            const double psi_scaled = obs.Mz_raw * pow((double)L, Q->beta_over_nu);
            fprintf(fp,
                    "%.12g  %.12g  %d  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g\n",
                    theta, t, L, Q->g_i, Q->g_f, Q->h_i, Q->h_f,
                    obs.Mz_raw, psi_scaled, obs.Mx, obs.Mz2, E_t, obs.norm2,
                    A_re, A_im, loschmidt_echo, Q_echo);
        } else if (Q->mode == MODE_FOQT) {
            const double psi_over_m0 = obs.Mz_raw / Q->m0;
            fprintf(fp,
                    "%.12g  %.12g  %d  %.15g  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g\n",
                    theta, t, L, Q->g_fixed, Q->Delta0, Q->m0,
                    Q->h_i, Q->h_f, Q->kappa0, Q->kappa1,
                    obs.Mz_raw, psi_over_m0, obs.Mx, obs.Mz2, E_t, obs.norm2,
                    A_re, A_im, loschmidt_echo, Q_echo);
        } else if (Q->mode == MODE_LOSCHMIDT) {
            fprintf(fp,
                    "%.12g  %.12g  %d  %.15g  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g\n",
                    theta, t, L, Q->Phi_i, Q->Phi_f, Q->delta_w,
                    A_re, A_im, loschmidt_echo, Q_echo, obs.norm2, E_t);
        } else {
            fprintf(fp,
                    "%.12g  %.15g  %.15g  %.15g  %.15g  %.15g  %.15g  "
                    "%.15g  %.15g  %.15g  %.15g\n",
                    t, obs.Mx, obs.Czz, obs.Mz2, obs.Mz_raw, E_t, obs.norm2,
                    A_re, A_im, loschmidt_echo, Q_echo);
        }
    }

    if (max_norm_dev > 1e-6) {
        fprintf(stderr,
                "[WARN] max norm drift = %.3e exceeds 1e-6\n",
                max_norm_dev);
    } else if (max_norm_dev > 1e-8) {
        fprintf(stderr,
                "[WARN] max norm drift = %.3e exceeds 1e-8\n",
                max_norm_dev);
    }

    if (max_E_dev > 1e-12) {
        fprintf(stderr,
                "[WARN] max energy drift = %.3e (E should be constant)\n",
                max_E_dev);
    }

    if (loschmidt_t0_dev > 1e-10) {
        fprintf(stderr,
                "[WARN] |Loschmidt_echo(t=0)-1| = %.3e exceeds 1e-10\n",
                loschmidt_t0_dev);
    }

    if (max_loschmidt_excess > 1e-10) {
        fprintf(stderr,
                "[WARN] max Loschmidt_echo-1 = %.3e exceeds 1e-10\n",
                max_loschmidt_excess);
    }

    fclose(fp);
    printf("  -> Wrote %s\n", fname);

    free(Ham);
    free(eig);
    free(eig_f);
    free(psi0);
    free(cn);
    free(psi_re);
    free(psi_im);
    return 0;

cleanup_fail:
    free(Ham);
    free(eig);
    free(eig_f);
    free(psi0);
    free(cn);
    free(psi_re);
    free(psi_im);
    return 1;
}

static Observables measure_observables(const Basis *b,
                                       const double *psi_re,
                                       const double *psi_im)
{
    const long long dim = b->dim;
    const int L = b->L;
    const int r = L / 2;
    Observables obs;
    obs.Mx = 0.0;
    obs.Czz = 0.0;
    obs.Mz2 = 0.0;
    obs.Mz_raw = 0.0;
    obs.norm2 = 0.0;

    for (int j = 0; j < L; j++) {
        for (long long ii = 0; ii < dim; ii++) {
            const long long exc = flip(ii, j);
            obs.Mx += psi_re[ii] * psi_re[exc] + psi_im[ii] * psi_im[exc];
        }
    }
    obs.Mx /= (double)L;

    for (long long ii = 0; ii < dim; ii++) {
        const double prob = psi_re[ii] * psi_re[ii] + psi_im[ii] * psi_im[ii];
        const double mz = (double)sz_total(ii, L) / (double)L;

        obs.norm2 += prob;
        obs.Czz += prob * (double)(sz_val(ii, 0) * sz_val(ii, r));
        obs.Mz2 += prob * mz * mz;
        obs.Mz_raw += prob * mz;
    }

    return obs;
}

static int prepare_foqt_params(QuenchParams *Q)
{
    if (Q->g_fixed <= 0.0 || Q->g_fixed >= 1.0) {
        fprintf(stderr, "[ERROR] FOQT requires 0 < g < 1, got g=%.12g\n",
                Q->g_fixed);
        return 1;
    }

    if (compute_delta0(Q->L, Q->pbc, Q->g_fixed, &Q->Delta0) != 0)
        return 1;

    if (!(Q->Delta0 > 0.0) || !isfinite(Q->Delta0)) {
        fprintf(stderr, "[ERROR] invalid Delta0=%.15e\n", Q->Delta0);
        return 1;
    }

    Q->m0 = pow(1.0 - Q->g_fixed * Q->g_fixed, 1.0 / 8.0);
    Q->g_i = Q->g_fixed;
    Q->g_f = Q->g_fixed;
    Q->h_i = Q->kappa0 * Q->Delta0 / (2.0 * Q->m0 * (double)Q->L);
    Q->h_f = Q->kappa1 * Q->Delta0 / (2.0 * Q->m0 * (double)Q->L);
    Q->t_max = Q->theta_max / Q->Delta0;

    return 0;
}

static int compute_delta0(int L, int pbc, double g, double *Delta0)
{
    Basis b = basis_init(L, pbc);
    const long long dim = b.dim;

    if (dim > MAX_DIM) {
        fprintf(stderr,
                "[ERROR] Delta0 solve dim=%lld > MAX_DIM=%d\n",
                dim, MAX_DIM);
        return 1;
    }

    const size_t dim_sz = (size_t)dim;
    double *Ham = calloc(dim_sz * dim_sz, sizeof(double));
    double *eig = malloc(dim_sz * sizeof(double));

    if (!Ham || !eig) {
        fprintf(stderr, "[ERROR] Allocation failed while computing Delta0\n");
        free(Ham);
        free(eig);
        return 1;
    }

    build_ham(&b, g, 0.0, Ham);
    if (diagonalize(Ham, dim, eig) != 0) {
        free(Ham);
        free(eig);
        return 1;
    }

    *Delta0 = eig[1] - eig[0];

    free(Ham);
    free(eig);
    return 0;
}

static void output_path(const QuenchParams *Q, char *fname, size_t n)
{
    if (Q->mode == MODE_CQT) {
        snprintf(fname, n, "data/quench/cqt/quench_cqt_%s_L%02d.dat",
                 bc_name(Q->pbc), Q->L);
    } else if (Q->mode == MODE_FOQT) {
        snprintf(fname, n, "data/quench/foqt/quench_foqt_%s_g%.3f_L%02d.dat",
                 bc_name(Q->pbc), Q->g_fixed, Q->L);
    } else if (Q->mode == MODE_LOSCHMIDT) {
        snprintf(fname, n, "data/quench/loschmidt/loschmidt_%s_Phi%.3f_L%02d.dat",
                 bc_name(Q->pbc), Q->Phi_f, Q->L);
    } else {
        snprintf(fname, n,
                 "data/quench/quench_L%02d_gi%.3f_gf%.3f_hi%.4f_hf%.4f.dat",
                 Q->L, Q->g_i, Q->g_f, Q->h_i, Q->h_f);
    }
}

static void write_header(FILE *fp,
                         const QuenchParams *Q,
                         double E0_i,
                         double E0_f,
                         double overlap_norm2,
                         double E_quench)
{
    time_t now = time(NULL);

    fprintf(fp,
            "# 1D Quantum Ising -- full-spectrum ED quench dynamics\n"
            "# Generated: %s"
            "# Hamiltonian: H=-sum_j sz_j sz_{j+1}-g sum_j sx_j-h sum_j sz_j\n"
            "# L=%d  bc=%s  pbc=%d  MAX_DIM=%d\n"
            "# g_i=%.17g  g_f=%.17g  h_i=%.17g  h_f=%.17g\n"
            "# E0_i=%.17g  E0_f=%.17g  E_quench_per_site=%.17g\n"
            "# overlap_sum_cn2=%.17g\n",
            ctime(&now), Q->L, bc_label(Q->pbc), Q->pbc, MAX_DIM,
            Q->g_i, Q->g_f, Q->h_i, Q->h_f,
            E0_i, E0_f, E_quench, overlap_norm2);

    if (Q->mode == MODE_CQT) {
        fprintf(fp,
                "# mode=CQT\n"
                "# kappa_g0=-1  kappa_g1=+1  kappa_h0=+1  kappa_h1=-1\n"
                "# g_pc=%.17g  beta_over_nu=%.17g  nu=%.17g  z=%.17g  y_h=%.17g\n"
                "# g0=g_pc-L^(-1/nu)  g1=g_pc+L^(-1/nu)\n"
                "# h0=L^(-y_h)  h1=-L^(-y_h)\n"
                "# theta=t/L^z  t=theta*L^z  theta_max=%.17g  N_theta=%d\n"
                "# scaling constants source: values passed by run_quench_ed.sh\n"
                "# Mz_raw=<Mz/L> signed; no Z2 pseudo-order parameter is used.\n"
                "# Psi_scaled=Mz_raw*L^(beta_over_nu)\n"
                "# A(t)=sum_n |c_n|^2 exp[-i(E_n-E0_f)t]\n"
                "# Columns: theta  t  L  g0  g1  h0  h1  Mz_raw  Psi_scaled  Mx  Mz2  E  norm2  A_re  A_im  Loschmidt_echo  Q_echo\n"
                "#\n",
                Q->g_pc, Q->beta_over_nu, Q->nu, Q->z, Q->y_h,
                Q->theta_max, Q->N_t);
    } else if (Q->mode == MODE_FOQT) {
        fprintf(fp,
                "# mode=FOQT\n"
                "# Delta0=E1-E0 of H(g,h=0) for the same L and BC.\n"
                "# m0=(1-g^2)^(1/8)\n"
                "# h0=kappa0*Delta0/(2*m0*L)  h1=kappa1*Delta0/(2*m0*L)\n"
                "# g=%.17g  Delta0=%.17g  m0=%.17g\n"
                "# kappa0=%.17g  kappa1=%.17g\n"
                "# theta=Delta0*t  t=theta/Delta0  theta_max=%.17g  N_theta=%d\n"
                "# Mz_raw=<Mz/L> signed; no Z2 pseudo-order parameter is used.\n"
                "# Psi_over_m0=Mz_raw/m0\n"
                "# A(t)=sum_n |c_n|^2 exp[-i(E_n-E0_f)t]\n"
                "# Columns: theta  t  L  g  Delta0  m0  h0  h1  kappa0  kappa1  Mz_raw  Psi_over_m0  Mx  Mz2  E  norm2  A_re  A_im  Loschmidt_echo  Q_echo\n"
                "#\n",
                Q->g_fixed, Q->Delta0, Q->m0,
                Q->kappa0, Q->kappa1, Q->theta_max, Q->N_t);
    } else if (Q->mode == MODE_LOSCHMIDT) {
        fprintf(fp,
                "# mode=LOSCHMIDT\n"
                "# Longitudinal CQT soft quench at g_i=g_f=g_pc.\n"
                "# Phi_i=h_i*L^(y_h)  Phi_f=h_f*L^(y_h)  delta_w=Phi_f/Phi_i-1\n"
                "# g_pc=%.17g  z=%.17g  Phi_i=%.17g  Phi_f=%.17g  delta_w=%.17g  y_h=%.17g\n"
                "# theta=t/L^z  t=theta*L^z  theta_max=%.17g  N_theta=%d\n"
                "# scaling constants source: values passed by run_quench_ed.sh\n"
                "# A(t)=sum_n |c_n|^2 exp[-i(E_n-E0_f)t]\n"
                "# Loschmidt_echo=|A(t)|^2  Q_echo=-log(max(Loschmidt_echo,1e-300))\n"
                "# Columns: theta  t  L  Phi_i  Phi_f  delta_w  A_re  A_im  Loschmidt_echo  Q_echo  norm2  E\n"
                "#\n",
                Q->g_pc, Q->z, Q->Phi_i, Q->Phi_f, Q->delta_w, Q->y_h,
                Q->theta_max, Q->N_t);
    } else {
        fprintf(fp,
                "# mode=legacy\n"
                "# Mx=<sum_j sx_j/L>; Czz=<sz_0 sz_{L/2}>; Mz2=<(Mz/L)^2>\n"
                "# Mz_raw=<Mz/L> signed; E=<H_f>/L is conserved.\n"
                "# A(t)=sum_n |c_n|^2 exp[-i(E_n-E0_f)t]\n"
                "# Columns: t  Mx  Czz  Mz2  Mz_raw  E  norm2  A_re  A_im  Loschmidt_echo  Q_echo\n"
                "#\n");
    }
}

static int ensure_output_dirs(void)
{
    const char *dirs[] = {
        "data",
        "data/quench",
        "data/quench/cqt",
        "data/quench/foqt",
        "data/quench/loschmidt",
        "plots",
        "plots/quench"
    };

    for (size_t i = 0; i < sizeof(dirs) / sizeof(dirs[0]); i++) {
        if (mkdir(dirs[i], 0775) != 0 && errno != EEXIST) {
            fprintf(stderr, "[ERROR] mkdir %s failed: %s\n",
                    dirs[i], strerror(errno));
            return 1;
        }
    }
    return 0;
}

static double parse_double(const char *s, const char *name)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (*end || errno || !isfinite(v)) {
        fprintf(stderr, "[ERROR] Invalid float <%s> = '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_int_min(const char *s, const char *name, int min_value)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (*end || errno || v < min_value) {
        fprintf(stderr, "[ERROR] <%s> must be an integer >= %d\n",
                name, min_value);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static int parse_L(const char *s)
{
    int L = parse_int_min(s, "L", 1);
    if (L > 28) {
        fprintf(stderr, "[ERROR] L=%d out of range [1,28]\n", L);
        exit(EXIT_FAILURE);
    }
    return L;
}

static int parse_bc(const char *s)
{
    if ((s[0] == '0' && s[1] == '\0') ||
        strcmp(s, "obc") == 0 || strcmp(s, "OBC") == 0)
        return 0;

    if ((s[0] == '1' && s[1] == '\0') ||
        strcmp(s, "pbc") == 0 || strcmp(s, "PBC") == 0)
        return 1;

    fprintf(stderr, "[ERROR] <bc> must be 0/OBC/obc or 1/PBC/pbc\n");
    exit(EXIT_FAILURE);
}

static const char *bc_name(int pbc)
{
    return pbc ? "pbc" : "obc";
}

static const char *bc_label(int pbc)
{
    return pbc ? "PBC" : "OBC";
}

static const char *mode_name(QuenchMode mode)
{
    switch (mode) {
        case MODE_CQT:
            return "CQT";
        case MODE_FOQT:
            return "FOQT";
        case MODE_LOSCHMIDT:
            return "Loschmidt";
        case MODE_LEGACY:
        default:
            return "legacy";
    }
}

static int is_mode(const char *s, const char *mode)
{
    return strcmp(s, mode) == 0;
}

static QuenchParams parse_args(int argc, char **argv)
{
    QuenchParams Q;
    memset(&Q, 0, sizeof(Q));
    Q.kappa0 = +1.0;
    Q.kappa1 = -1.0;
    Q.g_pc = 1.0;
    Q.z = 1.0;

    if (argc >= 2 && is_mode(argv[1], "cqt")) {
        if (argc != 9 && argc != 11) {
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }

        Q.mode = MODE_CQT;
        Q.L = parse_L(argv[2]);
        Q.pbc = parse_bc(argv[3]);
        if (argc == 9) {
            fprintf(stderr, "[WARN] short CQT interface: assuming g_pc=1 and z=1\n");
            Q.g_pc = 1.0;
            Q.beta_over_nu = parse_double(argv[4], "beta_over_nu");
            Q.nu = parse_double(argv[5], "nu");
            Q.z = 1.0;
            Q.y_h = parse_double(argv[6], "y_h");
            Q.theta_max = parse_double(argv[7], "theta_max");
            Q.N_t = parse_int_min(argv[8], "N_theta", 2);
        } else {
            Q.g_pc = parse_double(argv[4], "g_pc");
            Q.beta_over_nu = parse_double(argv[5], "beta_over_nu");
            Q.nu = parse_double(argv[6], "nu");
            Q.z = parse_double(argv[7], "z");
            Q.y_h = parse_double(argv[8], "y_h");
            Q.theta_max = parse_double(argv[9], "theta_max");
            Q.N_t = parse_int_min(argv[10], "N_theta", 2);
        }

        if (Q.nu <= 0.0 || Q.z <= 0.0 || Q.y_h <= 0.0 || Q.theta_max <= 0.0) {
            fprintf(stderr, "[ERROR] nu, z, y_h and theta_max must be > 0\n");
            exit(EXIT_FAILURE);
        }

        const double Ld = (double)Q.L;
        const double dg = pow(Ld, -1.0 / Q.nu);
        const double dh = pow(Ld, -Q.y_h);
        Q.g_i = Q.g_pc - dg;
        Q.g_f = Q.g_pc + dg;
        Q.h_i = +dh;
        Q.h_f = -dh;
        Q.t_max = Q.theta_max * pow(Ld, Q.z);
        return Q;
    }

    if (argc >= 2 && is_mode(argv[1], "foqt")) {
        if (argc != 7 && argc != 9) {
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }

        Q.mode = MODE_FOQT;
        Q.L = parse_L(argv[2]);
        Q.pbc = parse_bc(argv[3]);
        Q.g_fixed = parse_double(argv[4], "g_fixed");

        if (argc == 7) {
            Q.theta_max = parse_double(argv[5], "theta_max");
            Q.N_t = parse_int_min(argv[6], "N_theta", 2);
        } else {
            Q.kappa0 = parse_double(argv[5], "kappa0");
            Q.kappa1 = parse_double(argv[6], "kappa1");
            Q.theta_max = parse_double(argv[7], "theta_max");
            Q.N_t = parse_int_min(argv[8], "N_theta", 2);
        }

        if (Q.theta_max <= 0.0) {
            fprintf(stderr, "[ERROR] theta_max must be > 0\n");
            exit(EXIT_FAILURE);
        }
        return Q;
    }

    if (argc >= 2 && is_mode(argv[1], "loschmidt")) {
        if (argc != 8 && argc != 10) {
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }

        Q.mode = MODE_LOSCHMIDT;
        Q.L = parse_L(argv[2]);
        Q.pbc = parse_bc(argv[3]);
        if (argc == 8) {
            fprintf(stderr, "[WARN] short Loschmidt interface: assuming g_pc=1 and z=1\n");
            Q.g_pc = 1.0;
            Q.Phi_f = parse_double(argv[4], "Phi");
            Q.y_h = parse_double(argv[5], "y_h");
            Q.z = 1.0;
            Q.theta_max = parse_double(argv[6], "theta_max");
            Q.N_t = parse_int_min(argv[7], "N_theta", 2);
        } else {
            Q.g_pc = parse_double(argv[4], "g_pc");
            Q.Phi_f = parse_double(argv[5], "Phi");
            Q.y_h = parse_double(argv[6], "y_h");
            Q.z = parse_double(argv[7], "z");
            Q.theta_max = parse_double(argv[8], "theta_max");
            Q.N_t = parse_int_min(argv[9], "N_theta", 2);
        }

        if (Q.Phi_f <= 0.0 || Q.y_h <= 0.0 || Q.z <= 0.0 || Q.theta_max <= 0.0) {
            fprintf(stderr, "[ERROR] Phi, y_h, z and theta_max must be > 0\n");
            exit(EXIT_FAILURE);
        }

        const double Ld = (double)Q.L;
        const double h_scale = pow(Ld, -Q.y_h);
        Q.Phi_i = -Q.Phi_f;
        Q.delta_w = Q.Phi_f / Q.Phi_i - 1.0;
        Q.g_i = Q.g_pc;
        Q.g_f = Q.g_pc;
        Q.h_i = Q.Phi_i * h_scale;
        Q.h_f = Q.Phi_f * h_scale;
        Q.t_max = Q.theta_max * pow(Ld, Q.z);
        return Q;
    }

    if (argc == 9) {
        Q.mode = MODE_LEGACY;
        Q.L = parse_L(argv[1]);
        Q.g_i = parse_double(argv[2], "g_i");
        Q.g_f = parse_double(argv[3], "g_f");
        Q.h_i = parse_double(argv[4], "h_i");
        Q.h_f = parse_double(argv[5], "h_f");
        Q.t_max = parse_double(argv[6], "t_max");
        Q.N_t = parse_int_min(argv[7], "N_t", 2);
        Q.pbc = parse_bc(argv[8]);

        if (Q.t_max <= 0.0) {
            fprintf(stderr, "[ERROR] t_max must be > 0\n");
            exit(EXIT_FAILURE);
        }
        return Q;
    }

    print_usage(argv[0]);
    exit(EXIT_FAILURE);
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "\nUsage:\n"
            "  %s <L> <g_i> <g_f> <h_i> <h_f> <t_max> <N_t> <bc>\n"
            "  %s cqt <L> <bc> <g_pc> <beta_over_nu> <nu> <z> <y_h> <theta_max> <N_theta>\n"
            "  %s cqt <L> <bc> <beta_over_nu> <nu> <y_h> <theta_max> <N_theta>\n"
            "  %s foqt <L> <bc> <g_fixed> <kappa0> <kappa1> <theta_max> <N_theta>\n"
            "  %s foqt <L> <bc> <g_fixed> <theta_max> <N_theta>\n"
            "  %s loschmidt <L> <bc> <g_pc> <Phi> <y_h> <z> <theta_max> <N_theta>\n"
            "  %s loschmidt <L> <bc> <Phi> <y_h> <theta_max> <N_theta>\n\n"
            "BC can be 1/PBC/pbc or 0/OBC/obc.\n\n"
            "Examples:\n"
            "  %s 10 5.0 1.0 0.0 0.0 20.0 500 pbc\n"
            "  %s cqt 10 pbc 0.999893388127 0.1247342048706 1.0253174105560887 1.001492387467 1.8767581825964 10.0 201\n"
            "  %s foqt 10 pbc 0.5 1.0 -1.0 10.0 201\n\n",
            prog, prog, prog, prog, prog, prog, prog, prog, prog, prog);
}
