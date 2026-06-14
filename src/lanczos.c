/* Lanczos eigensolver with full reorthogonalization. */

#include "lanczos.h"
#include "apply_H.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <lapacke.h>

/* Dot product of two vectors. */
static double dot(const double *a, const double *b, long long n)
{
    double s = 0.0;
    for (long long i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* Euclidean vector norm. */
static double norm(const double *v, long long n)
{
    return sqrt(dot(v, v, n));
}

/* In-place v += alpha * u. */
static void axpy(double alpha, const double *u, double *v, long long n)
{
    for (long long i = 0; i < n; i++) v[i] += alpha * u[i];
}

/* In-place vector scaling. */
static void scale(double scalar, double *v, long long n)
{
    for (long long i = 0; i < n; i++) v[i] *= scalar;
}

/* Copy u into v. */
static void copy_vec(const double *u, double *v, long long n)
{
    memcpy(v, u, (size_t)n * sizeof(double));
}

/* Fill v with deterministic pseudo-random values in [-1,1]. */
static void rand_vec(double *v, long long n, unsigned long *seed)
{
    for (long long i = 0; i < n; i++) {
        *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (double)(int)(*seed >> 33) / (double)(1LL << 31);
    }
}

static const LanczosParams DEFAULTS = {
    .n_eig = LANCZOS_DEFAULT_N_EIG,
    .max_iter = LANCZOS_DEFAULT_MAX_ITER,
    .tol = LANCZOS_DEFAULT_TOL,
    .max_restarts = LANCZOS_DEFAULT_MAX_RESTARTS,
    .verbose = LANCZOS_DEFAULT_VERBOSE,
    .seed = LANCZOS_DEFAULT_SEED,
};

/*
 * Diagonalize the symmetric tridiagonal Krylov matrix.
 * alpha stores eigenvalues on exit; Z stores eigenvectors column-major.
 * Returns LAPACK info code (0 = success).
 */
static int tridiag_eig(int D, double *alpha, double *beta, double *Z)
{
    /* dstevd expects a compact sub-diagonal. */
    double *e = (double *)malloc((size_t)(D - 1) * sizeof(double));
    if (!e) return -999;
    for (int i = 0; i < D - 1; i++) e[i] = beta[i + 1];

    /* Z is D x D and column-major for LAPACK. */
    lapack_int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V',
                                     (lapack_int)D,
                                     alpha,  /* in: diag, out: evals */
                                     e,      /* sub-diagonal (length D-1) */
                                     Z,      /* out: eigvecs (col-major) */
                                     (lapack_int)D);
    free(e);
    return (int)info;
}

int lanczos_generic(long long dim,
                    lanczos_matvec_fn matvec, void *ctx,
                    const LanczosParams *p,
                    double *evals, double *evecs)
{
    /* Use defaults when the caller passes NULL. */
    const LanczosParams *par = (p != NULL) ? p : &DEFAULTS;
    const int n_eig = par->n_eig;
    const int max_iter = par->max_iter;
    const double tol = par->tol;
    const int max_rst = (par->max_restarts <= 0) ? 50 : par->max_restarts;
    const int verbose = par->verbose;

    if (n_eig < 1 || max_iter < n_eig + 2 || !evals) return -1;

    int D = max_iter;  /* Krylov dimension per restart. */

    /* V[j] = V + j*dim is the j-th Krylov vector. */
    double *V = (double *)malloc((size_t)(D + 1) * dim * sizeof(double));
    if (!V) { fprintf(stderr, "[lanczos] malloc V failed\n"); return -1; }

    double *Hv  = (double *)malloc((size_t)dim * sizeof(double));
    if (!Hv) {
        free(V); free(Hv);
        fprintf(stderr, "[lanczos] malloc workspace failed\n");
        return -1;
    }

    /* Tridiagonal diagonal and sub-diagonal. */
    double *alpha = (double *)calloc((size_t)D, sizeof(double));
    double *beta = (double *)calloc((size_t)(D+1), sizeof(double));
    /* Eigenvectors of the tridiagonal matrix. */
    double *Z     = (double *)malloc((size_t)D * D * sizeof(double));
    if (!alpha || !beta || !Z) {
        free(V); free(Hv); free(alpha); free(beta); free(Z);
        fprintf(stderr, "[lanczos] malloc tridiag buffers failed\n");
        return -1;
    }

    /* Previous Ritz values for convergence checks. */
    double *evals_prev = (double *)calloc((size_t)n_eig, sizeof(double));
    if (!evals_prev) {
        free(V); free(Hv); free(alpha); free(beta); free(Z);
        fprintf(stderr, "[lanczos] malloc evals_prev failed\n");
        return -1;
    }

    /* Initial vector seed. */
    unsigned long seed = par->seed ? par->seed :
                         (unsigned long)time(NULL);

    /* Start from a normalized random vector. */
    rand_vec(V, dim, &seed);
    double b0 = norm(V, dim);
    scale(1.0 / b0, V, dim);

    int converged = 0;
    int ret = 0;

    for (int restart = 0; restart < max_rst && !converged; restart++) {

        /* Build the Krylov basis. */
        memset(alpha, 0, (size_t)D * sizeof(double));
        memset(beta,  0, (size_t)(D + 1) * sizeof(double));

        int j_max = 0;

        for (int j = 0; j < D; j++) {
            double *Vj  = V + (long long)j * dim;
            double *Vjp = V + (long long)(j + 1) * dim;

            /* Hv = H V[j]. */
            matvec(Vj, Hv, ctx);

            /* Diagonal tridiagonal element. */
            alpha[j] = dot(Vj, Hv, dim);

            /* Remove current-vector component. */
            axpy(-alpha[j], Vj, Hv, dim);

            /* Remove previous-vector component. */
            if (j > 0) axpy(-beta[j], V + (long long)(j - 1) * dim, Hv, dim);

            /* Full reorthogonalization is needed near degeneracies. */
            for (int k = 0; k <= j; k++) {
                double proj = dot(V + (long long)k * dim, Hv, dim);
                axpy(-proj, V + (long long)k * dim, Hv, dim);
            }

            /* Sub-diagonal tridiagonal element. */
            beta[j + 1] = norm(Hv, dim);

            if (beta[j + 1] < 1e-14) {
            j_max = j + 1;
            break;   /* invariant subspace: Vjp is not needed */
            }

            /* Normalize and store the next Krylov vector. */
            copy_vec(Hv, Vjp, dim);
            scale(1.0 / beta[j + 1], Vjp, dim);
            j_max = j + 1;
        }

        if (j_max == 0) j_max = 1;

        /* Diagonalize the tridiagonal projection. */
        /* dstevd overwrites alpha, so work on a copy. */
        double *alpha_work = (double *)malloc((size_t)j_max * sizeof(double));
        if (!alpha_work) { ret = -1; break; }
        memcpy(alpha_work, alpha, (size_t)j_max * sizeof(double));

        /* Z is j_max x j_max. */
        int info = tridiag_eig(j_max, alpha_work, beta, Z);
        if (info != 0) {
            fprintf(stderr, "[lanczos] LAPACK dstevd returned info=%d "
                    "(restart %d)\n", info, restart);
            free(alpha_work);
            ret = -2;
            break;
        }

        /* Copy the requested lowest Ritz values. */
        int n_avail = (j_max < n_eig) ? j_max : n_eig;
        for (int k = 0; k < n_avail; k++) evals[k] = alpha_work[k];
        free(alpha_work);

        /* Check relative changes in the requested Ritz values. */
        converged = 1;
        for (int k = 0; k < n_avail; k++) {
            double dE = fabs(evals[k] - evals_prev[k]);
            double ref = fmax(1.0, fabs(evals[k]));
            if (dE / ref > tol) { converged = 0; break; }
        }
        for (int k = 0; k < n_avail; k++) evals_prev[k] = evals[k];

        if (verbose) {
            fprintf(stderr, "[lanczos] restart %2d  j_max=%3d  "
                    "E0=%+.10f  E1=%+.10f  conv=%d\n",
                    restart, j_max,
                    (n_avail > 0) ? evals[0] : 0.0,
                    (n_avail > 1) ? evals[1] : 0.0,
                    converged);
        }

        if (!converged) {
    int n_keep = (n_avail < n_eig) ? n_avail : n_eig;

    /* Keep the lowest Ritz vectors for restart. */
    double *Vtmp = (double *)malloc((size_t)n_keep * dim * sizeof(double));
    if (!Vtmp) { ret = -1; break; }

    /* Back-project Ritz vectors into the full space. */
    for (int k = 0; k < n_keep; k++) {
        double *vk = Vtmp + (long long)k * dim;
        memset(vk, 0, (size_t)dim * sizeof(double));
        for (int n = 0; n < j_max; n++)
            axpy(Z[k * j_max + n], V + (long long)n * dim, vk, dim);
    }

    /* Keep restart vectors mutually orthogonal. */
    for (int k = 0; k < n_keep; k++) {
        double *vk = Vtmp + (long long)k * dim;

        for (int p = 0; p < k; p++) {
            double *vp  = Vtmp + (long long)p * dim;
            double proj = dot(vp, vk, dim);
            axpy(-proj, vp, vk, dim);
        }

        double nn = norm(vk, dim);

        /* Replace collapsed restart vectors by fresh random directions. */
        if (nn < 1e-14) {
            rand_vec(vk, dim, &seed);
            for (int p = 0; p < k; p++) {
                double *vp = Vtmp + (long long)p * dim;
                axpy(-dot(vp, vk, dim), vp, vk, dim);
            }
            nn = norm(vk, dim);
        }

        scale(1.0 / nn, vk, dim);
    }

    /* Restart from the best current Ritz vectors. */
    for (int k = 0; k < n_keep; k++)
        copy_vec(Vtmp + (long long)k * dim, V + (long long)k * dim, dim);

    free(Vtmp);
}
    }

    if (!converged && ret == 0) {
        if (verbose)
            fprintf(stderr, "[lanczos] WARNING: did not converge in %d restarts\n",
                    max_rst);
        ret = -3;
    }

    /* Rebuild eigenvectors if requested. */
    if (evecs && (converged || ret == -3)) {
        /* Rebuild the Krylov basis from the converged starting vector. */

        /* No restart or convergence checks are needed here. */

        int D2 = (max_iter < n_eig + 4) ? n_eig + 4 : max_iter;
        D2 = (D2 < D) ? D2 : D;

        memset(alpha, 0, (size_t)D2 * sizeof(double));
        memset(beta,  0, (size_t)(D2 + 1) * sizeof(double));

        int j_max2 = 0;
        for (int j = 0; j < D2; j++) {
            double *Vj  = V + (long long)j * dim;
            double *Vjp = V + (long long)(j + 1) * dim;

            matvec(Vj, Hv, ctx);
            alpha[j] = dot(Vj, Hv, dim);
            axpy(-alpha[j], Vj, Hv, dim);
            if (j > 0) axpy(-beta[j], V + (long long)(j - 1) * dim, Hv, dim);

            for (int k = 0; k <= j; k++) {
                double proj = dot(V + (long long)k * dim, Hv, dim);
                axpy(-proj, V + (long long)k * dim, Hv, dim);
            }

            beta[j + 1] = norm(Hv, dim);
            if (beta[j + 1] < 1e-14) { j_max2 = j + 1; break; }
            copy_vec(Hv, Vjp, dim);
            scale(1.0 / beta[j + 1], Vjp, dim);
            j_max2 = j + 1;
        }
        if (j_max2 == 0) j_max2 = 1;

        /* Diagonalize the final tridiagonal matrix. */
        double *aw = (double *)malloc((size_t)j_max2 * sizeof(double));
        double *Zf = (double *)malloc((size_t)j_max2 * j_max2 * sizeof(double));
        if (aw && Zf) {
            memcpy(aw, alpha, (size_t)j_max2 * sizeof(double));
            int info2 = tridiag_eig(j_max2, aw, beta, Zf);
            if (info2 == 0) {
                int n_avail2 = (j_max2 < n_eig) ? j_max2 : n_eig;
                for (int k = 0; k < n_avail2; k++) {
                    evals[k] = aw[k];
                    /* Back-project Ritz vector k. */
                    double *ek = evecs + (long long)k * dim;
                    memset(ek, 0, (size_t)dim * sizeof(double));
                    for (int n = 0; n < j_max2; n++)
                        axpy(Zf[k * j_max2 + n], V + (long long)n * dim, ek, dim);
                    /* Normalize to guard against roundoff. */
                    double nn2 = norm(ek, dim);
                    if (nn2 > 1e-14) scale(1.0 / nn2, ek, dim);
                }
            }
        }
        free(aw); free(Zf);
    }

    free(V); free(Hv);
    free(alpha); free(beta); free(Z);
    free(evals_prev);

    return ret;
}

typedef struct {
    const Basis *b;
    double g, h;
} FullSpaceCtx;

static void matvec_full_space(const double *v_in, double *v_out, void *ctx)
{
    FullSpaceCtx *c = (FullSpaceCtx *)ctx;
    apply_H(c->b, c->g, c->h, v_in, v_out);
}

int lanczos(const Basis *b, double g, double h,
            const LanczosParams *p,
            double *evals, double *evecs)
{
    FullSpaceCtx ctx = { .b = b, .g = g, .h = h };
    return lanczos_generic(b->dim, matvec_full_space, &ctx, p, evals, evecs);
}
