#ifndef LANCZOS_H
#define LANCZOS_H

/*
 * Lanczos eigensolver for real symmetric Hamiltonians.
 * Uses a matrix-free matvec and full reorthogonalization.
 * Memory scales as O(dim * max_iter).
 */

#include "basis.h"

typedef struct {
    int    n_eig;         /* Number of lowest eigenpairs requested. */
    int    max_iter;      /* Maximum Krylov dimension per restart. */
    double tol;           /* Relative convergence tolerance. */
    int    max_restarts;  /* Maximum restarts; 0 means unlimited. */
    int    verbose;       /* Print iteration diagnostics when nonzero. */
    unsigned long seed;   /* Seed for the initial vector; 0 uses time. */
} LanczosParams;

#define LANCZOS_DEFAULT_N_EIG       4
#define LANCZOS_DEFAULT_MAX_ITER    80
#define LANCZOS_DEFAULT_TOL         1e-10
#define LANCZOS_DEFAULT_MAX_RESTARTS 20
#define LANCZOS_DEFAULT_VERBOSE     0
#define LANCZOS_DEFAULT_SEED        12345UL

/*
 * Matrix-vector callback used by the generic solver.
 * Implementations compute v_out = H v_in.
 */
typedef void (*lanczos_matvec_fn)(const double *v_in, double *v_out, void *ctx);

/*
 * Compute the n_eig lowest eigenpairs using a caller-provided matvec.
 * evals is ascending; evecs is row-major with shape n_eig x dim.
 * Return codes: 0 success, -1 allocation, -2 LAPACK, -3 no convergence.
 */
int lanczos_generic(long long dim,
                    lanczos_matvec_fn matvec, void *ctx,
                    const LanczosParams *p,
                    double *evals, double *evecs);

/*
 * Full-basis convenience wrapper using apply_H().
 * evals and evecs must be allocated by the caller.
 */
int lanczos(const Basis *b, double g, double h,
            const LanczosParams *p,
            double *evals, double *evecs);

#endif /* LANCZOS_H */
