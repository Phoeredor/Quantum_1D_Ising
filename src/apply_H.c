/*
 * Matrix-free Hamiltonian-vector products for the full basis.
 * H = -sum_j sz_j sz_{j+1} - h sum_j sz_j - g sum_j sx_j.
 */

#include "apply_H.h"
#include <stdio.h>   /* fprintf, stderr */
#include <string.h>  /* memset */

/* Uniform transverse field. */
void apply_H(const Basis *b, double g, double h,
             const double *v_in, double *v_out)
{
    const int L = b->L;
    const long long dim = b->dim;
    const int pbc = b->pbc;

    memset(v_out, 0, (size_t)dim * sizeof(double));

    for (long long ii = 0; ii < dim; ii++) {
        double diag = 0.0;

        for (int j = 0; j < L; j++) {
            int szj = sz_val(ii, j);

            /* ZZ coupling; PBC wraps the last bond back to site 0. */
            if (pbc || j < L - 1) {
                int jnext = (j + 1) % L;
                diag -= (double)(szj * sz_val(ii, jnext));
            }

            /* -h sz_j */
            diag -= h * (double)szj;
        }

        v_out[ii] += diag * v_in[ii];

        /* Transverse field flips one spin. */
        for (int j = 0; j < L; j++) {
            long long jj = flip(ii, j);
            v_out[jj] -= g * v_in[ii];
        }
    }
}

/* Site-dependent transverse field g_j = g * (1 + eps[j]). */
void apply_H_disorder(const Basis *b, double g, double h,
                      const double *eps,
                      const double *v_in, double *v_out)
{
    if (!eps) {
        fprintf(stderr, "[apply_H_disorder] eps array is NULL; aborting apply.\n");
        return;
    }

    const int L = b->L;
    const long long dim = b->dim;
    const int pbc = b->pbc;

    memset(v_out, 0, (size_t)dim * sizeof(double));

    for (long long ii = 0; ii < dim; ii++) {
        double diag = 0.0;

        for (int j = 0; j < L; j++) {
            int szj = sz_val(ii, j);

            /* ZZ coupling. */
            if (pbc || j < L - 1) {
                int jnext = (j + 1) % L;
                diag -= (double)(szj * sz_val(ii, jnext));
            }

            /* -h sz_j */
            diag -= h * (double)szj;
        }

        v_out[ii] += diag * v_in[ii];

        /* Transverse field with local disorder. */
        for (int j = 0; j < L; j++) {
            long long jj = flip(ii, j);
            double    gj = g * (1.0 + eps[j]);
            v_out[jj] -= gj * v_in[ii];
        }
    }
}
