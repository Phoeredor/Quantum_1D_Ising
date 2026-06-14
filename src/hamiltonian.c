/*
 * Dense full-basis Hamiltonian builder.
 * Storage is column-major: Ham[col * dim + row] = H_{row,col}.
 */

#include "../include/hamiltonian.h"

void build_ham(const Basis *b, double g, double h, double *Ham)
{
    const long long dim = b->dim;
    const int L = b->L;

    for (long long ii = 0; ii < dim; ii++) {

        double diag = 0.0;

        for (int j = 0; j < L - 1; j++) {
            /* ZZ: -sigma^z_j sigma^z_{j+1} */
            diag -= (double)(sz_val(ii, j) * sz_val(ii, j + 1));

            /* Transverse field: flip spin j. */
            Ham[flip(ii, j) * dim + ii] -= g;
        }

        /* Last site: sigma^x_{L-1} */
        Ham[flip(ii, L - 1) * dim + ii] -= g;

        /* Longitudinal field: -h * sum_j sigma^z_j */
        for (int j = 0; j < L; j++)
            diag -= h * (double)sz_val(ii, j);

        /* PBC bond: sigma^z_{L-1} sigma^z_0 */
        if (b->pbc)
            diag -= (double)(sz_val(ii, L - 1) * sz_val(ii, 0));

        Ham[ii * dim + ii] += diag;
    }
}
