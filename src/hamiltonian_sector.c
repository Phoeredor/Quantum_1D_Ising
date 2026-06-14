/*
 * Hamiltonian operations in a Z2 parity sector at h = 0.
 * Representatives are the lower half of the full bit basis.
 */

#include "../include/hamiltonian_sector.h"
#include <string.h>   /* memset */
#include <stdio.h>

/* Matrix-free sector matvec. */
void apply_H_sector(const BasisSector *bs, double g,
                    const double *v_in, double *v_out)
{
    const int L        = bs->L;
    const int pbc      = bs->pbc;
    const int parity   = bs->parity;
    const long long half = bs->dim;           /* 2^(L-1) */
    const long long mask = (1LL << L) - 1;    /* 2^L - 1 */
    const int jmax     = pbc ? L : (L - 1);   /* number of ZZ bonds */

    memset(v_out, 0, (size_t)half * sizeof(double));

    for (long long r = 0; r < half; r++) {

        /* ZZ diagonal term. */
        double diag = 0.0;
        for (int j = 0; j < jmax; j++) {
            int jnext = (j + 1) % L;
            diag -= (double)(sz_val(r, j) * sz_val(r, jnext));
        }
        v_out[r] += diag * v_in[r];

        /* Transverse field; parity fixes the representative sign. */
        for (int j = 0; j < L; j++) {
            long long s = r ^ (1LL << j);     /* flip bit j */
            long long r2;
            double sign;

            if (s < half) {
                r2   = s;
                sign = 1.0;
            } else {
                r2   = s ^ mask;               /* canonical representative */
                sign = (parity == 0) ? 1.0 : -1.0;
            }

            v_out[r2] += -g * sign * v_in[r];
        }
    }
}

/* Dense sector matrix in column-major storage. */
void build_ham_sector(const BasisSector *bs, double g, double *Ham)
{
    const int L        = bs->L;
    const int pbc      = bs->pbc;
    const int parity   = bs->parity;
    const long long dim  = bs->dim;           /* 2^(L-1) */
    const long long mask = (1LL << L) - 1;
    const int jmax     = pbc ? L : (L - 1);

    /* Ham must be zeroed by the caller. */

    for (long long r = 0; r < dim; r++) {

        /* ZZ diagonal term. */
        double diag = 0.0;
        for (int j = 0; j < jmax; j++) {
            int jnext = (j + 1) % L;
            diag -= (double)(sz_val(r, j) * sz_val(r, jnext));
        }
        Ham[r * dim + r] += diag;

        /* Transverse field. */
        for (int j = 0; j < L; j++) {
            long long s = r ^ (1LL << j);
            long long r2;
            double sign;

            if (s < dim) {
                r2   = s;
                sign = 1.0;
            } else {
                r2   = s ^ mask;
                sign = (parity == 0) ? 1.0 : -1.0;
            }

            /* H_{r2, r}: column-major => Ham[col=r, row=r2] */
            Ham[r * dim + r2] += -g * sign;
        }
    }
}

typedef struct {
    const BasisSector *bs;
    double g;
} SectorCtx;

static void matvec_sector(const double *v_in, double *v_out, void *ctx)
{
    SectorCtx *c = (SectorCtx *)ctx;
    apply_H_sector(c->bs, c->g, v_in, v_out);
}

int lanczos_sector(const BasisSector *bs, double g,
                   const LanczosParams *p,
                   double *evals, double *evecs)
{
    SectorCtx ctx = { .bs = bs, .g = g };
    return lanczos_generic(bs->dim, matvec_sector, &ctx, p, evals, evecs);
}
