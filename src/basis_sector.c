/* Z2 parity-sector basis descriptors. */

#include "../include/basis_sector.h"
#include <stdio.h>
#include <stdlib.h>

BasisSector basis_sector_init(int L, int pbc, int parity)
{
    if (L < 2 || L > 28) {
        fprintf(stderr, "[basis_sector_init] L=%d out of range [2,28]\n", L);
        exit(EXIT_FAILURE);
    }
    if (parity != 0 && parity != 1) {
        fprintf(stderr, "[basis_sector_init] parity=%d must be 0 or 1\n", parity);
        exit(EXIT_FAILURE);
    }

    BasisSector bs;
    bs.L      = L;
    bs.pbc    = pbc;
    bs.parity = parity;
    bs.dim    = 1LL << (L - 1);   /* 2^(L-1) */

    return bs;
}
