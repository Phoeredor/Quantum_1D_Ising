#ifndef BASIS_SECTOR_H
#define BASIS_SECTOR_H

#include "basis.h"

/*
 * Z2 parity-sector basis for h = 0.
 * The parity operator flips every spin in the z-basis.
 */
typedef struct {
    int        L;
    int        pbc;
    int        parity;    /* 0 = even (P=+1), 1 = odd (P=-1) */
    long long  dim;       /* 2^(L-1) */
} BasisSector;

/*
 * Initialize one parity sector.
 * Representatives use the lower half of the full bit basis.
 */
BasisSector basis_sector_init(int L, int pbc, int parity);

#endif /* BASIS_SECTOR_H */
