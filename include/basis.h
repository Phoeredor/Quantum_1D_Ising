#ifndef BASIS_H
#define BASIS_H

#include <stdint.h>

/*
 * Full computational basis for the 1D quantum Ising chain.
 * Bit j stores sigma^z_j: 1 -> +1, 0 -> -1. Site 0 is the LSB.
 */

typedef struct {
    int       L;    /* Chain length. */
    long long dim;  /* Hilbert-space dimension, dim = 2^L. */
    int       pbc;  /* 1 = periodic boundary conditions, 0 = open. */
} Basis;

/* Initialize a full-basis descriptor for the chosen size and boundary. */
Basis basis_init(int L, int pbc);

/* Return the sigma^z eigenvalue at site j. */
static inline int sz_val(long long state, int j) {
    return 2 * ((int)((state >> j) & 1)) - 1;
}

/* Return the raw bit value at site j. */
static inline int spin(long long state, int j) {
    return (int)((state >> j) & 1);
}

/* Apply sigma^x_j by flipping bit j. */
static inline long long flip(long long state, int j) {
    return state ^ ((long long)1 << j);
}

/* Return sum_j sigma^z_j for a computational-basis state. */
static inline int sz_total(long long state, int L) {
    int s = 0;
    for (int j = 0; j < L; j++) s += sz_val(state, j);
    return s;
}

#endif /* BASIS_H */
