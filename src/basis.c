/* Full computational-basis descriptor for the 1D Ising chain. */

#include "../include/basis.h"
#include <stdio.h>
#include <stdlib.h>

Basis basis_init(int L, int pbc)
{
    if (L <= 0 || L > 28) {
        fprintf(stderr, "[basis] L=%d out of range [1,28]\n", L);
        exit(EXIT_FAILURE);
    }

    Basis b;
    b.L = L;
    b.dim = (long long)1 << L;   /* dim = 2^L. */
    b.pbc = pbc;
    return b;
}
