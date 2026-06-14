/*
 * Dense full-basis Hamiltonian builder.
 * H = -sum_j sz_j sz_{j+1} - h sum_j sz_j - g sum_j sx_j
 * Ham is column-major and must be allocated by the caller.
 */
#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "basis.h"

void build_ham(const Basis *b, double g, double h, double *Ham);

#endif /* HAMILTONIAN_H */
