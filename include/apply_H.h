#ifndef APPLY_H_H
#define APPLY_H_H

/*
 * Matrix-free Hamiltonian-vector products.
 * H = -sum_j sz_j sz_{j+1} - h sum_j sz_j - g sum_j sx_j
 * Complexity: O(L * dim) time, O(1) extra space.
 */

#include "basis.h"

/* Compute v_out = H v_in. v_out is cleared internally. */
void apply_H(const Basis *b, double g, double h,
             const double *v_in, double *v_out);

/* Same matvec with site-dependent transverse field g_j = g*(1 + eps[j]). */
void apply_H_disorder(const Basis *b, double g, double h,
                      const double *eps,
                      const double *v_in, double *v_out);

#endif /* APPLY_H_H */
