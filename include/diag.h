/*
 * Full diagonalization of a real symmetric matrix with LAPACK dsyevd.
 * Ham is column-major on entry and stores eigenvectors on exit.
 * Returns 0 on success, nonzero on LAPACK error.
 */
#ifndef DIAG_H
#define DIAG_H

#include "basis.h"

int diagonalize(double *Ham, long long dim, double *eigenvalues);

#endif /* DIAG_H */
