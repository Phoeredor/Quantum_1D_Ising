#ifndef HAMILTONIAN_SECTOR_H
#define HAMILTONIAN_SECTOR_H

#include "basis_sector.h"
#include "lanczos.h"

/*
 * Hamiltonian operations in a Z2 parity sector.
 * The sector construction assumes h = 0.
 */
void apply_H_sector(const BasisSector *bs, double g,
                    const double *v_in, double *v_out);

/*
 * Build the dense sector Hamiltonian in column-major order.
 * Ham must be allocated by the caller.
 */
void build_ham_sector(const BasisSector *bs, double g, double *Ham);

/*
 * Compute low-energy eigenpairs in the sector basis.
 * Return codes match lanczos_generic().
 */
int lanczos_sector(const BasisSector *bs, double g,
                   const LanczosParams *p,
                   double *evals, double *evecs);

#endif /* HAMILTONIAN_SECTOR_H */
