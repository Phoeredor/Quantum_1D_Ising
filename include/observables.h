#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include "basis.h"
#include <math.h>

/* Ground-state observables in the sigma^z computational basis. */

/* Transverse magnetization per site. */
double obs_mx(const Basis *b, const double *psi0);

/* Squared longitudinal magnetization density, <(Mz/L)^2>. */
double obs_mz_sq(const Basis *b, const double *psi0);

/*
 * Average absolute longitudinal magnetization.
 * This differs from sqrt(obs_mz_sq()).
 */
double obs_psi_bar(const Basis *b, const double *psi0);

/* Signed longitudinal magnetization per site. */
double obs_mz_raw(const Basis *b, const double *psi0);

/* Fourth moment of the longitudinal magnetization density. */
double obs_mz4(const Basis *b, const double *psi0);

/*
 * Binder ratio U = mz4 / mz2^2.
 * Uses precomputed moments to avoid repeated basis loops.
 */
double obs_binder(double mz2, double mz4);

/* Equal-time spin correlator Czz(r) = <sigma^z_0 sigma^z_r>. */
double obs_czz(const Basis *b, const double *psi0, int r);

/*
 * Histogram of the number of up spins in the ground state.
 * hist must have length L+1 and be zeroed by the caller.
 */
void obs_pm_histogram(const Basis *b, const double *psi0, double *hist);

/*
 * Parity-sector order parameter |<even|Mz/L|odd>|.
 * Valid only at h = 0, where parity sectors are well defined.
 */
double obs_psi_tilde(const Basis *b,
                     const double *gs_even,
                     const double *gs_odd,
                     double h);

/*
 * Same parity-sector order parameter, using reduced sector vectors.
 * gs_even and gs_odd have length 2^(L-1).
 */
double obs_psi_tilde_sector(int L,
                            const double *gs_even,
                            const double *gs_odd);

/*
 * Longitudinal susceptibility from finite differences of E0.
 * Disabled here because production chi_z uses signed-magnetization stencils.
 */
// double obs_chi_z_diff(const Basis *b, double g, double h, double dh);

/*
 * Zero-temperature specific-heat proxy from finite differences of E0.
 * Kept disabled to avoid mixing derivative-based observables into runs.
 */
// double obs_c_diff(const Basis *b, double g, double h, double dg);

/*
 * Sum-over-states susceptibility from the complete spectrum.
 * eigvecs is column-major and eigvals must be sorted ascending.
 */
double obs_chi_z(const Basis *b,
                 const double *eigvecs,
                 const double *eigvals);

#endif /* OBSERVABLES_H */
