/* Ground-state observables in the sigma^z computational basis. */

#include "../include/observables.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Transverse magnetization per site. */
double obs_mx(const Basis *b, const double *psi0)
{
    const long long dim = b->dim;
    const int L = b->L;
    double mx = 0.0;

    for (int j = 0; j < L; j++)
        for (long long ii = 0; ii < dim; ii++)
            mx += psi0[ii] * psi0[flip(ii, j)];

    return mx / L;
}

/* Squared longitudinal magnetization density, <(Mz/L)^2>. */
double obs_mz_sq(const Basis *b, const double *psi0)
{
    const long long dim = b->dim;
    const int L = b->L;
    double mz2 = 0.0;

    for (long long ii = 0; ii < dim; ii++) {
        double sz = (double)sz_total(ii, L);
        mz2 += psi0[ii] * psi0[ii] * sz * sz;
    }

    return mz2 / ((double)L * L);
}

/* Average absolute longitudinal magnetization. */
double obs_psi_bar(const Basis *b, const double *psi0)
{
    const long long dim = b->dim;
    const int L = b->L;
    double psi_bar = 0.0;

    for (long long ii = 0; ii < dim; ii++)
        psi_bar += psi0[ii] * psi0[ii] * fabs((double)sz_total(ii, L));

    return psi_bar / L;
}

/* Signed longitudinal magnetization per site. */
double obs_mz_raw(const Basis *b, const double *psi0)
{
    const long long dim = b->dim;
    const int L = b->L;
    double mz = 0.0;

    for (long long ii = 0; ii < dim; ii++)
        mz += psi0[ii] * psi0[ii] * (double)sz_total(ii, L);

    return mz / L;
}

/* Fourth moment of the longitudinal magnetization density. */
double obs_mz4(const Basis *b, const double *psi0)
{
    const long long dim = b->dim;
    const int L = b->L;
    double mz4 = 0.0;

    for (long long ii = 0; ii < dim; ii++) {
        double sz = (double)sz_total(ii, L) / L;
        double sz2 = sz * sz;
        mz4 += psi0[ii] * psi0[ii] * sz2 * sz2;
    }

    return mz4;
}

/* Binder ratio U = mz4 / mz2^2. */
double obs_binder(double mz2, double mz4)
{
    if (mz2 < 1e-30) return 0.0;
    return mz4 / (mz2 * mz2);
}

/* Equal-time spin correlator Czz(r) = <sigma^z_0 sigma^z_r>. */
double obs_czz(const Basis *b, const double *psi0, int r)
{
    const long long dim = b->dim;
    double czz = 0.0;

    for (long long ii = 0; ii < dim; ii++) {
        double w = psi0[ii] * psi0[ii];
        czz += w * (double)(sz_val(ii, 0) * sz_val(ii, r));
    }

    return czz;
}

/* Histogram of the number of up spins; hist has length L+1. */
void obs_pm_histogram(const Basis *b, const double *psi0, double *hist)
{
    const long long dim = b->dim;

    for (long long ii = 0; ii < dim; ii++) {
        int k = __builtin_popcountll((unsigned long long)ii);
        hist[k] += psi0[ii] * psi0[ii];
    }
}

/* Parity-sector order parameter |<even|Mz/L|odd>|, valid only at h = 0. */
double obs_psi_tilde(const Basis *b,
                     const double *gs_even,
                     const double *gs_odd,
                     double h)
{
    if (h != 0.0) {
        fprintf(stderr,
            "[obs_psi_tilde] called with h=%.6g != 0: "
            "Z2 symmetry is broken, parity sectors are ill-defined. "
            "Aborting.\n", h);
        abort();
    }

    const long long dim = b->dim;
    const int L = b->L;
    double mel = 0.0;

    for (long long ii = 0; ii < dim; ii++)
        mel += gs_even[ii] * (double)sz_total(ii, L) * gs_odd[ii];

    return fabs(mel) / L;
}

/* Same order parameter using reduced parity-sector vectors. */
double obs_psi_tilde_sector(int L,
                            const double *gs_even,
                            const double *gs_odd)
{
    const long long half = 1LL << (L - 1);
    double mel = 0.0;

    for (long long r = 0; r < half; r++)
        mel += gs_even[r] * (double)sz_total(r, L) * gs_odd[r];

    return fabs(mel) / L;
}

/*
 * Disabled derivative-based chi_z prototype.
 * Production chi_z uses signed-magnetization finite differences.
double obs_chi_z_diff(const Basis *b, double g, double h, double dh)
{
    double em2 = lanczos_ground_energy(b, g, h - 2.0*dh);
    double em1 = lanczos_ground_energy(b, g, h - dh);
    double e0 = lanczos_ground_energy(b, g, h);
    double ep1 = lanczos_ground_energy(b, g, h + dh);
    double ep2 = lanczos_ground_energy(b, g, h + 2.0*dh);

    double d2 = (-ep2 + 16.0*ep1 - 30.0*e0
                 + 16.0*em1 - em2) / (12.0 * dh * dh);

    return -d2 / b->L;
}
*/

/*
 * Disabled derivative-based specific-heat proxy.
 * Kept out of production runs to avoid mixing derivative conventions.
double obs_c_diff(const Basis *b, double g, double h, double dg)
{
    double em2 = lanczos_ground_energy(b, g - 2.0*dg, h);
    double em1 = lanczos_ground_energy(b, g - dg, h);
    double e0 = lanczos_ground_energy(b, g, h);
    double ep1 = lanczos_ground_energy(b, g + dg, h);
    double ep2 = lanczos_ground_energy(b, g + 2.0*dg, h);

    double d2 = (-ep2 + 16.0*ep1 - 30.0*e0
                 + 16.0*em1 - em2) / (12.0 * dg * dg);

    return -g * d2 / b->L;
}
*/


/*
 * Sum-over-states susceptibility from the complete dense spectrum.
 * eigvecs is column-major and eigvals must be sorted ascending.
 */
double obs_chi_z(const Basis *b, const double *eigvecs, const double *eigvals)
{
    const long long dim = b->dim;
    const int L = b->L;

    double *v = malloc((size_t)dim * sizeof(double));
    if (!v) {
        fprintf(stderr, "[obs_chi_z] malloc failed\n");
        return 0.0;
    }

    const double *psi0 = eigvecs;
    for (long long ii = 0; ii < dim; ii++)
        v[ii] = (double)sz_total(ii, L) * psi0[ii];

    double chi = 0.0;
    const double E0 = eigvals[0];

    for (long long n = 1; n < dim; n++) {
        double dE = eigvals[n] - E0;
        if (dE < 1e-10) continue;

        const double *psin = eigvecs + (long long)n * dim;
        double mel = 0.0;
        for (long long ii = 0; ii < dim; ii++)
            mel += psin[ii] * v[ii];
        chi += mel * mel / dE;
    }

    free(v);
    return 2.0 * chi / L;
}
