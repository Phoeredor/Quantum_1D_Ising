/* Dense real-symmetric diagonalization via LAPACK dsyevd. */

#include <lapacke.h>
#include <stdio.h>
#include "../include/diag.h"

int diagonalize(double *Ham, long long dim, double *eigenvalues)
{
    lapack_int info = LAPACKE_dsyevd(
        LAPACK_COL_MAJOR,  /* Column-major storage. */
        'V',               /* Compute eigenvalues and eigenvectors. */
        'U',               /* use upper triangle */
        (lapack_int)dim,
        Ham,
        (lapack_int)dim,
        eigenvalues
    );

    if (info != 0)
        fprintf(stderr, "[diag] LAPACK dsyevd failed: info = %d\n"
                        "       info > 0 means convergence failure\n"
                        "       info < 0 means illegal argument\n",
                (int)info);

    return (int)info;
}
