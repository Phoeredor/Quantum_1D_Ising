/**
 * @file pcg32.c
 * @brief Minimal implementation of the PCG32 random number generator.
 *
 * Algorithm and implementation based on M.E. O'Neill / pcg-random.org
 * and licensed under Apache License 2.0.
 */

#include "pcg32.h"

#include <stdint.h>

uint32_t pcg32_random_r(pcg32_random_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1U);

    uint32_t xorshifted = (uint32_t)(((oldstate >> 18U) ^ oldstate) >> 27U);
    uint32_t rot = (uint32_t)(oldstate >> 59U);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31U));
}

void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1U) | 1U;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}
