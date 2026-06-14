/**
 * @file pcg32.h
 * @brief Header for the PCG32 random number generator.
 *
 * Provides the state structure and function prototypes for the
 * Permuted Congruential Generator (PCG-XSH-RR).
 * Reference: https://www.pcg-random.org/
 */

#ifndef PCG32_H
#define PCG32_H

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t *rng);
void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq);

#endif /* PCG32_H */
