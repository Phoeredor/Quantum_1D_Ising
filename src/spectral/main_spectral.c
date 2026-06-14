#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <lapacke.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pcg32.h"

typedef struct {
    int L;
    double g;
    double h;
    double omega;
    long realization;
    unsigned long seed;
    const char *out_path;
    double bulk_frac;
    int local_window;
    int overwrite;
} Config;

typedef struct {
    uint64_t initstate;
    uint64_t initseq;
    double epsilon_min;
    double epsilon_max;
    double epsilon_mean;
} RngMetadata;

enum {
    SPECTRAL_MIN_L = 4,
    SPECTRAL_MAX_L = 12,
};

static char *g_lock_path = NULL;

static void cleanup_lock(void) {
    if (g_lock_path != NULL) {
        unlink(g_lock_path);
    }
}

static void handle_signal(int signo) {
    cleanup_lock();
    _exit(128 + signo);
}

static void print_usage(const char *progname) {
    printf("Usage:\n");
    printf("  %s --L L --g G --h H --omega W --realization R --seed MASTER_SEED --out OUTFILE [options]\n",
           progname);
    printf("\n");
    printf("Options:\n");
    printf("  --bulk-frac F       Fraction of central spectrum to keep (default: 0.5)\n");
    printf("  --local-window WN   Half-window for local unfolding (default: 25)\n");
    printf("  --master-seed SEED  Alias for --seed\n");
    printf("  --overwrite         Replace OUTFILE if it already exists\n");
    printf("  -h, --help          Show this help and exit\n");
    printf("\n");
    printf("ED-only, PBC-only, target L <= %d.\n", SPECTRAL_MAX_L);
}

static int parse_long(const char *text, long min_value, long max_value, long *out) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < min_value || value > max_value) {
        return 0;
    }
    *out = value;
    return 1;
}

static int parse_ulong(const char *text, unsigned long *out) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = value;
    return 1;
}

static int parse_double(const char *text, double *out) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        return 0;
    }
    *out = value;
    return 1;
}

static int require_value(int argc, char **argv, int i) {
    if (i + 1 >= argc) {
        fprintf(stderr, "[ising_spectral] missing value for %s\n", argv[i]);
        return 0;
    }
    return 1;
}

static int parse_args(int argc, char **argv, Config *cfg) {
    int seen_L = 0;
    int seen_g = 0;
    int seen_h = 0;
    int seen_omega = 0;
    int seen_realization = 0;
    int seen_seed = 0;
    int seen_out = 0;

    cfg->L = 0;
    cfg->g = 0.0;
    cfg->h = 0.0;
    cfg->omega = 0.0;
    cfg->realization = -1;
    cfg->seed = 0UL;
    cfg->out_path = NULL;
    cfg->bulk_frac = 0.5;
    cfg->local_window = 25;
    cfg->overwrite = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--L") == 0) {
            long value = 0;
            if (!require_value(argc, argv, i) ||
                !parse_long(argv[++i], SPECTRAL_MIN_L, SPECTRAL_MAX_L, &value)) {
                fprintf(stderr,
                        "[ising_spectral] invalid --L: expected integer %d <= L <= %d\n",
                        SPECTRAL_MIN_L, SPECTRAL_MAX_L);
                return 0;
            }
            cfg->L = (int)value;
            seen_L = 1;
        } else if (strcmp(argv[i], "--g") == 0) {
            if (!require_value(argc, argv, i) || !parse_double(argv[++i], &cfg->g)) {
                fprintf(stderr, "[ising_spectral] invalid --g\n");
                return 0;
            }
            seen_g = 1;
        } else if (strcmp(argv[i], "--h") == 0) {
            if (!require_value(argc, argv, i) || !parse_double(argv[++i], &cfg->h)) {
                fprintf(stderr, "[ising_spectral] invalid --h\n");
                return 0;
            }
            seen_h = 1;
        } else if (strcmp(argv[i], "--omega") == 0) {
            if (!require_value(argc, argv, i) || !parse_double(argv[++i], &cfg->omega) ||
                cfg->omega < 0.0) {
                fprintf(stderr, "[ising_spectral] invalid --omega: expected omega >= 0\n");
                return 0;
            }
            seen_omega = 1;
        } else if (strcmp(argv[i], "--realization") == 0) {
            if (!require_value(argc, argv, i) ||
                !parse_long(argv[++i], 0, 2147483647L, &cfg->realization)) {
                fprintf(stderr, "[ising_spectral] invalid --realization: expected integer >= 0\n");
                return 0;
            }
            seen_realization = 1;
        } else if (strcmp(argv[i], "--seed") == 0 ||
                   strcmp(argv[i], "--master-seed") == 0) {
            if (!require_value(argc, argv, i) || !parse_ulong(argv[++i], &cfg->seed)) {
                fprintf(stderr, "[ising_spectral] invalid master seed\n");
                return 0;
            }
            seen_seed = 1;
        } else if (strcmp(argv[i], "--out") == 0) {
            if (!require_value(argc, argv, i)) {
                return 0;
            }
            cfg->out_path = argv[++i];
            seen_out = 1;
        } else if (strcmp(argv[i], "--bulk-frac") == 0) {
            if (!require_value(argc, argv, i) || !parse_double(argv[++i], &cfg->bulk_frac) ||
                cfg->bulk_frac <= 0.0 || cfg->bulk_frac > 1.0) {
                fprintf(stderr, "[ising_spectral] invalid --bulk-frac: expected 0 < F <= 1\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--local-window") == 0) {
            long value = 0;
            if (!require_value(argc, argv, i) || !parse_long(argv[++i], 0, 1000000L, &value)) {
                fprintf(stderr, "[ising_spectral] invalid --local-window: expected integer >= 0\n");
                return 0;
            }
            cfg->local_window = (int)value;
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            cfg->overwrite = 1;
        } else {
            fprintf(stderr, "[ising_spectral] unknown argument: %s\n", argv[i]);
            return 0;
        }
    }

    if (!seen_L || !seen_g || !seen_h || !seen_omega || !seen_realization ||
        !seen_seed || !seen_out) {
        fprintf(stderr, "[ising_spectral] missing required arguments; use --help\n");
        return 0;
    }
    return 1;
}

static char *path_with_suffix(const char *path, const char *suffix) {
    size_t n_path = strlen(path);
    size_t n_suffix = strlen(suffix);
    char *result = (char *)malloc(n_path + n_suffix + 1U);
    if (result == NULL) {
        return NULL;
    }
    memcpy(result, path, n_path);
    memcpy(result + n_path, suffix, n_suffix + 1U);
    return result;
}

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static int acquire_lock(const char *out_path) {
    g_lock_path = path_with_suffix(out_path, ".lock");
    if (g_lock_path == NULL) {
        fprintf(stderr, "[ising_spectral] failed to allocate lock path\n");
        return 0;
    }

    int fd = open(g_lock_path, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        fprintf(stderr, "[ising_spectral] failed to create lock %s: %s\n",
                g_lock_path, strerror(errno));
        return 0;
    }
    if (close(fd) != 0) {
        fprintf(stderr, "[ising_spectral] failed to close lock %s: %s\n",
                g_lock_path, strerror(errno));
        cleanup_lock();
        return 0;
    }
    return 1;
}

static unsigned long long splitmix64_next(unsigned long long *state) {
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t canonical_double_bits(double value) {
    if (value == 0.0) {
        value = 0.0;
    }
    uint64_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t mix_seed(uint64_t state, uint64_t value) {
    unsigned long long x = (unsigned long long)(state ^ value);
    return (uint64_t)splitmix64_next(&x);
}

static RngMetadata derive_rng_metadata(const Config *cfg) {
    uint64_t master_seed = (uint64_t)cfg->seed;
    uint64_t h_bits = canonical_double_bits(cfg->h);
    uint64_t omega_bits = canonical_double_bits(cfg->omega);

    uint64_t state_acc = mix_seed(master_seed, 0x8CB92BA72F3D8DD7ULL);
    state_acc = mix_seed(state_acc, (uint64_t)(uint32_t)cfg->L);
    state_acc = mix_seed(state_acc, h_bits);
    state_acc = mix_seed(state_acc, omega_bits);
    state_acc = mix_seed(state_acc, (uint64_t)cfg->realization);

    uint64_t seq_acc = mix_seed(master_seed, 0xDB4F0B9175AE2165ULL);
    seq_acc = mix_seed(seq_acc, (uint64_t)(uint32_t)cfg->L << 32);
    seq_acc = mix_seed(seq_acc, omega_bits ^ 0x9E3779B97F4A7C15ULL);
    seq_acc = mix_seed(seq_acc, h_bits ^ 0xD1B54A32D192ED03ULL);
    seq_acc = mix_seed(seq_acc, ((uint64_t)cfg->realization << 1) | 1ULL);

    RngMetadata meta;
    meta.initstate = state_acc;
    meta.initseq = seq_acc;
    meta.epsilon_min = NAN;
    meta.epsilon_max = NAN;
    meta.epsilon_mean = NAN;
    return meta;
}

static double pcg32_uniform_01(pcg32_random_t *rng) {
    return (double)pcg32_random_r(rng) / 4294967296.0;
}

static void generate_epsilons(const Config *cfg, double *eps, RngMetadata *meta) {
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, meta->initstate, meta->initseq);

    double eps_min = INFINITY;
    double eps_max = -INFINITY;
    double eps_sum = 0.0;
    for (int j = 0; j < cfg->L; ++j) {
        double u = pcg32_uniform_01(&rng);
        eps[j] = cfg->omega * (2.0 * u - 1.0);
        if (eps[j] < eps_min) {
            eps_min = eps[j];
        }
        if (eps[j] > eps_max) {
            eps_max = eps[j];
        }
        eps_sum += eps[j];
    }

    meta->epsilon_min = eps_min;
    meta->epsilon_max = eps_max;
    meta->epsilon_mean = eps_sum / (double)cfg->L;
}

static int build_hamiltonian(const Config *cfg, const double *eps, double *ham, int dim) {
    for (int state = 0; state < dim; ++state) {
        double diag = 0.0;
        for (int j = 0; j < cfg->L; ++j) {
            int bit_j = (state >> j) & 1;
            int bit_next = (state >> ((j + 1) % cfg->L)) & 1;
            int sz_j = 1 - 2 * bit_j;
            int sz_next = 1 - 2 * bit_next;
            int flipped = state ^ (1 << j);

            diag += -(double)(sz_j * sz_next);
            diag += -cfg->h * (double)sz_j;
            ham[(size_t)state * (size_t)dim + (size_t)flipped] += -cfg->g * (1.0 + eps[j]);
        }
        ham[(size_t)state * (size_t)dim + (size_t)state] += diag;
    }
    return 1;
}

static double finite_positive_or_nan(double value) {
    return (isfinite(value) && value > 0.0) ? value : NAN;
}

static int write_output(const Config *cfg, const double *eps, const double *evals,
                        int dim, int keep_start, int keep_end,
                        const double *delta, const double *s_global,
                        const double *s_local, const double *ratio,
                        int n_spacings, const RngMetadata *rng_meta) {
    char *tmp_path = path_with_suffix(cfg->out_path, ".tmp");
    if (tmp_path == NULL) {
        fprintf(stderr, "[ising_spectral] failed to allocate scratch path\n");
        return 0;
    }

    FILE *fp = fopen(tmp_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "[ising_spectral] failed to open %s: %s\n", tmp_path, strerror(errno));
        free(tmp_path);
        return 0;
    }

    fprintf(fp, "# spectral ED-only raw sample\n");
    fprintf(fp, "# Hamiltonian = -sum sz sz - g sum (1+eps_j) sx - h sum sz\n");
    fprintf(fp, "# L = %d\n", cfg->L);
    fprintf(fp, "# dim = %d\n", dim);
    fprintf(fp, "# pbc = 1\n");
    fprintf(fp, "# g = %.17e\n", cfg->g);
    fprintf(fp, "# h = %.17e\n", cfg->h);
    fprintf(fp, "# omega = %.17e\n", cfg->omega);
    fprintf(fp, "# realization = %ld\n", cfg->realization);
    fprintf(fp, "# seed = %lu\n", cfg->seed);
    fprintf(fp, "# rng = pcg32\n");
    fprintf(fp, "# master_seed = %lu\n", cfg->seed);
    fprintf(fp, "# initstate = %" PRIu64 "\n", rng_meta->initstate);
    fprintf(fp, "# initseq = %" PRIu64 "\n", rng_meta->initseq);
    fprintf(fp, "# epsilon_min = %.17e\n", rng_meta->epsilon_min);
    fprintf(fp, "# epsilon_max = %.17e\n", rng_meta->epsilon_max);
    fprintf(fp, "# epsilon_mean = %.17e\n", rng_meta->epsilon_mean);
    fprintf(fp, "# epsilon_summary = min %.17e max %.17e mean %.17e\n",
            rng_meta->epsilon_min, rng_meta->epsilon_max, rng_meta->epsilon_mean);
    fprintf(fp, "# bulk_frac = %.17e\n", cfg->bulk_frac);
    fprintf(fp, "# keep_start = %d\n", keep_start);
    fprintf(fp, "# keep_end = %d\n", keep_end);
    fprintf(fp, "# local_window = %d\n", cfg->local_window);
    fprintf(fp, "# epsilon_values =");
    for (int j = 0; j < cfg->L; ++j) {
        fprintf(fp, " %.17e", eps[j]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "# eps_values =");
    for (int j = 0; j < cfg->L; ++j) {
        fprintf(fp, " %.17e", eps[j]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "# columns = realization sample_index level_index E_i E_ip1 delta s_global s_local r\n");

    for (int s = 0; s < n_spacings; ++s) {
        int level_index = keep_start + s;
        fprintf(fp, "%ld %d %d %.17e %.17e %.17e %.17e %.17e %.17e\n",
                cfg->realization, s, level_index,
                evals[level_index], evals[level_index + 1],
                delta[s], s_global[s], s_local[s], ratio[s]);
    }

    if (fflush(fp) != 0) {
        fprintf(stderr, "[ising_spectral] fflush failed for %s: %s\n", tmp_path, strerror(errno));
        fclose(fp);
        unlink(tmp_path);
        free(tmp_path);
        return 0;
    }

    int fd = fileno(fp);
    if (fd >= 0 && fsync(fd) != 0) {
        fprintf(stderr, "[ising_spectral] fsync failed for %s: %s\n", tmp_path, strerror(errno));
        fclose(fp);
        unlink(tmp_path);
        free(tmp_path);
        return 0;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "[ising_spectral] fclose failed for %s: %s\n", tmp_path, strerror(errno));
        unlink(tmp_path);
        free(tmp_path);
        return 0;
    }

    if (rename(tmp_path, cfg->out_path) != 0) {
        fprintf(stderr, "[ising_spectral] rename %s -> %s failed: %s\n",
                tmp_path, cfg->out_path, strerror(errno));
        unlink(tmp_path);
        free(tmp_path);
        return 0;
    }

    free(tmp_path);
    return 1;
}

int main(int argc, char **argv) {
    Config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        return EXIT_FAILURE;
    }

    if (!cfg.overwrite && file_exists(cfg.out_path)) {
        fprintf(stderr, "[ising_spectral] output exists and --overwrite was not set: %s\n",
                cfg.out_path);
        return EXIT_FAILURE;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    if (!acquire_lock(cfg.out_path)) {
        return EXIT_FAILURE;
    }
    atexit(cleanup_lock);

    int dim = 1 << cfg.L;
    size_t n_matrix = (size_t)dim * (size_t)dim;
    double *eps = (double *)malloc((size_t)cfg.L * sizeof(double));
    double *ham = (double *)calloc(n_matrix, sizeof(double));
    double *evals = (double *)malloc((size_t)dim * sizeof(double));
    if (eps == NULL || ham == NULL || evals == NULL) {
        fprintf(stderr, "[ising_spectral] allocation failed for L=%d dim=%d\n", cfg.L, dim);
        free(eps);
        free(ham);
        free(evals);
        return EXIT_FAILURE;
    }

    RngMetadata rng_meta = derive_rng_metadata(&cfg);
    generate_epsilons(&cfg, eps, &rng_meta);

    build_hamiltonian(&cfg, eps, ham, dim);

    int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'U', dim, ham, dim, evals);
    if (info != 0) {
        fprintf(stderr, "[ising_spectral] LAPACKE_dsyevd failed with info=%d\n", info);
        free(eps);
        free(ham);
        free(evals);
        return EXIT_FAILURE;
    }

    int keep_start = (int)floor(0.5 * (1.0 - cfg.bulk_frac) * (double)dim);
    int keep_end = (int)ceil(0.5 * (1.0 + cfg.bulk_frac) * (double)dim);
    if (keep_start < 0) {
        keep_start = 0;
    }
    if (keep_end > dim) {
        keep_end = dim;
    }
    int n_spacings = keep_end - keep_start - 1;
    if (n_spacings <= 0) {
        fprintf(stderr, "[ising_spectral] bulk selection has no spacings\n");
        free(eps);
        free(ham);
        free(evals);
        return EXIT_FAILURE;
    }

    double *delta = (double *)malloc((size_t)n_spacings * sizeof(double));
    double *s_global = (double *)malloc((size_t)n_spacings * sizeof(double));
    double *s_local = (double *)malloc((size_t)n_spacings * sizeof(double));
    double *ratio = (double *)malloc((size_t)n_spacings * sizeof(double));
    if (delta == NULL || s_global == NULL || s_local == NULL || ratio == NULL) {
        fprintf(stderr, "[ising_spectral] spacing allocation failed\n");
        free(eps);
        free(ham);
        free(evals);
        free(delta);
        free(s_global);
        free(s_local);
        free(ratio);
        return EXIT_FAILURE;
    }

    double mean_delta = 0.0;
    int mean_count = 0;
    for (int s = 0; s < n_spacings; ++s) {
        int level_index = keep_start + s;
        delta[s] = evals[level_index + 1] - evals[level_index];
        if (isfinite(delta[s]) && delta[s] > 0.0) {
            mean_delta += delta[s];
            ++mean_count;
        }
    }
    mean_delta = (mean_count > 0) ? mean_delta / (double)mean_count : NAN;

    for (int s = 0; s < n_spacings; ++s) {
        s_global[s] = (isfinite(mean_delta) && mean_delta > 0.0) ? delta[s] / mean_delta : NAN;

        int i0 = s - cfg.local_window;
        int i1 = s + cfg.local_window;
        if (i0 < 0) {
            i0 = 0;
        }
        if (i1 >= n_spacings) {
            i1 = n_spacings - 1;
        }
        double local_sum = 0.0;
        int local_count = 0;
        for (int k = i0; k <= i1; ++k) {
            double valid = finite_positive_or_nan(delta[k]);
            if (isfinite(valid)) {
                local_sum += valid;
                ++local_count;
            }
        }
        double local_mean = (local_count >= 3) ? local_sum / (double)local_count : NAN;
        s_local[s] = (isfinite(local_mean) && local_mean > 0.0) ? delta[s] / local_mean : NAN;

        if (s == 0) {
            ratio[s] = NAN;
        } else if (isfinite(delta[s]) && isfinite(delta[s - 1]) &&
                   delta[s] > 0.0 && delta[s - 1] > 0.0) {
            double dmin = (delta[s] < delta[s - 1]) ? delta[s] : delta[s - 1];
            double dmax = (delta[s] > delta[s - 1]) ? delta[s] : delta[s - 1];
            ratio[s] = dmin / dmax;
        } else {
            ratio[s] = NAN;
        }
    }

    int ok = write_output(&cfg, eps, evals, dim, keep_start, keep_end,
                          delta, s_global, s_local, ratio, n_spacings, &rng_meta);

    free(eps);
    free(ham);
    free(evals);
    free(delta);
    free(s_global);
    free(s_local);
    free(ratio);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
