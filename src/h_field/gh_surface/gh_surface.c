#define _POSIX_C_SOURCE 200809L

/*
 * Two-parameter (g,h) surface worker for the 1D quantum Ising chain.
 *
 * Hamiltonian convention:
 *   H = - sum_j sigma^z_j sigma^z_{j+1}
 *       - g * sum_j sigma^x_j
 *       - h * sum_j sigma^z_j
 *
 * The input grid is external and each useful row contains:
 *   g h kappa_g kappa_h
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "basis.h"
#include "diag.h"
#include "hamiltonian.h"
#include "lanczos.h"
#include "observables.h"

#define ED_L_MAX 12
#define MAX_L_VALUE 28
#define MAX_PATH_LEN 1024
#define MAX_COMMAND_LEN 2048
#define N_EIG_SOLVE 4
#define N_EIG_WRITE 3
#define GRID_TOL 1e-10

typedef enum {
    GRID_PHYSICAL = 0,
    GRID_SCALING = 1
} GridType;

typedef struct {
    int L;
    int pbc;
    GridType grid_type;
    char grid[MAX_PATH_LEN];
    char out[MAX_PATH_LEN];
    unsigned long seed;
    bool resume;
    bool overwrite;
    bool force_unlock;
    char command[MAX_COMMAND_LEN];
} Params;

typedef struct {
    double g;
    double h;
    double kg;
    double kh;
} GridPoint;

typedef struct {
    GridPoint *points;
    int n;
    int cap;
    double g_min, g_max;
    double h_min, h_max;
    double kg_min, kg_max;
    double kh_min, kh_max;
    int kg_finite;
    int kh_finite;
} Grid;

typedef struct {
    double evals[N_EIG_SOLVE];
    double resid[N_EIG_SOLVE];
    double mz;
    double abs_mz;
    double mx;
    int method_code;
    int lanczos_ret;
} SolveResult;

typedef struct {
    GridPoint p;
    double evals[N_EIG_WRITE];
    double delta0;
    double delta1;
    double mz;
    double abs_mz;
    double mx;
    int method_code;
    double resid[N_EIG_WRITE];
} DataRow;

typedef struct {
    bool held;
    char path[MAX_PATH_LEN];
} LockFile;

#ifndef _WIN32
static volatile sig_atomic_t g_lock_held = 0;
static char g_active_lock_path[MAX_PATH_LEN];
#endif

static void usage(const char *prog);
static Params parse_args(int argc, char **argv);
static int parse_int_arg(const char *s, const char *name);
static unsigned long parse_ulong_arg(const char *s, const char *name);
static int parse_pbc(const char *s);
static GridType parse_grid_type(const char *s);
static const char *grid_type_name(GridType t);
static void build_command_line(int argc, char **argv, char *buf, size_t n);
static bool path_exists(const char *path);
static int remove_existing_outputs(const char *path);
static void build_lock_path(const char *output_path, char *lock_path, size_t n);
static int acquire_lock(const char *output_path, const Params *P, LockFile *lock);
static int release_lock(LockFile *lock);
static int fsync_stream(FILE *fp, const char *path);
static int write_checkpoint(const char *path, const Params *P, const Grid *grid,
                            int completed, int skipped, int newly_computed,
                            int last_index, double run_t0);
static int load_grid(const char *path, Grid *grid);
static int grid_push(Grid *grid, GridPoint p);
static void free_grid(Grid *grid);
static void update_grid_ranges(Grid *grid, GridPoint p);
static int parse_existing_rows(const char *path, const Grid *grid,
                               bool *completed, int *n_completed);
static int verify_header_compatible(const char *path, const Params *P,
                                    const Grid *grid);
static bool read_header_value(const char *path, const char *key,
                              char *value, size_t n);
static void trim_right(char *s);
static bool parse_output_row(const char *line, DataRow *row);
static bool parse_double_field(const char **p, double *out);
static bool parse_int_field(const char **p, int *out);
static int match_grid_index(const Grid *grid, GridPoint p);
static bool coord_close(double a, double b);
static int write_new_header(const char *path, const Params *P,
                            const Grid *grid, const char *method);
static int append_data_row(const char *path, const GridPoint *p,
                           const SolveResult *r);
static int solve_low_energy(const Basis *b, double g, double h,
                            unsigned long seed, SolveResult *out);
static int solve_ed(const Basis *b, double g, double h, SolveResult *out);
static int solve_lanczos(const Basis *b, double g, double h,
                         unsigned long seed, SolveResult *out);
static int lanczos_max_iter(int L);
static void print_allocation_failure(const char *where, const Basis *b,
                                     double g, double h);
static double now_seconds(void);
#ifndef _WIN32
static void handle_signal(int sig);
static int install_signal_handlers(void);
#endif

int main(int argc, char **argv)
{
    Params P = parse_args(argc, argv);

#ifndef _WIN32
    if (install_signal_handlers() != EXIT_SUCCESS)
        return EXIT_FAILURE;
#endif

    Grid grid;
    memset(&grid, 0, sizeof(grid));
    if (load_grid(P.grid, &grid) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (grid.n <= 0) {
        fprintf(stderr, "[ERROR] empty grid: %s\n", P.grid);
        free_grid(&grid);
        return EXIT_FAILURE;
    }

    char lock_path[MAX_PATH_LEN];
    build_lock_path(P.out, lock_path, sizeof(lock_path));
    if (P.force_unlock && lock_path[0] != '\0') {
        if (remove(lock_path) != EXIT_SUCCESS && errno != ENOENT) {
            fprintf(stderr, "[ERROR] cannot remove lock %s: %s\n",
                    lock_path, strerror(errno));
            free_grid(&grid);
            return EXIT_FAILURE;
        }
    }

    LockFile lock;
    memset(&lock, 0, sizeof(lock));
    if (acquire_lock(P.out, &P, &lock) != EXIT_SUCCESS) {
        free_grid(&grid);
        return EXIT_FAILURE;
    }

    int status = EXIT_FAILURE;
    bool *completed = calloc((size_t)grid.n, sizeof(bool));
    if (!completed) {
        fprintf(stderr, "[ERROR] calloc completed flags failed\n");
        goto cleanup;
    }

    const char *method = (P.L <= ED_L_MAX) ? "ED" : "Lanczos";
    if (P.overwrite) {
        if (remove_existing_outputs(P.out) != EXIT_SUCCESS)
            goto cleanup;
        if (write_new_header(P.out, &P, &grid, method) != EXIT_SUCCESS)
            goto cleanup;
    } else if (P.resume && path_exists(P.out)) {
        int completed_before = 0;
        if (verify_header_compatible(P.out, &P, &grid) != EXIT_SUCCESS)
            goto cleanup;
        if (parse_existing_rows(P.out, &grid, completed,
                                &completed_before) != EXIT_SUCCESS)
            goto cleanup;
        printf("resume_existing_rows = %d\n", completed_before);
    } else if (path_exists(P.out)) {
        fprintf(stderr,
                "[ERROR] output already exists: %s\n"
                "        Pass --resume or --overwrite.\n",
                P.out);
        goto cleanup;
    } else {
        if (write_new_header(P.out, &P, &grid, method) != EXIT_SUCCESS)
            goto cleanup;
    }

    Basis b = basis_init(P.L, P.pbc);
    int completed_initial = 0;
    for (int i = 0; i < grid.n; i++)
        if (completed[i])
            completed_initial++;

    printf("=====================================================\n");
    printf("|  1D QUANTUM ISING - GH SURFACE WORKER             |\n");
    printf("=====================================================\n");
    printf(" convention = H=-sum zz - g sum sx - h sum sz\n");
    printf(" L          = %d\n", P.L);
    printf(" pbc        = %d (%s)\n", P.pbc, P.pbc ? "PBC" : "OBC");
    printf(" grid_type  = %s\n", grid_type_name(P.grid_type));
    printf(" grid       = %s\n", P.grid);
    printf(" output     = %s\n", P.out);
    printf(" n_points   = %d\n", grid.n);
    printf(" g_min/max  = %.17e %.17e\n", grid.g_min, grid.g_max);
    printf(" h_min/max  = %.17e %.17e\n", grid.h_min, grid.h_max);
    printf(" kg_min/max = %.17e %.17e\n", grid.kg_min, grid.kg_max);
    printf(" kh_min/max = %.17e %.17e\n", grid.kh_min, grid.kh_max);
    printf(" method     = %s\n", method);
    printf(" seed       = %lu\n", P.seed);
    printf(" resume     = %s (completed=%d)\n",
           P.resume ? "yes" : "no", completed_initial);
    printf(" overwrite  = %s\n", P.overwrite ? "yes" : "no");
    fflush(stdout);

    int newly_computed = 0;
    int completed_total = completed_initial;
    int last_index = -1;
    double run_t0 = now_seconds();

    for (int i = 0; i < grid.n; i++) {
        if (completed[i])
            continue;

        double t0 = now_seconds();
        SolveResult r;
        memset(&r, 0, sizeof(r));
        unsigned long seed = P.seed + (unsigned long)P.L * 104729UL
                           + (unsigned long)i * 8191UL + 709UL;
        if (solve_low_energy(&b, grid.points[i].g, grid.points[i].h,
                             seed, &r) != EXIT_SUCCESS)
            goto cleanup;

        if (append_data_row(P.out, &grid.points[i], &r) != EXIT_SUCCESS)
            goto cleanup;

        completed[i] = true;
        newly_computed++;
        completed_total++;
        last_index = i;

        if (write_checkpoint(P.out, &P, &grid, completed_total,
                             completed_initial, newly_computed,
                             last_index, run_t0) != EXIT_SUCCESS)
            goto cleanup;

        double dt = now_seconds() - t0;
        double elapsed = now_seconds() - run_t0;
        double avg = (newly_computed > 0) ? elapsed / (double)newly_computed : 0.0;
        printf("point %d/%d g=%.17e h=%.17e Delta0=%.17e mz=%.17e "
               "dt=%.2fs avg=%.2fs\n",
               i + 1, grid.n, grid.points[i].g, grid.points[i].h,
               r.evals[1] - r.evals[0], r.mz, dt, avg);
        fflush(stdout);
    }

    if (write_checkpoint(P.out, &P, &grid, completed_total,
                         completed_initial, newly_computed,
                         last_index, run_t0) != EXIT_SUCCESS)
        goto cleanup;

    double elapsed = now_seconds() - run_t0;
    double avg = (newly_computed > 0) ? elapsed / (double)newly_computed : 0.0;
    printf("done L=%d pbc=%d grid_type=%s rows_total=%d "
           "completed_valid_rows=%d skipped_rows_this_run=%d "
           "newly_computed_rows_this_run=%d avg_seconds_per_new_point=%.6g "
           "output=%s\n",
           P.L, P.pbc, grid_type_name(P.grid_type), grid.n, completed_total,
           completed_initial, newly_computed, avg, P.out);
    fflush(stdout);
    status = EXIT_SUCCESS;

cleanup:
    free(completed);
    if (release_lock(&lock) != EXIT_SUCCESS)
        status = EXIT_FAILURE;
    free_grid(&grid);
    if (status == EXIT_SUCCESS)
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "\nUsage: %s --L L --pbc 0|1 --grid-type physical|scaling "
            "--grid file --out file [options]\n\n"
            "Options:\n"
            "  --seed SEED       default 42\n"
            "  --resume\n"
            "  --overwrite\n"
            "  --force-unlock    remove this output's lock before starting\n\n",
            prog);
}

static Params parse_args(int argc, char **argv)
{
    Params P;
    memset(&P, 0, sizeof(P));
    P.L = -1;
    P.pbc = -1;
    P.seed = 42UL;
    P.grid_type = GRID_PHYSICAL;
    build_command_line(argc, argv, P.command, sizeof(P.command));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --L requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.L = parse_int_arg(argv[i], "L");
        } else if (strcmp(argv[i], "--pbc") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --pbc requires 0 or 1\n");
                exit(EXIT_FAILURE);
            }
            P.pbc = parse_pbc(argv[i]);
        } else if (strcmp(argv[i], "--grid-type") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --grid-type requires physical or scaling\n");
                exit(EXIT_FAILURE);
            }
            P.grid_type = parse_grid_type(argv[i]);
        } else if (strcmp(argv[i], "--grid") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --grid requires a path\n");
                exit(EXIT_FAILURE);
            }
            snprintf(P.grid, sizeof(P.grid), "%s", argv[i]);
        } else if (strcmp(argv[i], "--out") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --out requires a path\n");
                exit(EXIT_FAILURE);
            }
            snprintf(P.out, sizeof(P.out), "%s", argv[i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "[ERROR] --seed requires a value\n");
                exit(EXIT_FAILURE);
            }
            P.seed = parse_ulong_arg(argv[i], "seed");
        } else if (strcmp(argv[i], "--resume") == 0) {
            P.resume = true;
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            P.overwrite = true;
        } else if (strcmp(argv[i], "--force-unlock") == 0) {
            P.force_unlock = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "[ERROR] unknown option '%s'\n", argv[i]);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (P.L < 2 || P.L > MAX_L_VALUE) {
        fprintf(stderr, "[ERROR] L=%d out of supported range [2,%d]\n",
                P.L, MAX_L_VALUE);
        exit(EXIT_FAILURE);
    }
    if (P.pbc != 0 && P.pbc != 1) {
        fprintf(stderr, "[ERROR] missing or invalid --pbc\n");
        exit(EXIT_FAILURE);
    }
    if (P.grid[0] == '\0') {
        fprintf(stderr, "[ERROR] missing --grid\n");
        exit(EXIT_FAILURE);
    }
    if (P.out[0] == '\0') {
        fprintf(stderr, "[ERROR] missing --out\n");
        exit(EXIT_FAILURE);
    }
    if (P.resume && P.overwrite) {
        fprintf(stderr, "[ERROR] --resume and --overwrite are mutually exclusive\n");
        exit(EXIT_FAILURE);
    }
    return P;
}

static int parse_int_arg(const char *s, const char *name)
{
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno ||
        v < -2147483647L || v > 2147483647L) {
        fprintf(stderr, "[ERROR] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static unsigned long parse_ulong_arg(const char *s, const char *name)
{
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || errno) {
        fprintf(stderr, "[ERROR] invalid %s: '%s'\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int parse_pbc(const char *s)
{
    int pbc = parse_int_arg(s, "pbc");
    if (pbc != 0 && pbc != 1) {
        fprintf(stderr, "[ERROR] pbc must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    return pbc;
}

static GridType parse_grid_type(const char *s)
{
    if (strcmp(s, "physical") == 0)
        return GRID_PHYSICAL;
    if (strcmp(s, "scaling") == 0)
        return GRID_SCALING;
    fprintf(stderr, "[ERROR] --grid-type must be physical or scaling\n");
    exit(EXIT_FAILURE);
}

static const char *grid_type_name(GridType t)
{
    return (t == GRID_SCALING) ? "scaling" : "physical";
}

static void build_command_line(int argc, char **argv, char *buf, size_t n)
{
    size_t used = 0;
    if (n == 0)
        return;
    buf[0] = '\0';
    for (int i = 0; i < argc; i++) {
        int written = snprintf(buf + used, n - used, "%s%s",
                               (i == 0) ? "" : " ", argv[i]);
        if (written < 0)
            break;
        if ((size_t)written >= n - used) {
            buf[n - 1] = '\0';
            break;
        }
        used += (size_t)written;
    }
}

static bool path_exists(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return false;
    fclose(fp);
    return true;
}

static int remove_existing_outputs(const char *path)
{
    char ckpt[MAX_PATH_LEN + 8];
    snprintf(ckpt, sizeof(ckpt), "%s.ckpt", path);
    if (remove(path) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] cannot remove %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (remove(ckpt) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] cannot remove %s: %s\n", ckpt, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static void build_lock_path(const char *output_path, char *lock_path, size_t n)
{
    int written = snprintf(lock_path, n, "%s.lock", output_path);
    if (written < 0 || (size_t)written >= n)
        lock_path[0] = '\0';
}

static int acquire_lock(const char *output_path, const Params *P, LockFile *lock)
{
    build_lock_path(output_path, lock->path, sizeof(lock->path));
    lock->held = false;
    if (lock->path[0] == '\0') {
        fprintf(stderr, "[ERROR] lock path too long for %s\n", output_path);
        return EXIT_FAILURE;
    }

#ifndef _WIN32
    int fd = open(lock->path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] failed to acquire lock %s: %s\n",
                lock->path, strerror(errno));
        return EXIT_FAILURE;
    }
    FILE *fp = fdopen(fd, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] fdopen failed for lock %s: %s\n",
                lock->path, strerror(errno));
        close(fd);
        remove(lock->path);
        return EXIT_FAILURE;
    }
#else
    FILE *fp = fopen(lock->path, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] failed to create lock %s: %s\n",
                lock->path, strerror(errno));
        return EXIT_FAILURE;
    }
#endif

    time_t now = time(NULL);
    int rc = fprintf(fp,
            "output = %s\n"
            "timestamp = %ld\n"
            "command = %s\n",
            output_path, (long)now, P->command);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] failed while writing lock %s\n", lock->path);
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for lock %s: %s\n",
                lock->path, strerror(errno));
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, lock->path) != EXIT_SUCCESS) {
        fclose(fp);
        remove(lock->path);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for lock %s: %s\n",
                lock->path, strerror(errno));
        remove(lock->path);
        return EXIT_FAILURE;
    }

    lock->held = true;
#ifndef _WIN32
    strncpy(g_active_lock_path, lock->path, sizeof(g_active_lock_path) - 1);
    g_active_lock_path[sizeof(g_active_lock_path) - 1] = '\0';
    g_lock_held = 1;
#endif
    return EXIT_SUCCESS;
}

static int release_lock(LockFile *lock)
{
    if (!lock->held)
        return EXIT_SUCCESS;
    if (remove(lock->path) != EXIT_SUCCESS && errno != ENOENT) {
        fprintf(stderr, "[ERROR] failed to remove lock %s: %s\n",
                lock->path, strerror(errno));
        return EXIT_FAILURE;
    }
    lock->held = false;
#ifndef _WIN32
    g_lock_held = 0;
    g_active_lock_path[0] = '\0';
#endif
    return EXIT_SUCCESS;
}

static int fsync_stream(FILE *fp, const char *path)
{
#ifndef _WIN32
    int fd = fileno(fp);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] fileno failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (fsync(fd) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fsync failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
#else
    (void)fp;
    (void)path;
#endif
    return EXIT_SUCCESS;
}

static int write_checkpoint(const char *path, const Params *P, const Grid *grid,
                            int completed, int skipped, int newly_computed,
                            int last_index, double run_t0)
{
    char ckpt[MAX_PATH_LEN + 8], tmp[MAX_PATH_LEN + 16];
    snprintf(ckpt, sizeof(ckpt), "%s.ckpt", path);
    snprintf(tmp, sizeof(tmp), "%s.tmp", ckpt);
    FILE *fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot write checkpoint %s: %s\n",
                tmp, strerror(errno));
        return EXIT_FAILURE;
    }

    double elapsed = now_seconds() - run_t0;
    double avg = (newly_computed > 0) ? elapsed / (double)newly_computed : NAN;
    GridPoint last = {NAN, NAN, NAN, NAN};
    if (last_index >= 0 && last_index < grid->n)
        last = grid->points[last_index];

    int rc = fprintf(fp,
            "output = %s\n"
            "L = %d\n"
            "pbc = %d\n"
            "BC = %s\n"
            "grid_type = %s\n"
            "grid = %s\n"
            "total_points = %d\n"
            "completed_valid_rows = %d\n"
            "skipped_rows_this_run = %d\n"
            "newly_computed_rows_this_run = %d\n"
            "last_index = %d\n"
            "last_g = %.17e\n"
            "last_h = %.17e\n"
            "last_kappa_g = %.17e\n"
            "last_kappa_h = %.17e\n"
            "avg_seconds_per_new_point = %.17g\n"
            "elapsed_seconds_this_run = %.17g\n"
            "command = %s\n",
            path, P->L, P->pbc, P->pbc ? "PBC" : "OBC",
            grid_type_name(P->grid_type), P->grid, grid->n, completed,
            skipped, newly_computed, last_index, last.g, last.h, last.kg,
            last.kh, avg, elapsed, P->command);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] fprintf failed for checkpoint %s\n", tmp);
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for checkpoint %s: %s\n",
                tmp, strerror(errno));
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, tmp) != EXIT_SUCCESS) {
        fclose(fp);
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for checkpoint %s: %s\n",
                tmp, strerror(errno));
        remove(tmp);
        return EXIT_FAILURE;
    }
    if (rename(tmp, ckpt) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] rename checkpoint %s -> %s failed: %s\n",
                tmp, ckpt, strerror(errno));
        remove(tmp);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int load_grid(const char *path, Grid *grid)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot open grid %s: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }

    grid->g_min = INFINITY;
    grid->g_max = -INFINITY;
    grid->h_min = INFINITY;
    grid->h_max = -INFINITY;
    grid->kg_min = INFINITY;
    grid->kg_max = -INFINITY;
    grid->kh_min = INFINITY;
    grid->kh_max = -INFINITY;

    char line[4096];
    int line_no = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '\0' || *p == '\n' || *p == '#')
            continue;

        const char *q = p;
        GridPoint gp;
        if (!parse_double_field(&q, &gp.g) ||
            !parse_double_field(&q, &gp.h) ||
            !parse_double_field(&q, &gp.kg) ||
            !parse_double_field(&q, &gp.kh)) {
            fprintf(stderr, "[ERROR] invalid grid row in %s:%d\n", path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }
        if (!isfinite(gp.g) || !isfinite(gp.h)) {
            fprintf(stderr, "[ERROR] non-finite g or h in %s:%d\n", path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }
        while (isspace((unsigned char)*q))
            q++;
        if (*q != '\0' && *q != '\n' && *q != '#') {
            fprintf(stderr, "[ERROR] trailing tokens in %s:%d\n", path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }
        if (grid_push(grid, gp) != EXIT_SUCCESS) {
            fclose(fp);
            return EXIT_FAILURE;
        }
        update_grid_ranges(grid, gp);
    }

    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for grid %s: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (grid->n == 0)
        return EXIT_SUCCESS;
    if (grid->kg_finite == 0) {
        grid->kg_min = NAN;
        grid->kg_max = NAN;
    }
    if (grid->kh_finite == 0) {
        grid->kh_min = NAN;
        grid->kh_max = NAN;
    }
    return EXIT_SUCCESS;
}

static int grid_push(Grid *grid, GridPoint p)
{
    if (grid->n == grid->cap) {
        int new_cap = (grid->cap == 0) ? 1024 : 2 * grid->cap;
        GridPoint *new_points = realloc(grid->points,
                                        (size_t)new_cap * sizeof(GridPoint));
        if (!new_points) {
            fprintf(stderr, "[ERROR] realloc grid failed\n");
            return EXIT_FAILURE;
        }
        grid->points = new_points;
        grid->cap = new_cap;
    }
    grid->points[grid->n++] = p;
    return EXIT_SUCCESS;
}

static void free_grid(Grid *grid)
{
    free(grid->points);
    grid->points = NULL;
    grid->n = 0;
    grid->cap = 0;
}

static void update_grid_ranges(Grid *grid, GridPoint p)
{
    if (p.g < grid->g_min) grid->g_min = p.g;
    if (p.g > grid->g_max) grid->g_max = p.g;
    if (p.h < grid->h_min) grid->h_min = p.h;
    if (p.h > grid->h_max) grid->h_max = p.h;
    if (isfinite(p.kg)) {
        if (p.kg < grid->kg_min) grid->kg_min = p.kg;
        if (p.kg > grid->kg_max) grid->kg_max = p.kg;
        grid->kg_finite++;
    }
    if (isfinite(p.kh)) {
        if (p.kh < grid->kh_min) grid->kh_min = p.kh;
        if (p.kh > grid->kh_max) grid->kh_max = p.kh;
        grid->kh_finite++;
    }
}

static int parse_existing_rows(const char *path, const Grid *grid,
                               bool *completed, int *n_completed)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot read %s for resume: %s\n",
                path, strerror(errno));
        return EXIT_FAILURE;
    }

    char line[4096];
    int line_no = 0;
    *n_completed = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '\0' || *p == '\n' || *p == '#')
            continue;

        DataRow row;
        if (!parse_output_row(p, &row)) {
            fprintf(stderr, "[ERROR] invalid row while resuming %s:%d\n",
                    path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }
        int idx = match_grid_index(grid, row.p);
        if (idx < 0) {
            fprintf(stderr,
                    "[ERROR] row g=%.17e h=%.17e kg=%.17e kh=%.17e "
                    "in %s:%d is not present in --grid\n",
                    row.p.g, row.p.h, row.p.kg, row.p.kh, path, line_no);
            fclose(fp);
            return EXIT_FAILURE;
        }
        if (!completed[idx]) {
            completed[idx] = true;
            (*n_completed)++;
        }
    }

    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int verify_header_compatible(const char *path, const Params *P,
                                    const Grid *grid)
{
    char value[MAX_PATH_LEN];
    char expected[MAX_PATH_LEN];

    if (!read_header_value(path, "L", value, sizeof(value)) ||
        atoi(value) != P->L) {
        fprintf(stderr, "[ERROR] resume header mismatch for L in %s\n", path);
        return EXIT_FAILURE;
    }
    if (!read_header_value(path, "pbc", value, sizeof(value)) ||
        atoi(value) != P->pbc) {
        fprintf(stderr, "[ERROR] resume header mismatch for pbc in %s\n", path);
        return EXIT_FAILURE;
    }
    if (!read_header_value(path, "grid_type", value, sizeof(value)) ||
        strcmp(value, grid_type_name(P->grid_type)) != 0) {
        fprintf(stderr, "[ERROR] resume header mismatch for grid_type in %s\n", path);
        return EXIT_FAILURE;
    }
    if (!read_header_value(path, "N_points", value, sizeof(value)) ||
        atoi(value) != grid->n) {
        fprintf(stderr, "[ERROR] resume header mismatch for N_points in %s\n", path);
        return EXIT_FAILURE;
    }
    if (!read_header_value(path, "grid", value, sizeof(value))) {
        fprintf(stderr, "[ERROR] resume header missing grid path in %s\n", path);
        return EXIT_FAILURE;
    }
    snprintf(expected, sizeof(expected), "%s", P->grid);
    if (strcmp(value, expected) != 0) {
        fprintf(stderr,
                "[ERROR] resume header mismatch for grid path in %s\n"
                "        existing: %s\n"
                "        current : %s\n",
                path, value, expected);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static bool read_header_value(const char *path, const char *key,
                              char *value, size_t n)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return false;

    char line[4096];
    size_t key_len = strlen(key);
    bool found = false;
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p != '#') {
            if (*p != '\0' && *p != '\n')
                break;
            continue;
        }
        p++;
        while (isspace((unsigned char)*p))
            p++;
        if (strncmp(p, key, key_len) != 0)
            continue;
        p += key_len;
        while (isspace((unsigned char)*p))
            p++;
        if (*p != '=')
            continue;
        p++;
        while (isspace((unsigned char)*p))
            p++;
        snprintf(value, n, "%s", p);
        trim_right(value);
        found = true;
        break;
    }

    fclose(fp);
    return found;
}

static void trim_right(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) {
        s[n - 1] = '\0';
        n--;
    }
}

static bool parse_output_row(const char *line, DataRow *row)
{
    const char *p = line;
    if (!parse_double_field(&p, &row->p.g)) return false;
    if (!parse_double_field(&p, &row->p.h)) return false;
    if (!parse_double_field(&p, &row->p.kg)) return false;
    if (!parse_double_field(&p, &row->p.kh)) return false;
    for (int k = 0; k < N_EIG_WRITE; k++)
        if (!parse_double_field(&p, &row->evals[k])) return false;
    if (!parse_double_field(&p, &row->delta0)) return false;
    if (!parse_double_field(&p, &row->delta1)) return false;
    if (!parse_double_field(&p, &row->mz)) return false;
    if (!parse_double_field(&p, &row->abs_mz)) return false;
    if (!parse_double_field(&p, &row->mx)) return false;
    if (!parse_int_field(&p, &row->method_code)) return false;
    for (int k = 0; k < N_EIG_WRITE; k++)
        if (!parse_double_field(&p, &row->resid[k])) return false;
    while (isspace((unsigned char)*p))
        p++;
    return *p == '\0';
}

static bool parse_double_field(const char **p, double *out)
{
    while (isspace((unsigned char)**p))
        (*p)++;
    if (**p == '\0')
        return false;
    errno = 0;
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p || errno)
        return false;
    *out = v;
    *p = end;
    return true;
}

static bool parse_int_field(const char **p, int *out)
{
    while (isspace((unsigned char)**p))
        (*p)++;
    if (**p == '\0')
        return false;
    errno = 0;
    char *end = NULL;
    long v = strtol(*p, &end, 10);
    if (end == *p || errno || v < -2147483647L || v > 2147483647L)
        return false;
    *out = (int)v;
    *p = end;
    return true;
}

static int match_grid_index(const Grid *grid, GridPoint p)
{
    for (int i = 0; i < grid->n; i++) {
        const GridPoint *q = &grid->points[i];
        if (coord_close(p.g, q->g) && coord_close(p.h, q->h) &&
            coord_close(p.kg, q->kg) && coord_close(p.kh, q->kh))
            return i;
    }
    return -1;
}

static bool coord_close(double a, double b)
{
    if (isnan(a) && isnan(b))
        return true;
    if (!isfinite(a) || !isfinite(b))
        return false;
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return fabs(a - b) <= GRID_TOL * scale;
}

static int write_new_header(const char *path, const Params *P,
                            const Grid *grid, const char *method)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot create %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    time_t now = time(NULL);
    int rc = fprintf(fp,
            "# 1D Quantum Ising -- gh surface output\n"
            "# Generated: %s"
            "# Hamiltonian convention: H = -sum_j sigma^z_j sigma^z_{j+1}"
            " - g sum_j sigma^x_j - h sum_j sigma^z_j\n"
            "# order_parameter = mz = <sum_j sigma^z_j>/L\n"
            "# grid_type = %s\n"
            "# L = %d\n"
            "# pbc = %d\n"
            "# BC = %s\n"
            "# seed = %lu\n"
            "# grid = %s\n"
            "# g_min = %.17e\n"
            "# g_max = %.17e\n"
            "# h_min = %.17e\n"
            "# h_max = %.17e\n"
            "# kappa_g_min = %.17e\n"
            "# kappa_g_max = %.17e\n"
            "# kappa_h_min = %.17e\n"
            "# kappa_h_max = %.17e\n"
            "# N_points = %d\n"
            "# method = %s\n"
            "# method_code: 0=ED, 1=Lanczos\n"
            "# residuals: Lanczos API does not expose them here; NaN unless "
            "available in a future implementation\n"
            "# columns = g h kappa_g kappa_h E0 E1 E2 Delta0 Delta1 "
            "mz abs_mz mx method_code resid0 resid1 resid2\n"
            "# command = %s\n"
            "#\n",
            ctime(&now), grid_type_name(P->grid_type), P->L, P->pbc,
            P->pbc ? "PBC" : "OBC", P->seed, P->grid,
            grid->g_min, grid->g_max, grid->h_min, grid->h_max,
            grid->kg_min, grid->kg_max, grid->kh_min, grid->kh_max,
            grid->n, method, P->command);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] failed to write header to %s\n", path);
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, path) != EXIT_SUCCESS) {
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int append_data_row(const char *path, const GridPoint *p,
                           const SolveResult *r)
{
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "[ERROR] cannot append to %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    double delta0 = r->evals[1] - r->evals[0];
    double delta1 = r->evals[2] - r->evals[0];
    int rc = fprintf(fp,
            "%.17e %.17e %.17e %.17e "
            "%.17e %.17e %.17e %.17e %.17e "
            "%.17e %.17e %.17e %d %.17e %.17e %.17e\n",
            p->g, p->h, p->kg, p->kh,
            r->evals[0], r->evals[1], r->evals[2], delta0, delta1,
            r->mz, r->abs_mz, r->mx, r->method_code,
            r->resid[0], r->resid[1], r->resid[2]);
    if (rc < 0) {
        fprintf(stderr, "[ERROR] fprintf failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fflush(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fflush failed for %s: %s\n", path, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fsync_stream(fp, path) != EXIT_SUCCESS) {
        fclose(fp);
        return EXIT_FAILURE;
    }
    if (fclose(fp) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] fclose failed for %s: %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int solve_low_energy(const Basis *b, double g, double h,
                            unsigned long seed, SolveResult *out)
{
    if (b->L <= ED_L_MAX)
        return solve_ed(b, g, h, out);
    return solve_lanczos(b, g, h, seed, out);
}

static int solve_ed(const Basis *b, double g, double h, SolveResult *out)
{
    long long dim = b->dim;
    double *Ham = calloc((size_t)dim * dim, sizeof(double));
    double *eig = malloc((size_t)dim * sizeof(double));
    if (!Ham || !eig) {
        print_allocation_failure("ED", b, g, h);
        free(Ham);
        free(eig);
        return EXIT_FAILURE;
    }

    build_ham(b, g, h, Ham);
    if (diagonalize(Ham, dim, eig) != EXIT_SUCCESS) {
        free(Ham);
        free(eig);
        return EXIT_FAILURE;
    }

    for (int k = 0; k < N_EIG_SOLVE; k++) {
        out->evals[k] = (k < dim) ? eig[k] : NAN;
        out->resid[k] = NAN;
    }
    out->mz = obs_mz_raw(b, Ham);
    out->abs_mz = obs_psi_bar(b, Ham);
    out->mx = obs_mx(b, Ham);
    out->method_code = 0;
    out->lanczos_ret = EXIT_SUCCESS;

    free(Ham);
    free(eig);
    return EXIT_SUCCESS;
}

static int solve_lanczos(const Basis *b, double g, double h,
                         unsigned long seed, SolveResult *out)
{
    static bool residual_warning_printed = false;
    long long dim = b->dim;
    double evals[N_EIG_SOLVE] = {NAN, NAN, NAN, NAN};
    double *evecs = malloc((size_t)N_EIG_SOLVE * dim * sizeof(double));
    if (!evecs) {
        print_allocation_failure("Lanczos eigenvectors", b, g, h);
        return EXIT_FAILURE;
    }

    LanczosParams par = {
        .n_eig = N_EIG_SOLVE,
        .max_iter = lanczos_max_iter(b->L),
        .tol = 1e-10,
        .max_restarts = 80,
        .verbose = 0,
        .seed = seed,
    };

    int ret = lanczos(b, g, h, &par, evals, evecs);
    if (ret != EXIT_SUCCESS && ret != -3) {
        fprintf(stderr,
                "[ERROR] Lanczos failed for L=%d g=%.17g h=%.17e ret=%d\n",
                b->L, g, h, ret);
        free(evecs);
        return EXIT_FAILURE;
    }
    if (ret == -3) {
        fprintf(stderr,
                "[WARN] Lanczos did not fully converge for L=%d g=%.17g "
                "h=%.17e; writing available Ritz values.\n",
                b->L, g, h);
    }
    if (!residual_warning_printed) {
        fprintf(stderr,
                "[WARN] Lanczos residuals are not exposed by the current API; "
                "writing resid0..resid2 as NaN.\n");
        residual_warning_printed = true;
    }

    for (int k = 0; k < N_EIG_SOLVE; k++) {
        out->evals[k] = evals[k];
        out->resid[k] = NAN;
    }
    out->mz = obs_mz_raw(b, evecs);
    out->abs_mz = obs_psi_bar(b, evecs);
    out->mx = obs_mx(b, evecs);
    out->method_code = 1;
    out->lanczos_ret = ret;

    free(evecs);
    return EXIT_SUCCESS;
}

static int lanczos_max_iter(int L)
{
    if (L <= 16)
        return 200;
    if (L <= 18)
        return 150;
    if (L <= 20)
        return 100;
    return 60;
}

static void print_allocation_failure(const char *where, const Basis *b,
                                     double g, double h)
{
    fprintf(stderr,
            "[ERROR] allocation failed in %s: L=%d dim=%lld g=%.17g h=%.17e\n",
            where, b->L, b->dim, g, h);
}

static double now_seconds(void)
{
    return (double)time(NULL);
}

#ifndef _WIN32
static void handle_signal(int sig)
{
    (void)sig;
    if (g_lock_held && g_active_lock_path[0] != '\0')
        unlink(g_active_lock_path);
    _Exit(EXIT_FAILURE);
}

static int install_signal_handlers(void)
{
    if (signal(SIGINT, handle_signal) == SIG_ERR)
        return EXIT_FAILURE;
    if (signal(SIGTERM, handle_signal) == SIG_ERR)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
#endif
