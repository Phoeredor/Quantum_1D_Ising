# Makefile — 1D Quantum Ising Chain: Exact Diagonalization
# =============================================================
# Targets:
#   make static    — ising_static   (full ED, dense LAPACK)
#   make quench    — ising_quench   (real-time dynamics)
#   make lanczos   — ising_lanczos  (Lanczos for large L)
#   make spectral  — ising_spectral (level-spacing statistics)
#   make hfield    — ising_hfield   (longitudinal-field generator)
#   make ghsurface — ising_gh_surface (two-parameter gh surface worker)
#   make chizfd    — ising_chiz_fd (finite-difference susceptibility)
#   make all       — all active executables
#   make clean     — remove objects and executables
#   make dep       — print dependency info

CC     = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c99 \
         -MMD -MP -Iinclude
LDFLAGS = -llapacke -llapack -lopenblas -lm

# Source / object directories
SRC = src
INC = include
OBJ = obj
HNULL_DEPFILES = $(OBJ)/main_static_h_null.d $(OBJ)/main_lanczos_h_null.d $(OBJ)/main_chiz_fd_h_null.d
DEPFILES = $(filter-out $(OBJ)/main_static.d $(OBJ)/main_lanczos.d $(OBJ)/main_chiz_fd.d,$(wildcard $(OBJ)/*.d))
DEPFILES += $(wildcard $(HNULL_DEPFILES))

# Quench full ED is meant for the L<=12 production driver by default.
# Override with: make quench QUENCH_MAX_DIM=8192
QUENCH_MAX_DIM ?= 4096

# ---- Object groups ----

# Full ED: all modules
STATIC_OBJ = $(OBJ)/basis.o $(OBJ)/basis_sector.o $(OBJ)/hamiltonian.o \
             $(OBJ)/hamiltonian_sector.o $(OBJ)/diag.o \
             $(OBJ)/observables.o $(OBJ)/apply_H.o $(OBJ)/lanczos.o

# Lanczos: basis + hamiltonian + observables (no diag.o) + apply_H + lanczos
LANCZOS_OBJ = $(OBJ)/basis.o $(OBJ)/basis_sector.o $(OBJ)/hamiltonian.o \
              $(OBJ)/hamiltonian_sector.o $(OBJ)/observables.o \
              $(OBJ)/apply_H.o $(OBJ)/lanczos.o

# Spectral: ED-only worker plus re-entrant PCG32 disorder RNG
SPECTRAL_OBJ = $(OBJ)/pcg32.o

# Quench: same as full ED (uses dsyevd for initial state)
QUENCH_OBJ = $(STATIC_OBJ)

# GH surface: low-energy gaps plus ground-state observables.
GHSURFACE_OBJ = $(OBJ)/basis.o $(OBJ)/hamiltonian.o $(OBJ)/diag.o \
                $(OBJ)/observables.o $(OBJ)/apply_H.o $(OBJ)/lanczos.o

# ---- Executables ----
all: static quench lanczos spectral hfield ghsurface chizfd

static:  $(OBJ)/main_static.o  $(STATIC_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_static

quench:  $(OBJ)/main_quench.o  $(QUENCH_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_quench

lanczos: $(OBJ)/main_lanczos.o $(LANCZOS_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_lanczos

spectral: $(OBJ)/main_spectral_spectral.o $(SPECTRAL_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_spectral

hfield: $(OBJ)/main_hfield.o $(STATIC_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_hfield

ghsurface: $(OBJ)/gh_surface.o $(GHSURFACE_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_gh_surface

chizfd: $(OBJ)/main_chiz_fd.o $(STATIC_OBJ)
	$(CC) $^ $(LDFLAGS) -o ising_chiz_fd

# ---- h=0 production compile rules ----
$(OBJ):
	mkdir -p $@

$(OBJ)/main_static.o: $(SRC)/h_null/main_static.c | $(OBJ)
	$(CC) $(CFLAGS) -MF $(OBJ)/main_static_h_null.d -c $< -o $@

$(OBJ)/main_lanczos.o: $(SRC)/h_null/main_lanczos.c | $(OBJ)
	$(CC) $(CFLAGS) -MF $(OBJ)/main_lanczos_h_null.d -c $< -o $@

$(OBJ)/main_chiz_fd.o: $(SRC)/h_null/main_chiz_fd.c | $(OBJ)
	$(CC) $(CFLAGS) -MF $(OBJ)/main_chiz_fd_h_null.d -c $< -o $@

$(OBJ)/gh_surface.o: $(SRC)/h_field/gh_surface/gh_surface.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/main_spectral_spectral.o: $(SRC)/spectral/main_spectral.c | $(OBJ)
	$(CC) $(CFLAGS) -MF $(OBJ)/main_spectral_spectral.d -c $< -o $@

$(OBJ)/main_quench.o: $(SRC)/main_quench.c | $(OBJ)
	$(CC) $(CFLAGS) -DMAX_DIM=$(QUENCH_MAX_DIM) -c $< -o $@

# ---- Generic compile rule ----
$(OBJ)/%.o: $(SRC)/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Auto-generated header dependencies ----
-include $(DEPFILES)

# ---- Housekeeping ----
clean:
	rm -rf $(OBJ)
	rm -f ising_static ising_quench ising_lanczos
	rm -f ising_spectral ising_hfield ising_gh_surface ising_chiz_fd

dep:
	@echo "Runtime dependencies: LAPACKE, LAPACK, OpenBLAS"
	@echo "Ubuntu/Debian:  sudo apt install liblapacke-dev libopenblas-dev gcc make"
	@echo "Fedora/RHEL:    sudo dnf install lapack-devel openblas-devel gcc make"
	@echo "macOS:          brew install openblas lapack"

.PHONY: all static quench lanczos spectral hfield ghsurface chizfd clean dep
