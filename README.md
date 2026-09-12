# Spatial Disorder Generalized Langevin Equation

This repository provides a stable reference implementation of SD-GLE, a
coarse-grained stochastic dynamics model that combines

- replica-dependent static spatial disorder, represented by a Gaussian random
  force field derived from a random potential, and
- non-Markovian memory, represented by a finite-dimensional auxiliary-variable
  embedding of a generalized Langevin equation.

The implementation is designed for short-trajectory inference: it jointly fits
the spatial force kernel and the memory kernel from trajectory positions,
velocities, and accelerations.

## Repository contents

- `sdgle_core.py`: stable SD-GLE core implementation with a synthetic benchmark.
- `main.py`: backward-compatible entry point that calls `sdgle_core.main()`.
- `requirements.txt`: minimal Python dependencies.
- `LICENSE`: MIT license.

The public repository intentionally excludes large generated data, paper
figures, and local experiment logs.

## Installation

Use Python 3.8 or newer.

```bash
git clone https://github.com/aweqardf/Spatial_disorder_GLE.git
cd Spatial_disorder_GLE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build matching your CUDA runtime
before installing the remaining requirements.

## Quick smoke test

The following command runs a small CPU synthetic benchmark and writes outputs
under `results/smoke`:

```bash
python main.py \
  --outdir results/smoke \
  --device cpu \
  --n-replicas 3 \
  --n-trajs 3 \
  --n-steps 24 \
  --n-grid 64 \
  --m-inducing 12 \
  --epochs-spatial 3 \
  --epochs-memory 3 \
  --epochs-joint 3 \
  --a-modes diag_skew \
  --memory-only-modes diag_skew \
  --kernel-points 80
```

The output directory contains:

- `summary.json`: arguments, true parameters, and metrics.
- `metrics.csv`: tabulated spatial and memory inference errors.
- `joint_rmse_summary.png`: comparison of joint model errors.
- `README.md`: auto-generated run report.

## Standard synthetic run

```bash
python main.py --outdir results/sdgle_formal_results --device auto
```

This compares several auxiliary-memory parameterizations:

- `diag`: diagonal auxiliary dynamics, equivalent to a sum of purely viscous
  exponentials.
- `diag_skew`: diagonal dissipative rates plus skew-symmetric elastic coupling.
- `spd_skew`: full symmetric positive dissipative part plus skew-symmetric
  elastic coupling.
- `free`: unconstrained diagnostic parameterization with a stability penalty.

For production experiments, `spd_skew` is usually the preferred stable model:
it relaxes the diagonal-memory restriction while keeping a positive
dissipative part. The unconstrained `free` model is useful for diagnostics, but
can overfit or become weakly stable.

## Model summary

The latent dynamics are

```text
dx/dt = v
dv/dt = F_i(x) + c^T z
dz/dt = -A z - c v + noise
```

where each replica `i` has its own static random force field `F_i`, while the
memory parameters `A` and `c` are shared across replicas.

The spatial force fields are inferred with a sparse variational Gaussian
process. The inducing values are replica-batched; the initial inducing
locations are shared copies of a common grid. If `--learn-inducing` is enabled,
each replica can move its own inducing locations independently. The spatial
kernel hyperparameters remain shared across replicas.

The Kalman filter marginalizes the auxiliary variables in the likelihood, and
the sparse GP variational KL regularizes the spatial force posterior.

## Important options

```text
--n-replicas              number of independent spatial environments
--n-trajs                 trajectories per replica
--n-steps                 frames per trajectory
--n-aux                   number of auxiliary variables in the memory model
--a-modes                 joint memory parameterizations to compare
--spatial-kernel          rbf, rq, or matern52 potential kernel
--learn-inducing          learn inducing point locations
--spatial-prior-weight    weak log prior weight for spatial kernel scale
--device                  cpu, cuda, or auto
```

The joint spatial GP and memory likelihood has a scale degeneracy: the memory
block can absorb part of the static force and push the spatial kernel to an
effective solution. The default therefore includes a weak log hyperprior on the
spatial kernel around the initialization:

```bash
--spatial-prior-weight 1.0 --spatial-prior-ls 0.8 --spatial-prior-os 0.3
```

For real data, set these prior centers from an external spatial estimate or a
trusted initialization.

## Citation

If this code supports your work, please cite the associated SD-GLE manuscript
and this repository.
