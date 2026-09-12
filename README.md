# Spatial Disorder Generalized Langevin Equation

This repository contains the core code for SD-GLE, a one-dimensional
coarse-grained model with replica-dependent static disorder and non-Markovian
memory. The model represents the static force field with a sparse Gaussian
process and treats the memory term with auxiliary variables marginalized by a
Kalman filter.

Large datasets, generated figures, and local experiment logs are not included.

## Installation

Use Python 3.8 or newer.

```bash
git clone https://github.com/aweqardf/Spatial_disorder_GLE.git
cd Spatial_disorder_GLE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build first if you want GPU acceleration.

## Quick start

Run a small synthetic comparison:

```bash
python synthetic_comparison.py \
  --outdir results/synthetic_comparison \
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
  --memory-modes diag_skew \
  --kernel-points 80
```

Run the MSD extrapolation plot:

```bash
python plot_msd_extrapolation.py --device cpu
```

The first command writes numerical metrics to `results/synthetic_comparison`.
The second command writes a single MSD comparison figure with a reference
trajectory, the fitted SD-GLE model, and a GLE baseline without spatial
disorder.

## Notes

The memory parameters are shared across replicas. The inducing values are
replica-batched. The inducing locations start from the same grid for every
replica; with `--learn-inducing`, each replica can move its own locations
independently. Spatial kernel hyperparameters are shared.

For matrix-memory inference, `spd_skew` is the recommended parameterization: it
allows elastic coupling while keeping a positive dissipative part. The
unconstrained `free` mode is useful for diagnostics, but can overfit.

The joint spatial/memory likelihood has a scale degeneracy, so the default
uses a weak log prior on the spatial lengthscale and outputscale. For real data,
set the prior centers from a trusted spatial estimate or a conservative
initialization.

## Citation

If you use this code, please cite the associated SD-GLE manuscript and this
repository.
