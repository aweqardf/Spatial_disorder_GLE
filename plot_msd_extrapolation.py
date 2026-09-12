#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gpytorch
import matplotlib
import torch

from synthetic_comparison import (
    fit_joint_model,
    interpolate_grid,
    make_parser as comparison_parser,
    make_synthetic_data,
    select_device,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float32)


def memory_from_truth(truth: dict, temperature: float, dt: float, device: torch.device):
    A = torch.tensor(truth["A"], device=device)
    c = torch.tensor(truth["c"], device=device)
    sym = torch.tensor(truth["sym"], device=device)
    eye = torch.eye(A.shape[0], device=device)
    phi = eye - A * dt
    drive = c * dt
    q = 2.0 * temperature * sym * dt + eye * 1e-7
    return phi, drive, torch.linalg.cholesky(q), c


def memory_from_model(model, dt: float):
    phi, drive, q = model.memory.discrete_matrices(dt)
    return phi, drive, torch.linalg.cholesky(q), model.memory.c_vec


@torch.no_grad()
def force_mean_on_grid(model, x_grid: torch.Tensor, n_replicas: int) -> torch.Tensor:
    model.eval()
    with gpytorch.settings.fast_pred_var():
        xq = x_grid.unsqueeze(0).repeat(n_replicas, 1).unsqueeze(-1)
        return model(xq).mean.detach()


@torch.no_grad()
def simulate_msd(
    force_grid: torch.Tensor,
    x_grid: torch.Tensor,
    memory_terms,
    *,
    temperature: float,
    dt: float,
    n_trajs: int,
    n_steps: int,
    seed: int,
):
    torch.manual_seed(seed)
    n_replicas = force_grid.shape[0]
    phi, drive, q_chol, c = memory_terms
    n_aux = c.numel()
    x = torch.randn(n_replicas, n_trajs, device=x_grid.device) * 0.5
    v = torch.randn(n_replicas, n_trajs, device=x_grid.device) * math.sqrt(temperature)
    z = torch.randn(n_replicas, n_trajs, n_aux, device=x_grid.device) * math.sqrt(temperature)
    x0 = x.clone()
    msd = torch.empty(n_steps, device=x_grid.device)
    times = torch.arange(1, n_steps + 1, device=x_grid.device, dtype=x_grid.dtype) * dt

    for step in range(n_steps):
        force = interpolate_grid(x, x_grid, force_grid)
        memory = torch.sum(z * c.view(1, 1, n_aux), dim=-1)
        acc = force + memory
        x = x + v * dt
        v = v + acc * dt
        z = z.matmul(phi.T) - v.unsqueeze(-1) * drive.view(1, 1, n_aux)
        z = z + torch.randn_like(z).matmul(q_chol.T)
        msd[step] = (x - x0).pow(2).mean()
    return times.detach().cpu(), msd.detach().cpu()


def make_plot(path: Path, curves: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=180)
    styles = {
        "Reference": {"color": "black", "lw": 1.8, "marker": None},
        "SD-GLE": {"color": "#1f77b4", "lw": 1.4, "marker": "o"},
        "GLE without disorder": {"color": "#7f7f7f", "lw": 1.2, "marker": "s"},
    }
    for label, (t, msd) in curves.items():
        style = styles[label]
        if style["marker"] is None:
            ax.plot(t, msd, color=style["color"], lw=style["lw"], label=label)
        else:
            stride = max(1, len(t) // 36)
            ax.plot(t, msd, color=style["color"], lw=style["lw"], alpha=0.85)
            ax.scatter(t[::stride], msd[::stride], facecolors="none", edgecolors=style["color"], marker=style["marker"], s=16, lw=0.8, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("MSD")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = comparison_parser()
    parser.set_defaults(
        outdir=Path("results/msd_extrapolation"),
        n_replicas=6,
        n_trajs=5,
        n_steps=70,
        n_grid=160,
        m_inducing=28,
        epochs_spatial=0,
        epochs_memory=0,
        epochs_joint=120,
        a_modes="spd_skew",
        device="cpu",
        true_os=1.2,
    )
    parser.add_argument("--figure", type=Path, default=Path("results/msd_extrapolation/msd_extrapolation.png"))
    parser.add_argument("--rollout-steps", type=int, default=5000)
    parser.add_argument("--rollout-trajs", type=int, default=128)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    device = select_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    X, V, A_obs, _, _, x_grid, true_force, truth = make_synthetic_data(args, device)
    true_A = torch.tensor(truth["A"], device=device)
    true_c = torch.tensor(truth["c"], device=device)
    row = fit_joint_model("spd_skew", X, V, A_obs, true_A, true_c, args, device)
    model = row["model"]
    fitted_force = force_mean_on_grid(model, x_grid, args.n_replicas)
    zero_force = torch.zeros_like(fitted_force)

    curves = {
        "Reference": simulate_msd(
            true_force,
            x_grid,
            memory_from_truth(truth, args.temperature, args.dt, device),
            temperature=args.temperature,
            dt=args.dt,
            n_trajs=args.rollout_trajs,
            n_steps=args.rollout_steps,
            seed=args.seed + 100,
        ),
        "SD-GLE": simulate_msd(
            fitted_force,
            x_grid,
            memory_from_model(model, args.dt),
            temperature=args.temperature,
            dt=args.dt,
            n_trajs=args.rollout_trajs,
            n_steps=args.rollout_steps,
            seed=args.seed + 100,
        ),
        "GLE without disorder": simulate_msd(
            zero_force,
            x_grid,
            memory_from_model(model, args.dt),
            temperature=args.temperature,
            dt=args.dt,
            n_trajs=args.rollout_trajs,
            n_steps=args.rollout_steps,
            seed=args.seed + 100,
        ),
    }
    make_plot(args.figure, curves)
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
