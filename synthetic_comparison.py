#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import gpytorch
import matplotlib
import numpy as np
import torch

from sdgle_core import (
    ForceGP,
    MatrixMemory,
    SDGLE,
    force_covariance,
    force_covariance_rbf,
    make_inducing_grid,
    matrix_memory_kernel,
    parse_float_list,
    skew_from_upper,
    spatial_hyperprior_loss,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float32)


def interpolate_grid(x: torch.Tensor, x_grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    n_rep, n_traj = x.shape
    n_grid = x_grid.numel()
    u = (x - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (n_grid - 1)
    u = u.clamp(0.0, n_grid - 1.0)
    idx0 = torch.floor(u).long().clamp(0, n_grid - 1)
    idx1 = (idx0 + 1).clamp(0, n_grid - 1)
    w = u - idx0.to(u.dtype)
    v0 = values.gather(1, idx0.reshape(n_rep, n_traj))
    v1 = values.gather(1, idx1.reshape(n_rep, n_traj))
    return (1.0 - w) * v0 + w * v1


def true_memory_matrices(args, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rates = torch.tensor(parse_float_list(args.true_rates), device=device)
    c = torch.tensor(parse_float_list(args.true_c), device=device)
    skew_values = torch.tensor(parse_float_list(args.true_skew_upper), device=device)
    n_aux = rates.numel()
    expected = n_aux * (n_aux - 1) // 2
    if c.numel() != n_aux:
        raise ValueError(f"--true-c has {c.numel()} entries, expected {n_aux}")
    if skew_values.numel() != expected:
        raise ValueError(f"--true-skew-upper has {skew_values.numel()} entries, expected {expected}")
    A = torch.diag(rates) + skew_from_upper(skew_values, n_aux)
    sym = torch.diag(rates)
    return A, c, sym


@torch.no_grad()
def make_synthetic_data(args, device: torch.device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x_grid = torch.linspace(args.grid_min, args.grid_max, args.n_grid, device=device)
    dist = x_grid.unsqueeze(0) - x_grid.unsqueeze(1)
    cov = force_covariance_rbf(dist, args.true_ls, args.true_os)
    cov = cov + torch.eye(args.n_grid, device=device, dtype=x_grid.dtype) * 1e-4
    chol = torch.linalg.cholesky(cov)
    force_fields = (chol @ torch.randn(args.n_grid, args.n_replicas, device=device)).T

    A, c, sym = true_memory_matrices(args, device)
    n_aux = A.shape[0]
    eye = torch.eye(n_aux, device=device, dtype=x_grid.dtype)
    phi = eye - A * args.dt
    drive = c * args.dt
    q = 2.0 * args.temperature * sym * args.dt + eye * 1e-7
    q_chol = torch.linalg.cholesky(q)

    curr_x = torch.randn(args.n_replicas, args.n_trajs, device=device) * 0.5
    curr_v = torch.randn(args.n_replicas, args.n_trajs, device=device) * math.sqrt(args.temperature)
    curr_z = torch.randn(args.n_replicas, args.n_trajs, n_aux, device=device) * math.sqrt(args.temperature)

    X = torch.zeros(args.n_replicas, args.n_trajs, args.n_steps, device=device)
    V = torch.zeros_like(X)
    A_obs = torch.zeros_like(X)
    F_at_x = torch.zeros_like(X)
    M_true = torch.zeros_like(X)

    for t in range(args.n_steps):
        f_c = interpolate_grid(curr_x, x_grid, force_fields)
        mem = torch.sum(curr_z * c.view(1, 1, n_aux), dim=-1)
        acc_true = f_c + mem
        X[:, :, t] = curr_x
        V[:, :, t] = curr_v
        A_obs[:, :, t] = acc_true + args.obs_noise_std * torch.randn_like(acc_true)
        F_at_x[:, :, t] = f_c
        M_true[:, :, t] = mem

        curr_x = curr_x + curr_v * args.dt
        curr_v = curr_v + acc_true * args.dt
        noise = torch.randn_like(curr_z).matmul(q_chol.T)
        curr_z = curr_z.matmul(phi.T) - curr_v.unsqueeze(-1) * drive.view(1, 1, n_aux) + noise

    truth = {
        "A": A.detach().cpu().tolist(),
        "c": c.detach().cpu().tolist(),
        "sym": sym.detach().cpu().tolist(),
        "n_aux": n_aux,
    }
    return X, V, A_obs, F_at_x, M_true, x_grid, force_fields, truth


class MemoryFit(torch.nn.Module):
    def __init__(self, n_aux: int, a_mode: str, args):
        super().__init__()
        self.obs_var = float(args.obs_noise_std**2)
        self.memory = MatrixMemory(
            n_aux,
            a_mode,
            temperature=args.temperature,
            init_c_scale=args.init_c_scale,
            init_rate=args.init_rate,
            init_skew=args.init_skew,
            init_rates_values=None,
        )

    def nll(self, residual: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        N, M, T = residual.shape
        n_aux = self.memory.n_aux
        phi, drive, q = self.memory.discrete_matrices(dt)
        c = self.memory.c_vec
        eye = torch.eye(n_aux, device=residual.device, dtype=residual.dtype)
        z_mean = torch.zeros(N, M, n_aux, device=residual.device, dtype=residual.dtype)
        z_cov = eye.expand(N, M, n_aux, n_aux).clone() * self.memory.temperature
        log2pi = math.log(2.0 * math.pi)
        nll = torch.tensor(0.0, device=residual.device, dtype=residual.dtype)

        for t in range(T):
            pred = torch.sum(z_mean * c.view(1, 1, n_aux), dim=-1)
            pc = torch.matmul(z_cov, c.view(1, 1, n_aux, 1)).squeeze(-1)
            S = torch.sum(c.view(1, 1, n_aux) * pc, dim=-1) + self.obs_var
            S = S.clamp_min(1e-8)
            innov = residual[:, :, t] - pred
            nll = nll + 0.5 * (innov.pow(2) / S + torch.log(S) + log2pi).sum()
            gain = pc / S.unsqueeze(-1)
            z_mean = z_mean + gain * innov.unsqueeze(-1)
            ikh = eye.view(1, 1, n_aux, n_aux) - gain.unsqueeze(-1) * c.view(1, 1, 1, n_aux)
            z_cov = torch.matmul(torch.matmul(ikh, z_cov), ikh.transpose(-1, -2))
            z_cov = z_cov + self.obs_var * gain.unsqueeze(-1) * gain.unsqueeze(-2)
            z_cov = 0.5 * (z_cov + z_cov.transpose(-1, -2))
            if t + 1 < T:
                z_mean = z_mean.matmul(phi.T) - velocity[:, :, t + 1].unsqueeze(-1) * drive.view(1, 1, n_aux)
                z_cov = torch.matmul(torch.matmul(phi.view(1, 1, n_aux, n_aux), z_cov), phi.T.view(1, 1, n_aux, n_aux))
                z_cov = z_cov + q.view(1, 1, n_aux, n_aux)
                z_cov = 0.5 * (z_cov + z_cov.transpose(-1, -2))
        return nll


def inducing_from_args(args, device: torch.device) -> torch.Tensor:
    return make_inducing_grid(args.grid_min, args.grid_max, args.m_inducing, args.n_replicas, device)


def kernel_prior(model: ForceGP, args) -> torch.Tensor:
    return spatial_hyperprior_loss(
        model,
        weight=args.spatial_prior_weight,
        lengthscale=args.spatial_prior_ls,
        outputscale=args.spatial_prior_os,
    )


def spatial_kernel_metrics(model: ForceGP, args) -> dict[str, float]:
    device = model.covar_module.outputscale.device
    r_eval = torch.linspace(0.0, args.kernel_r_max, args.kernel_points, device=device)
    ls = model.covar_module.base_kernel.lengthscale.squeeze()
    os = model.covar_module.outputscale.squeeze()
    rq_alpha = float(getattr(args, "rq_alpha", 1.0))
    if hasattr(model.covar_module.base_kernel, "alpha"):
        rq_alpha = float(model.covar_module.base_kernel.alpha.detach().cpu())
    truth = force_covariance(r_eval, args.true_ls, args.true_os, "rbf", rq_alpha)
    pred = force_covariance(r_eval, ls, os, getattr(args, "spatial_kernel", "rbf"), rq_alpha)
    rel = torch.sqrt(torch.mean((pred - truth).pow(2))) / torch.sqrt(torch.mean(truth.pow(2))).clamp_min(1e-12)
    return {
        "lengthscale": float(ls.detach().cpu()),
        "outputscale": float(os.detach().cpu()),
        "spatial_kernel_rel_rmse": float(rel.detach().cpu()),
    }


def memory_metrics(memory: MatrixMemory, true_A: torch.Tensor, true_c: torch.Tensor, args) -> dict[str, float]:
    device = true_A.device
    t_eval = torch.linspace(0.0, args.kernel_t_max, args.kernel_points, device=device)
    pred = memory.kernel(t_eval)
    truth = matrix_memory_kernel(true_A, true_c, t_eval)
    rel = torch.sqrt(torch.mean((pred - truth).pow(2))) / torch.sqrt(torch.mean(truth.pow(2))).clamp_min(1e-12)
    return {"memory_kernel_rel_rmse": float(rel.detach().cpu()), **memory.summary()}


def fit_force_given_memory(X, target_force, args, device):
    model = ForceGP(
        inducing_from_args(args, device),
        args.n_replicas,
        args.init_ls,
        args.init_os,
        args.learn_inducing,
        args.spatial_kernel,
        args.rq_alpha,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_spatial)
    N, M, T = X.shape
    x_in = X.reshape(N, M * T).unsqueeze(-1)
    target = target_force.reshape(N, M * T)
    start = time.time()
    for _ in range(args.epochs_spatial):
        opt.zero_grad()
        dist = model(x_in)
        var = dist.variance + args.obs_noise_std**2
        nll = 0.5 * ((target - dist.mean).pow(2) / var + torch.log(var) + math.log(2.0 * math.pi)).sum()
        loss = (nll + args.beta_kl * model.variational_strategy.kl_divergence().sum()) / X.numel() + kernel_prior(model, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
    return {"name": "force_known_memory", "loss": float(loss.detach().cpu()), "seconds": time.time() - start, **spatial_kernel_metrics(model, args)}


def fit_memory_given_force(residual, V, true_A, true_c, args, device):
    best = None
    for a_mode in args.memory_modes.split(","):
        model = MemoryFit(args.n_aux, a_mode.strip(), args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_memory)
        start = time.time()
        for _ in range(args.epochs_memory):
            opt.zero_grad()
            penalty = args.stability_penalty * model.memory.stability_penalty(args.stability_margin)
            loss = model.nll(residual, V, args.dt) / residual.numel() + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        row = {"name": f"memory_known_force_{a_mode.strip()}", "loss": float(loss.detach().cpu()), "seconds": time.time() - start}
        row.update(memory_metrics(model.memory, true_A, true_c, args))
        if best is None or row["memory_kernel_rel_rmse"] < best["memory_kernel_rel_rmse"]:
            best = row
    return best


def fit_joint_model(a_mode, X, V, A_obs, true_A, true_c, args, device):
    best = None
    for start_id in range(args.starts):
        torch.manual_seed(args.seed * 1009 + start_id * 97 + len(a_mode))
        model = SDGLE(inducing_from_args(args, device), args.n_replicas, args.n_aux, a_mode, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_joint)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_joint, 1), eta_min=args.lr_joint * 0.2)
        start = time.time()
        last = None
        for _ in range(args.epochs_joint):
            opt.zero_grad()
            ll = model.kalman_log_likelihood(X, V, A_obs, args.dt)
            kl = model.variational_strategy.kl_divergence().sum()
            penalty = args.stability_penalty * model.memory.stability_penalty(args.stability_margin)
            loss = -(ll - args.beta_kl * kl) / X.numel() + penalty + kernel_prior(model, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            last = float(loss.detach().cpu())
        row = {
            "name": f"joint_{a_mode}",
            "loss": last,
            "seconds": time.time() - start,
            "model": model,
            **spatial_kernel_metrics(model, args),
            **memory_metrics(model.memory, true_A, true_c, args),
        }
        score = row["spatial_kernel_rel_rmse"] + row["memory_kernel_rel_rmse"]
        if best is None or score < best[0]:
            best = (score, row)
    assert best is not None
    return best[1]


def write_csv(path: Path, rows: list[dict]):
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: value for key, value in row.items() if key != "model"})
    keys = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scalar_rows)


def plot_comparison(path: Path, rows: list[dict]):
    joint = [r for r in rows if r["name"].startswith("joint")]
    labels = [r["name"].replace("joint_", "") for r in joint]
    x = np.arange(len(joint))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(x, [r["spatial_kernel_rel_rmse"] for r in joint], marker="o", label="spatial")
    ax.plot(x, [r["memory_kernel_rel_rmse"] for r in joint], marker="s", label="memory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("relative RMSE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/synthetic_comparison"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--n-replicas", type=int, default=6)
    parser.add_argument("--n-trajs", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=70)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--m-inducing", type=int, default=28)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--grid-min", type=float, default=-4.0)
    parser.add_argument("--grid-max", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.16)
    parser.add_argument("--true-ls", type=float, default=0.8)
    parser.add_argument("--true-os", type=float, default=0.3)
    parser.add_argument("--true-rates", default="0.18,0.65,1.55")
    parser.add_argument("--true-skew-upper", default="-0.85,0.25,-1.15")
    parser.add_argument("--true-c", default="0.95,-0.60,0.40")
    parser.add_argument("--obs-noise-std", type=float, default=0.08)
    parser.add_argument("--n-aux", type=int, default=3)
    parser.add_argument("--a-modes", default="diag,diag_skew,spd_skew")
    parser.add_argument("--memory-modes", default="diag_skew,spd_skew")
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--epochs-spatial", type=int, default=150)
    parser.add_argument("--epochs-memory", type=int, default=150)
    parser.add_argument("--epochs-joint", type=int, default=180)
    parser.add_argument("--lr-spatial", type=float, default=1e-2)
    parser.add_argument("--lr-memory", type=float, default=1.5e-2)
    parser.add_argument("--lr-joint", type=float, default=1e-2)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--learn-inducing", action="store_true", default=False)
    parser.add_argument("--init-ls", type=float, default=0.8)
    parser.add_argument("--init-os", type=float, default=0.3)
    parser.add_argument("--spatial-kernel", choices=["rbf", "rq", "matern52"], default="rbf")
    parser.add_argument("--rq-alpha", type=float, default=1.0)
    parser.add_argument("--spatial-prior-ls", type=float, default=0.8)
    parser.add_argument("--spatial-prior-os", type=float, default=0.3)
    parser.add_argument("--spatial-prior-weight", type=float, default=1.0)
    parser.add_argument("--init-c-scale", type=float, default=1.2)
    parser.add_argument("--init-rate", type=float, default=0.7)
    parser.add_argument("--init-skew", type=float, default=-0.1)
    parser.add_argument("--stability-penalty", type=float, default=5.0)
    parser.add_argument("--stability-margin", type=float, default=0.03)
    parser.add_argument("--kernel-t-max", type=float, default=8.0)
    parser.add_argument("--kernel-r-max", type=float, default=4.0)
    parser.add_argument("--kernel-points", type=int, default=300)
    return parser


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def run_comparison(args, device: torch.device):
    X, V, A_obs, F_at_x, M_true, x_grid, force_fields, truth = make_synthetic_data(args, device)
    true_A = torch.tensor(truth["A"], device=device)
    true_c = torch.tensor(truth["c"], device=device)

    rows = [
        fit_force_given_memory(X, A_obs - M_true, args, device),
        fit_memory_given_force(A_obs - F_at_x, V, true_A, true_c, args, device),
    ]
    for a_mode in [item.strip() for item in args.a_modes.split(",") if item.strip()]:
        row = fit_joint_model(a_mode, X, V, A_obs, true_A, true_c, args, device)
        rows.append(row)
        public_row = {key: value for key, value in row.items() if key != "model"}
        print(json.dumps(public_row, sort_keys=True), flush=True)
    return rows, truth


def main():
    parser = make_parser()
    args = parser.parse_args()
    device = select_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, truth = run_comparison(args, device)
    public_rows = [{key: value for key, value in row.items() if key != "model"} for row in rows]
    (args.outdir / "summary.json").write_text(
        json.dumps({"device": str(device), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "truth": truth, "rows": public_rows}, indent=2),
        encoding="utf-8",
    )
    write_csv(args.outdir / "metrics.csv", rows)
    plot_comparison(args.outdir / "kernel_error_summary.png", rows)
    print(json.dumps({"saved": str(args.outdir), "n_rows": len(rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
