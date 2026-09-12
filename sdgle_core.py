#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


torch.set_default_dtype(torch.float32)


class RBFForceKernel(gpytorch.kernels.Kernel):
    """Covariance of F(x)=-dU/dx when U has an RBF covariance."""

    is_stationary = True
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        x1_scaled = x1.div(self.lengthscale)
        x2_scaled = x2.div(self.lengthscale)
        l_sq = self.lengthscale.squeeze(-1).pow(2)
        if diag:
            return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype) / l_sq
        diff = x1_scaled.unsqueeze(-2) - x2_scaled.unsqueeze(-3)
        sq_dist = diff.pow(2).squeeze(-1)
        exp_term = torch.exp(-0.5 * sq_dist)
        return (1.0 - sq_dist) * exp_term / l_sq.unsqueeze(-1)


class RQPotentialForceKernel(gpytorch.kernels.Kernel):
    """Covariance of F(x)=-dU/dx when U has a rational-quadratic covariance."""

    is_stationary = True
    has_lengthscale = True

    def __init__(self, *, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_alpha", torch.nn.Parameter(torch.tensor(float(alpha)).log()))
        self.register_constraint("raw_alpha", gpytorch.constraints.Positive())

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        alpha = self.alpha.clamp_min(1e-6)
        l_sq = self.lengthscale.squeeze(-1).pow(2)
        if diag:
            return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype) / l_sq

        x1_scaled = x1.div(self.lengthscale)
        x2_scaled = x2.div(self.lengthscale)
        diff = x1_scaled.unsqueeze(-2) - x2_scaled.unsqueeze(-3)
        y2 = diff.pow(2).squeeze(-1)
        s = 1.0 + y2 / (2.0 * alpha)
        cov = s.pow(-alpha - 1.0) - ((alpha + 1.0) / alpha) * y2 * s.pow(-alpha - 2.0)
        return cov / l_sq.unsqueeze(-1)


class Matern52PotentialForceKernel(gpytorch.kernels.Kernel):
    """Covariance of F(x)=-dU/dx when U has a Matern-5/2 covariance."""

    is_stationary = True
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        l_sq = self.lengthscale.squeeze(-1).pow(2)
        if diag:
            return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype) * (5.0 / 3.0) / l_sq

        x1_scaled = x1.div(self.lengthscale)
        x2_scaled = x2.div(self.lengthscale)
        y = torch.sqrt(torch.tensor(5.0, device=x1.device, dtype=x1.dtype)) * torch.abs(
            x1_scaled.unsqueeze(-2) - x2_scaled.unsqueeze(-3)
        ).squeeze(-1)
        cov = (5.0 / 3.0) * torch.exp(-y) * (1.0 + y - y.pow(2))
        return cov / l_sq.unsqueeze(-1)


def make_force_kernel(spatial_kernel: str, rq_alpha: float) -> gpytorch.kernels.Kernel:
    if spatial_kernel == "rbf":
        return RBFForceKernel()
    if spatial_kernel == "rq":
        return RQPotentialForceKernel(alpha=rq_alpha)
    if spatial_kernel == "matern52":
        return Matern52PotentialForceKernel()
    raise ValueError(f"unknown spatial kernel: {spatial_kernel}")


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def inv_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-8)
    return torch.log(torch.expm1(value))


def positive(raw: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + floor


def skew_from_upper(values: torch.Tensor, n_aux: int) -> torch.Tensor:
    skew = torch.zeros(n_aux, n_aux, device=values.device, dtype=values.dtype)
    cursor = 0
    for i in range(n_aux):
        for j in range(i + 1, n_aux):
            skew[i, j] = values[cursor]
            skew[j, i] = -values[cursor]
            cursor += 1
    return skew


def tril_from_raw(raw: torch.Tensor, n_aux: int, diag_floor: float) -> torch.Tensor:
    L = torch.zeros(n_aux, n_aux, device=raw.device, dtype=raw.dtype)
    cursor = 0
    for i in range(n_aux):
        for j in range(i + 1):
            value = raw[cursor]
            L[i, j] = positive(value, diag_floor) if i == j else value
            cursor += 1
    return L


def force_covariance(dist: torch.Tensor, lengthscale: float, outputscale: float) -> torch.Tensor:
    sq_scaled = dist.pow(2) / (lengthscale * lengthscale)
    return (outputscale / (lengthscale * lengthscale)) * (1.0 - sq_scaled) * torch.exp(-0.5 * sq_scaled)


def force_covariance_by_kernel(
    dist: torch.Tensor,
    lengthscale: torch.Tensor | float,
    outputscale: torch.Tensor | float,
    spatial_kernel: str,
    rq_alpha: torch.Tensor | float,
) -> torch.Tensor:
    lengthscale = torch.as_tensor(lengthscale, device=dist.device, dtype=dist.dtype).clamp_min(1e-8)
    outputscale = torch.as_tensor(outputscale, device=dist.device, dtype=dist.dtype)
    if spatial_kernel == "rbf":
        y = dist / lengthscale
        return outputscale * (1.0 - y.pow(2)) * torch.exp(-0.5 * y.pow(2)) / lengthscale.pow(2)
    if spatial_kernel == "rq":
        alpha = torch.as_tensor(rq_alpha, device=dist.device, dtype=dist.dtype).clamp_min(1e-6)
        y2 = (dist / lengthscale).pow(2)
        s = 1.0 + y2 / (2.0 * alpha)
        return outputscale * (
            s.pow(-alpha - 1.0) - ((alpha + 1.0) / alpha) * y2 * s.pow(-alpha - 2.0)
        ) / lengthscale.pow(2)
    if spatial_kernel == "matern52":
        y = math.sqrt(5.0) * torch.abs(dist) / lengthscale
        return outputscale * (5.0 / (3.0 * lengthscale.pow(2))) * torch.exp(-y) * (1.0 + y - y.pow(2))
    raise ValueError(f"unknown spatial kernel: {spatial_kernel}")


def parse_optional_rates(text: str | None, n_aux: int) -> list[float] | None:
    if not text:
        return None
    values = parse_float_list(text)
    if len(values) != n_aux:
        raise ValueError(f"--init-rates expected {n_aux} comma-separated values, got {len(values)}")
    return values


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


def matrix_memory_kernel(A: torch.Tensor, c: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
    vals = []
    for t in t_eval:
        vals.append(c @ torch.matrix_exp(-A * t) @ c)
    return torch.stack(vals)


def true_matrix(args, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rates = torch.tensor(parse_float_list(args.true_rates), device=device)
    c = torch.tensor(parse_float_list(args.true_c), device=device)
    n_aux = rates.numel()
    skew_values = torch.tensor(parse_float_list(args.true_skew_upper), device=device)
    expected = n_aux * (n_aux - 1) // 2
    if c.numel() != n_aux:
        raise ValueError(f"--true-c has {c.numel()} entries, expected {n_aux}")
    if skew_values.numel() != expected:
        raise ValueError(f"--true-skew-upper has {skew_values.numel()} entries, expected {expected}")
    A = torch.diag(rates) + skew_from_upper(skew_values, n_aux)
    sym = torch.diag(rates)
    return A, c, sym


@torch.no_grad()
def generate_data(args, device: torch.device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x_grid = torch.linspace(args.grid_min, args.grid_max, args.n_grid, device=device)
    dist = x_grid.unsqueeze(0) - x_grid.unsqueeze(1)
    cov = force_covariance(dist, args.true_ls, args.true_os)
    cov = cov + torch.eye(args.n_grid, device=device, dtype=x_grid.dtype) * 1e-4
    chol = torch.linalg.cholesky(cov)
    force_fields = (chol @ torch.randn(args.n_grid, args.n_replicas, device=device)).T

    A, c, sym = true_matrix(args, device)
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
        acc_obs = acc_true + args.obs_noise_std * torch.randn_like(acc_true)
        X[:, :, t] = curr_x
        V[:, :, t] = curr_v
        A_obs[:, :, t] = acc_obs
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
        "rates": torch.diag(sym).detach().cpu().tolist(),
        "n_aux": n_aux,
    }
    return X, V, A_obs, F_at_x, M_true, x_grid, force_fields, truth


class MatrixMemoryCore(torch.nn.Module):
    def __init__(
        self,
        n_aux: int,
        a_mode: str,
        *,
        temperature: float,
        init_c_scale: float,
        init_rate: float,
        init_skew: float,
        init_rates_values: list[float] | None = None,
        init_free_A: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_aux = int(n_aux)
        self.a_mode = a_mode
        self.temperature = float(temperature)
        if init_rates_values is None:
            init_rates = torch.linspace(init_rate * 0.6, init_rate * 1.6, n_aux).clamp_min(0.03)
        else:
            init_rates = torch.as_tensor(init_rates_values, dtype=torch.float32).reshape(-1)
            if init_rates.numel() != n_aux:
                raise ValueError(f"expected {n_aux} init rates, got {init_rates.numel()}")
            init_rates = init_rates.clamp_min(0.03)
        self.raw_rates = torch.nn.Parameter(inv_softplus(init_rates))
        n_skew = n_aux * (n_aux - 1) // 2
        self.raw_skew = torch.nn.Parameter(torch.ones(n_skew) * init_skew)
        signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n_aux)])
        self.c = torch.nn.Parameter(signs * init_c_scale / math.sqrt(n_aux))

        n_tril = n_aux * (n_aux + 1) // 2
        self.raw_spd_tril = torch.nn.Parameter(torch.zeros(n_tril))
        cursor = 0
        for i in range(n_aux):
            for j in range(i + 1):
                if i == j:
                    self.raw_spd_tril.data[cursor] = inv_softplus(torch.tensor(math.sqrt(init_rates[i].item())))
                cursor += 1

        if init_free_A is None:
            init_free_A = torch.diag(init_rates)
        self.raw_A_free = torch.nn.Parameter(init_free_A.clone().detach().float())
        self.raw_B_tril = torch.nn.Parameter(torch.zeros(n_tril))
        cursor = 0
        for i in range(n_aux):
            for j in range(i + 1):
                if i == j:
                    self.raw_B_tril.data[cursor] = inv_softplus(torch.tensor(0.35))
                cursor += 1

    @property
    def rates(self) -> torch.Tensor:
        return positive(self.raw_rates, 1e-4)

    @property
    def skew(self) -> torch.Tensor:
        return skew_from_upper(self.raw_skew, self.n_aux)

    @property
    def symmetric_part(self) -> torch.Tensor:
        eye = torch.eye(self.n_aux, device=self.c.device, dtype=self.c.dtype)
        if self.a_mode in {"diag", "diag_skew"}:
            return torch.diag(self.rates)
        if self.a_mode == "spd_skew":
            L = tril_from_raw(self.raw_spd_tril, self.n_aux, 1e-3)
            return L @ L.T + 1e-4 * eye
        if self.a_mode == "free":
            B = tril_from_raw(self.raw_B_tril, self.n_aux, 1e-4)
            return B @ B.T
        raise ValueError(self.a_mode)

    @property
    def A_matrix(self) -> torch.Tensor:
        if self.a_mode == "diag":
            return torch.diag(self.rates)
        if self.a_mode == "diag_skew":
            return torch.diag(self.rates) + self.skew
        if self.a_mode == "spd_skew":
            return self.symmetric_part + self.skew
        if self.a_mode == "free":
            return self.raw_A_free
        raise ValueError(self.a_mode)

    @property
    def c_vec(self) -> torch.Tensor:
        return self.c

    def discrete_matrices(self, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.A_matrix
        eye = torch.eye(self.n_aux, device=A.device, dtype=A.dtype)
        phi = eye - A * dt
        drive = self.c_vec * dt
        if self.a_mode == "free":
            B = tril_from_raw(self.raw_B_tril, self.n_aux, 1e-4)
            q = B @ B.T * dt + eye * 1e-7
        else:
            q = 2.0 * self.temperature * self.symmetric_part * dt + eye * 1e-7
        return phi, drive, q

    def stability_penalty(self, margin: float) -> torch.Tensor:
        eig = torch.linalg.eigvals(self.A_matrix)
        return torch.relu(margin - eig.real).pow(2).sum()

    def kernel(self, t_eval: torch.Tensor) -> torch.Tensor:
        return matrix_memory_kernel(self.A_matrix, self.c_vec, t_eval)

    def summary(self) -> dict[str, object]:
        A = self.A_matrix.detach().cpu()
        c = self.c_vec.detach().cpu()
        sym = self.symmetric_part.detach().cpu()
        eig = torch.linalg.eigvals(A).detach().cpu()
        return {
            "A": A.tolist(),
            "sym": sym.tolist(),
            "c": c.tolist(),
            "eig_real": eig.real.tolist(),
            "eig_imag": eig.imag.tolist(),
            "K0": float(torch.sum(c * c)),
        }


class ForceSVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        n_replicas: int,
        init_ls: float,
        init_os: float,
        learn_inducing: bool,
        spatial_kernel: str = "rbf",
        rq_alpha: float = 1.0,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([n_replicas]),
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_replicas]))
        self.spatial_kernel = spatial_kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(make_force_kernel(spatial_kernel, rq_alpha))
        self.covar_module.base_kernel.lengthscale = init_ls
        self.covar_module.outputscale = init_os

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class JointSDGLE(ForceSVGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        n_replicas: int,
        n_aux: int,
        a_mode: str,
        args,
        init_free_A: torch.Tensor | None = None,
    ):
        spatial_kernel = getattr(args, "spatial_kernel", "rbf")
        rq_alpha = float(getattr(args, "rq_alpha", 1.0))
        init_rates_values = parse_optional_rates(getattr(args, "init_rates", None), n_aux)
        super().__init__(
            inducing_points,
            n_replicas,
            args.init_ls,
            args.init_os,
            args.learn_inducing,
            spatial_kernel,
            rq_alpha,
        )
        self.obs_var = float(args.obs_noise_std**2)
        self.memory = MatrixMemoryCore(
            n_aux,
            a_mode,
            temperature=args.temperature,
            init_c_scale=args.init_c_scale,
            init_rate=args.init_rate,
            init_skew=args.init_skew,
            init_rates_values=init_rates_values,
            init_free_A=init_free_A,
        )

    def kalman_filter_likelihood_batch(self, x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, dt: float) -> torch.Tensor:
        N, M, T = x.shape
        n_aux = self.memory.n_aux
        x_in = x.reshape(N, M * T).unsqueeze(-1)
        gp_dist = self(x_in)
        mu_f = gp_dist.mean.reshape(N, M, T)
        var_f = gp_dist.variance.reshape(N, M, T)

        phi, drive, q = self.memory.discrete_matrices(dt)
        c = self.memory.c_vec
        eye = torch.eye(n_aux, device=x.device, dtype=x.dtype)
        z_mean = torch.zeros(N, M, n_aux, device=x.device, dtype=x.dtype)
        z_cov = eye.expand(N, M, n_aux, n_aux).clone() * self.memory.temperature
        log2pi = math.log(2.0 * math.pi)
        ll = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for t in range(T):
            y_obs = a[:, :, t] - mu_f[:, :, t]
            pc = torch.matmul(z_cov, c.view(1, 1, n_aux, 1)).squeeze(-1)
            S = torch.sum(c.view(1, 1, n_aux) * pc, dim=-1) + var_f[:, :, t] + self.obs_var
            S = S.clamp_min(1e-8)
            hz = torch.sum(z_mean * c.view(1, 1, n_aux), dim=-1)
            innov = y_obs - hz
            ll = ll + (-0.5 * (innov.pow(2) / S + torch.log(S) + log2pi)).sum()

            gain = pc / S.unsqueeze(-1)
            z_mean = z_mean + gain * innov.unsqueeze(-1)
            ikh = eye.view(1, 1, n_aux, n_aux) - gain.unsqueeze(-1) * c.view(1, 1, 1, n_aux)
            z_cov = torch.matmul(torch.matmul(ikh, z_cov), ikh.transpose(-1, -2))
            z_cov = z_cov + self.obs_var * gain.unsqueeze(-1) * gain.unsqueeze(-2)
            z_cov = 0.5 * (z_cov + z_cov.transpose(-1, -2))

            if t + 1 < T:
                z_mean = z_mean.matmul(phi.T) - v[:, :, t + 1].unsqueeze(-1) * drive.view(1, 1, n_aux)
                z_cov = torch.matmul(torch.matmul(phi.view(1, 1, n_aux, n_aux), z_cov), phi.T.view(1, 1, n_aux, n_aux))
                z_cov = z_cov + q.view(1, 1, n_aux, n_aux)
                z_cov = 0.5 * (z_cov + z_cov.transpose(-1, -2))
        return ll


class MemoryOnlyModel(torch.nn.Module):
    def __init__(self, n_aux: int, a_mode: str, args, init_free_A: torch.Tensor | None = None):
        super().__init__()
        self.obs_var = float(args.obs_noise_std**2)
        init_rates_values = parse_optional_rates(getattr(args, "init_rates", None), n_aux)
        self.memory = MatrixMemoryCore(
            n_aux,
            a_mode,
            temperature=args.temperature,
            init_c_scale=args.init_c_scale,
            init_rate=args.init_rate,
            init_skew=args.init_skew,
            init_rates_values=init_rates_values,
            init_free_A=init_free_A,
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


def make_inducing(args, device: torch.device) -> torch.Tensor:
    z_base = torch.linspace(args.grid_min, args.grid_max, args.m_inducing, device=device).unsqueeze(-1)
    return z_base.unsqueeze(0).repeat(args.n_replicas, 1, 1).contiguous()


def spatial_kernel_metrics(model: ForceSVGP, args) -> dict[str, float]:
    device = model.covar_module.outputscale.device
    r_eval = torch.linspace(0.0, args.kernel_r_max, args.kernel_points, device=device)
    ls = model.covar_module.base_kernel.lengthscale.squeeze()
    os = model.covar_module.outputscale.squeeze()
    spatial_kernel = getattr(args, "spatial_kernel", "rbf")
    rq_alpha = float(getattr(args, "rq_alpha", 1.0))
    if hasattr(model.covar_module.base_kernel, "alpha"):
        rq_alpha = float(model.covar_module.base_kernel.alpha.detach().cpu())
    f_true = force_covariance(r_eval, args.true_ls, args.true_os)
    f_pred = force_covariance_by_kernel(r_eval, ls, os, spatial_kernel, rq_alpha)
    rmse = torch.sqrt(torch.mean((f_pred - f_true).pow(2)))
    denom = torch.sqrt(torch.mean(f_true.pow(2))).clamp_min(1e-12)
    return {
        "lengthscale": float(ls.detach().cpu()),
        "outputscale": float(os.detach().cpu()),
        "spatial_force_kernel_rel_rmse": float((rmse / denom).detach().cpu()),
        "lengthscale_rel_err": abs(float(ls.detach().cpu()) / args.true_ls - 1.0),
        "outputscale_rel_err": abs(float(os.detach().cpu()) / args.true_os - 1.0),
    }


def spatial_hyperprior_loss(model: ForceSVGP, args) -> torch.Tensor:
    if args.spatial_prior_weight <= 0.0:
        return torch.tensor(0.0, device=model.covar_module.outputscale.device)
    ls = model.covar_module.base_kernel.lengthscale.squeeze()
    os = model.covar_module.outputscale.squeeze()
    prior = (torch.log(ls / args.spatial_prior_ls).pow(2) + torch.log(os / args.spatial_prior_os).pow(2))
    return args.spatial_prior_weight * prior


def force_field_metrics(model: ForceSVGP, x_grid: torch.Tensor, force_fields: torch.Tensor) -> dict[str, float]:
    N = force_fields.shape[0]
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        xq = x_grid.unsqueeze(0).repeat(N, 1).unsqueeze(-1)
        pred = model(xq)
        err = pred.mean - force_fields
        rmse = torch.sqrt(torch.mean(err.pow(2)))
        denom = torch.sqrt(torch.mean(force_fields.pow(2))).clamp_min(1e-12)
    return {"force_field_nrmse": float((rmse / denom).detach().cpu())}


def memory_metrics(memory: MatrixMemoryCore, true_A: torch.Tensor, true_c: torch.Tensor, args) -> dict[str, float]:
    device = true_A.device
    t_eval = torch.linspace(0.0, args.kernel_t_max, args.kernel_points, device=device)
    pred = memory.kernel(t_eval)
    truth = matrix_memory_kernel(true_A, true_c, t_eval)
    rmse = torch.sqrt(torch.mean((pred - truth).pow(2)))
    denom = torch.sqrt(torch.mean(truth.pow(2))).clamp_min(1e-12)
    max_denom = torch.max(torch.abs(truth)).clamp_min(1e-12)
    summary = memory.summary()
    return {
        "memory_kernel_rel_rmse": float((rmse / denom).detach().cpu()),
        "memory_kernel_max_rel_err": float((torch.max(torch.abs(pred - truth)) / max_denom).detach().cpu()),
        "memory_K0": float(summary["K0"]),
        "A_json": json.dumps(summary["A"]),
        "c_json": json.dumps(summary["c"]),
        "eig_real_json": json.dumps(summary["eig_real"]),
        "eig_imag_json": json.dumps(summary["eig_imag"]),
    }


def train_spatial_oracle(X, target_force, x_grid, force_fields, args, device):
    model = ForceSVGP(
        make_inducing(args, device),
        args.n_replicas,
        args.init_ls,
        args.init_os,
        args.learn_inducing,
        getattr(args, "spatial_kernel", "rbf"),
        float(getattr(args, "rq_alpha", 1.0)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_spatial)
    N, M, T = X.shape
    x_in = X.reshape(N, M * T).unsqueeze(-1)
    target = target_force.reshape(N, M * T)
    t0 = time.time()
    for _ in range(args.epochs_spatial):
        opt.zero_grad()
        dist = model(x_in)
        var = dist.variance + args.obs_noise_std**2
        nll = 0.5 * ((target - dist.mean).pow(2) / var + torch.log(var) + math.log(2.0 * math.pi)).sum()
        kl = model.variational_strategy.kl_divergence().sum()
        loss = (nll + args.beta_kl * kl) / X.numel() + spatial_hyperprior_loss(model, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
    row = {
        "name": "oracle_spatial",
        "a_mode": "none",
        "n_aux": 0,
        "elbo_or_nll_per_obs": float(loss.detach().cpu()),
        "seconds": time.time() - t0,
        **spatial_kernel_metrics(model, args),
        **force_field_metrics(model, x_grid, force_fields),
    }
    row.update({"memory_kernel_rel_rmse": float("nan"), "memory_kernel_max_rel_err": float("nan")})
    return row


def train_memory_oracle(residual, V, true_A, true_c, args, device):
    best_row = None
    for a_mode in args.memory_only_modes.split(","):
        model = MemoryOnlyModel(args.n_aux, a_mode, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_memory)
        t0 = time.time()
        for _ in range(args.epochs_memory):
            opt.zero_grad()
            nll = model.nll(residual, V, args.dt)
            penalty = args.stability_penalty * model.memory.stability_penalty(args.stability_margin)
            loss = nll / residual.numel() + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        row = {
            "name": f"oracle_memory_{a_mode}",
            "a_mode": a_mode,
            "n_aux": args.n_aux,
            "elbo_or_nll_per_obs": float(loss.detach().cpu()),
            "seconds": time.time() - t0,
            **memory_metrics(model.memory, true_A, true_c, args),
        }
        row.update({
            "lengthscale": float("nan"),
            "outputscale": float("nan"),
            "spatial_force_kernel_rel_rmse": float("nan"),
            "force_field_nrmse": float("nan"),
            "lengthscale_rel_err": float("nan"),
            "outputscale_rel_err": float("nan"),
        })
        if best_row is None or row["memory_kernel_rel_rmse"] < best_row["memory_kernel_rel_rmse"]:
            best_row = row
    return best_row


def train_joint_variant(name, a_mode, X, V, A_obs, x_grid, force_fields, true_A, true_c, args, device):
    best = None
    for start in range(args.starts):
        torch.manual_seed(args.seed * 1009 + start * 97 + len(a_mode))
        model = JointSDGLE(make_inducing(args, device), args.n_replicas, args.n_aux, a_mode, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_joint)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_joint, 1), eta_min=args.lr_joint * 0.2)
        t0 = time.time()
        last = None
        for _ in range(args.epochs_joint):
            opt.zero_grad()
            ll = model.kalman_filter_likelihood_batch(X, V, A_obs, args.dt)
            kl = model.variational_strategy.kl_divergence().sum()
            penalty = args.stability_penalty * model.memory.stability_penalty(args.stability_margin)
            loss = -(ll - args.beta_kl * kl) / X.numel() + penalty + spatial_hyperprior_loss(model, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            last = float(loss.detach().cpu())
        row = {
            "name": name,
            "a_mode": a_mode,
            "n_aux": args.n_aux,
            "start": start,
            "elbo_or_nll_per_obs": last,
            "seconds": time.time() - t0,
            **spatial_kernel_metrics(model, args),
            **force_field_metrics(model, x_grid, force_fields),
            **memory_metrics(model.memory, true_A, true_c, args),
        }
        score = row["spatial_force_kernel_rel_rmse"] + row["memory_kernel_rel_rmse"]
        if best is None or score < best[0]:
            best = (score, row)
    assert best is not None
    return best[1]


def write_csv(path: Path, rows: list[dict]):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict]):
    joint = [r for r in rows if r["name"].startswith("joint")]
    labels = [r["name"].replace("joint_", "") for r in joint]
    x = np.arange(len(joint))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    width = 0.36
    spatial = [r["spatial_force_kernel_rel_rmse"] for r in joint]
    memory = [r["memory_kernel_rel_rmse"] for r in joint]
    ax.bar(x - width / 2, spatial, width, label="spatial kernel")
    ax.bar(x + width / 2, memory, width, label="memory kernel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("relative RMSE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path: Path, rows: list[dict], args, truth: dict):
    oracle_spatial = next(row for row in rows if row["name"] == "oracle_spatial")
    oracle_memory = next(row for row in rows if row["name"].startswith("oracle_memory"))
    joint = [row for row in rows if row["name"].startswith("joint")]
    best_joint = min(joint, key=lambda row: row["spatial_force_kernel_rel_rmse"] + row["memory_kernel_rel_rmse"])
    lines = [
        "# Formal SD-GLE Matrix-A Experiment",
        "",
        "Stable SD-GLE implementation with simultaneous spatial GP and matrix-memory inference.",
        "`diag` is the old pure multi-viscous baseline; `diag_skew` adds elastic couplings;",
        "`spd_skew` relaxes the dissipative part to a full SPD matrix plus skew coupling;",
        "`free` is an unconstrained A diagnostic with a stability penalty and learned full process noise.",
        "",
        "## True system",
        f"- true n_aux: `{truth['n_aux']}`",
        f"- true rates: `{truth['rates']}`",
        f"- true c: `{truth['c']}`",
        f"- true spatial lengthscale/outputscale: `{args.true_ls}`, `{args.true_os}`",
        "",
        "## Main Result",
        f"- oracle spatial kernel RMSE: `{oracle_spatial['spatial_force_kernel_rel_rmse']:.4f}`",
        f"- oracle memory kernel RMSE: `{oracle_memory['memory_kernel_rel_rmse']:.4f}`",
        f"- best joint model: `{best_joint['name']}`",
        f"- best joint spatial kernel RMSE: `{best_joint['spatial_force_kernel_rel_rmse']:.4f}`",
        f"- best joint memory kernel RMSE: `{best_joint['memory_kernel_rel_rmse']:.4f}`",
        "",
        "## Joint Variants",
        "| model | spatial kernel RMSE | force-field NRMSE | memory kernel RMSE | ls | os |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in joint:
        lines.append(
            f"| {row['name']} | {row['spatial_force_kernel_rel_rmse']:.4f} | "
            f"{row['force_field_nrmse']:.4f} | {row['memory_kernel_rel_rmse']:.4f} | "
            f"{row['lengthscale']:.4f} | {row['outputscale']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Compare joint spatial RMSE to `oracle_spatial` to quantify the cost of simultaneous memory inference.",
            "- Compare joint memory RMSE to `oracle_memory` to quantify the cost of unknown spatial force.",
            "- In production runs, prefer `diag_skew` or `spd_skew`; `free` is kept as a diagnostic because unconstrained A can overfit or become weakly stable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/sdgle_formal_results"))
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
    parser.add_argument("--a-modes", default="diag,diag_skew,spd_skew,free")
    parser.add_argument("--memory-only-modes", default="diag_skew,spd_skew")
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
    parser.add_argument("--init-rates", default="")
    parser.add_argument("--init-skew", type=float, default=-0.1)
    parser.add_argument("--stability-penalty", type=float, default=5.0)
    parser.add_argument("--stability-margin", type=float, default=0.03)
    parser.add_argument("--kernel-t-max", type=float, default=8.0)
    parser.add_argument("--kernel-r-max", type=float, default=4.0)
    parser.add_argument("--kernel-points", type=int, default=300)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.outdir.mkdir(parents=True, exist_ok=True)
    X, V, A_obs, F_at_x, M_true, x_grid, force_fields, truth = generate_data(args, device)
    true_A = torch.tensor(truth["A"], device=device)
    true_c = torch.tensor(truth["c"], device=device)

    rows = []
    rows.append(train_spatial_oracle(X, A_obs - M_true, x_grid, force_fields, args, device))
    rows.append(train_memory_oracle(A_obs - F_at_x, V, true_A, true_c, args, device))
    for a_mode in [item.strip() for item in args.a_modes.split(",") if item.strip()]:
        rows.append(train_joint_variant(f"joint_{a_mode}", a_mode, X, V, A_obs, x_grid, force_fields, true_A, true_c, args, device))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    result = {
        "device": str(device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "truth": truth,
        "rows": rows,
    }
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.outdir / "metrics.csv", rows)
    plot_summary(args.outdir / "joint_rmse_summary.png", rows)
    write_report(args.outdir / "README.md", rows, args, truth)
    print(json.dumps({"saved": str(args.outdir), "rows": rows}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
