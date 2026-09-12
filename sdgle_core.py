from __future__ import annotations

import math

import gpytorch
import torch


torch.set_default_dtype(torch.float32)


class RBFForceKernel(gpytorch.kernels.Kernel):
    """Force covariance induced by an RBF covariance for U(x)."""

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
        return (1.0 - sq_dist) * torch.exp(-0.5 * sq_dist) / l_sq.unsqueeze(-1)


class RQForceKernel(gpytorch.kernels.Kernel):
    """Force covariance induced by a rational-quadratic covariance for U(x)."""

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


class Matern52ForceKernel(gpytorch.kernels.Kernel):
    """Force covariance induced by a Matern-5/2 covariance for U(x)."""

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
        return (5.0 / 3.0) * torch.exp(-y) * (1.0 + y - y.pow(2)) / l_sq.unsqueeze(-1)


def make_force_kernel(kind: str, rq_alpha: float = 1.0) -> gpytorch.kernels.Kernel:
    if kind == "rbf":
        return RBFForceKernel()
    if kind == "rq":
        return RQForceKernel(alpha=rq_alpha)
    if kind == "matern52":
        return Matern52ForceKernel()
    raise ValueError(f"unknown spatial kernel: {kind}")


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_optional_rates(text: str | None, n_aux: int) -> list[float] | None:
    if not text:
        return None
    values = parse_float_list(text)
    if len(values) != n_aux:
        raise ValueError(f"expected {n_aux} comma-separated rates, got {len(values)}")
    return values


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
    lower = torch.zeros(n_aux, n_aux, device=raw.device, dtype=raw.dtype)
    cursor = 0
    for i in range(n_aux):
        for j in range(i + 1):
            value = raw[cursor]
            lower[i, j] = positive(value, diag_floor) if i == j else value
            cursor += 1
    return lower


def matrix_memory_kernel(A: torch.Tensor, c: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
    return torch.stack([c @ torch.matrix_exp(-A * t) @ c for t in t_eval])


def force_covariance_rbf(dist: torch.Tensor, lengthscale: float, outputscale: float) -> torch.Tensor:
    y2 = dist.pow(2) / (lengthscale * lengthscale)
    return outputscale * (1.0 - y2) * torch.exp(-0.5 * y2) / (lengthscale * lengthscale)


def force_covariance(
    dist: torch.Tensor,
    lengthscale: torch.Tensor | float,
    outputscale: torch.Tensor | float,
    kind: str = "rbf",
    rq_alpha: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    lengthscale = torch.as_tensor(lengthscale, device=dist.device, dtype=dist.dtype).clamp_min(1e-8)
    outputscale = torch.as_tensor(outputscale, device=dist.device, dtype=dist.dtype)
    if kind == "rbf":
        y = dist / lengthscale
        return outputscale * (1.0 - y.pow(2)) * torch.exp(-0.5 * y.pow(2)) / lengthscale.pow(2)
    if kind == "rq":
        alpha = torch.as_tensor(rq_alpha, device=dist.device, dtype=dist.dtype).clamp_min(1e-6)
        y2 = (dist / lengthscale).pow(2)
        s = 1.0 + y2 / (2.0 * alpha)
        return outputscale * (
            s.pow(-alpha - 1.0) - ((alpha + 1.0) / alpha) * y2 * s.pow(-alpha - 2.0)
        ) / lengthscale.pow(2)
    if kind == "matern52":
        y = math.sqrt(5.0) * torch.abs(dist) / lengthscale
        return outputscale * (5.0 / (3.0 * lengthscale.pow(2))) * torch.exp(-y) * (1.0 + y - y.pow(2))
    raise ValueError(f"unknown spatial kernel: {kind}")


class MatrixMemory(torch.nn.Module):
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
        self.raw_skew = torch.nn.Parameter(torch.ones(n_aux * (n_aux - 1) // 2) * init_skew)
        signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n_aux)])
        self.c = torch.nn.Parameter(signs * init_c_scale / math.sqrt(n_aux))

        n_tril = n_aux * (n_aux + 1) // 2
        self.raw_spd_tril = torch.nn.Parameter(torch.zeros(n_tril))
        self.raw_B_tril = torch.nn.Parameter(torch.zeros(n_tril))
        cursor = 0
        for i in range(n_aux):
            for j in range(i + 1):
                if i == j:
                    self.raw_spd_tril.data[cursor] = inv_softplus(torch.tensor(math.sqrt(init_rates[i].item())))
                    self.raw_B_tril.data[cursor] = inv_softplus(torch.tensor(0.35))
                cursor += 1

        if init_free_A is None:
            init_free_A = torch.diag(init_rates)
        self.raw_A_free = torch.nn.Parameter(init_free_A.clone().detach().float())

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
            lower = tril_from_raw(self.raw_spd_tril, self.n_aux, 1e-3)
            return lower @ lower.T + 1e-4 * eye
        if self.a_mode == "free":
            lower = tril_from_raw(self.raw_B_tril, self.n_aux, 1e-4)
            return lower @ lower.T
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
            lower = tril_from_raw(self.raw_B_tril, self.n_aux, 1e-4)
            q = lower @ lower.T * dt + eye * 1e-7
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
        eig = torch.linalg.eigvals(A).detach().cpu()
        return {
            "A": A.tolist(),
            "c": c.tolist(),
            "eig_real": eig.real.tolist(),
            "eig_imag": eig.imag.tolist(),
            "K0": float(torch.sum(c * c)),
        }


class ForceGP(gpytorch.models.ApproximateGP):
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


class SDGLE(ForceGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        n_replicas: int,
        n_aux: int,
        a_mode: str,
        args,
        init_free_A: torch.Tensor | None = None,
    ):
        init_rates_values = parse_optional_rates(getattr(args, "init_rates", None), n_aux)
        super().__init__(
            inducing_points,
            n_replicas,
            args.init_ls,
            args.init_os,
            args.learn_inducing,
            getattr(args, "spatial_kernel", "rbf"),
            float(getattr(args, "rq_alpha", 1.0)),
        )
        self.obs_var = float(args.obs_noise_std**2)
        self.memory = MatrixMemory(
            n_aux,
            a_mode,
            temperature=args.temperature,
            init_c_scale=args.init_c_scale,
            init_rate=args.init_rate,
            init_skew=args.init_skew,
            init_rates_values=init_rates_values,
            init_free_A=init_free_A,
        )

    def kalman_log_likelihood(self, x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, dt: float) -> torch.Tensor:
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


def make_inducing_grid(
    grid_min: float,
    grid_max: float,
    n_inducing: int,
    n_replicas: int,
    device: torch.device,
) -> torch.Tensor:
    z_base = torch.linspace(grid_min, grid_max, n_inducing, device=device).unsqueeze(-1)
    return z_base.unsqueeze(0).repeat(n_replicas, 1, 1).contiguous()


def spatial_hyperprior_loss(model: ForceGP, *, weight: float, lengthscale: float, outputscale: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.tensor(0.0, device=model.covar_module.outputscale.device)
    ls = model.covar_module.base_kernel.lengthscale.squeeze()
    os = model.covar_module.outputscale.squeeze()
    return weight * (torch.log(ls / lengthscale).pow(2) + torch.log(os / outputscale).pow(2))
