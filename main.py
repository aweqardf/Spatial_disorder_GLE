import math
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


class Config:
    # --- Data Generation Parameters ---
    N_REPLICAS = 100  # Number of independent spatial environments (replicas)
    N_TRAJS = 10  # Number of trajectories per replica
    N_STEPS = 200  # Number of time steps per trajectory
    DT = 0.01  # Integration time step
    OBS_NOISE_STD = 0.1  # Standard deviation of observation noise (for numerical stability)

    # --- Memory Kernel Parameters (using 1-d auxiliary variable for simplicity) ---
    TRUE_A = 0.5  # Decay rate
    TRUE_AP = 1.2  # Coupling constant
    TRUE_B = 0.4  # Auxiliary noise strength

    # --- True GP Hyperparameters (Potential U(x)) ---
    TRUE_LS = 1.0  # Lengthscale of the spatial potential
    TRUE_OS = 0.3  # Outputscale (variance) of the spatial potential

    # --- Training Parameters ---
    EPOCHS = 1500  # Number of training epochs
    LEARNING_RATE = 1e-2  # Optimizer learning rate
    M_INDUCING = 64  # Number of sparse inducing points


# ============================================================
# 1. Custom Force Kernel (Analytically Differentiated RBF)
# ============================================================
class RBFForceKernel(gpytorch.kernels.Kernel):
    """
    Kernel for the negative gradient of the RBF potential U(x): F(x) = -dU/dx.
    Mathematically: k_F(x, x') = d^2 / (dx dx') k_RBF(x, x')
    This prevents Out-Of-Memory (OOM) errors by avoiding PyTorch Autograd loops.
    """
    is_stationary = True
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        # Scale inputs by the lengthscale
        x1_scaled = x1.div(self.lengthscale)
        x2_scaled = x2.div(self.lengthscale)

        # Calculate pairwise scaled differences: (x - x') / l
        diff = x1_scaled.unsqueeze(-2) - x2_scaled.unsqueeze(-3)
        sq_dist = diff.pow(2).squeeze(-1)

        if diag:
            # Handle the diagonal (x1 == x2) where sq_dist = 0
            l_sq = self.lengthscale.squeeze(-1).pow(2)
            res = torch.ones(*x1.shape[:-1], device=x1.device)
            return res / l_sq

        # Compute the Force Kernel: (1/l^2) * (1 - sq_dist) * exp(-0.5 * sq_dist)
        exp_term = torch.exp(-0.5 * sq_dist)
        l_sq = self.lengthscale.squeeze(-1).unsqueeze(-1).pow(2)

        k_f = (1.0 - sq_dist) * exp_term / l_sq
        return k_f


# ============================================================
# 2. Data generator: Replica-dependent quenched GP force + GLE
# ============================================================
class GLESimulator:
    def __init__(self, config=Config):
        self.N = config.N_REPLICAS
        self.M = config.N_TRAJS
        self.T = config.N_STEPS
        self.dt = config.DT
        self.m = 1.0

        self.obs_noise_std = config.OBS_NOISE_STD

        # True GLE parameters (n_aux = 1)
        self.true_A = torch.tensor([[config.TRUE_A]], device=device)
        self.true_ap = torch.tensor([[config.TRUE_AP]], device=device)
        self.true_B = torch.tensor([[config.TRUE_B]], device=device)

        # True GP hyper-parameters
        self.true_ls = config.TRUE_LS
        self.true_os = config.TRUE_OS

    @torch.no_grad()
    def generate_data(self, grid_min=-5.0, grid_max=5.0, n_grid=500):
        x_grid = torch.linspace(grid_min, grid_max, n_grid, device=device)

        # ====== Generate true force field data using the Force Kernel ======
        dist = x_grid.unsqueeze(0) - x_grid.unsqueeze(1)
        sq_dist_scaled = (dist ** 2) / (self.true_ls ** 2)

        K = (self.true_os / (self.true_ls ** 2)) * \
            (1.0 - sq_dist_scaled) * \
            torch.exp(-0.5 * sq_dist_scaled)

        K = K + torch.eye(n_grid, device=device) * 1e-5
        L = torch.linalg.cholesky(K)
        F_fields = (L @ torch.randn(n_grid, self.N, device=device)).T

        def x_to_index(x_val):
            u = (x_val - grid_min) / (grid_max - grid_min)
            idx = (u * (n_grid - 1)).long()
            return idx.clamp(0, n_grid - 1)

        X = torch.zeros(self.N, self.M, self.T, device=device)
        V = torch.zeros(self.N, self.M, self.T, device=device)
        A_acc = torch.zeros(self.N, self.M, self.T, device=device)

        for i in range(self.N):
            fi_vals = F_fields[i]

            for j in range(self.M):
                curr_x = torch.randn(1, device=device) * 0.5
                curr_v = torch.randn(1, device=device) * 1.0
                curr_z = torch.zeros(1, 1, device=device)

                for t in range(self.T):
                    f_c = fi_vals[x_to_index(curr_x)].view(1)

                    # --- Core Physics Update ---
                    # Calculate true physical acceleration
                    acc_true = (f_c + (self.true_ap.T @ curr_z).view(1)) / self.m

                    # Generate observed acceleration by adding Gaussian noise
                    # Note: Only A_acc contains noise; physical evolution uses acc_true
                    obs_noise = torch.randn(1, device=device) * self.obs_noise_std
                    acc_obs = acc_true + obs_noise

                    X[i, j, t] = curr_x
                    V[i, j, t] = curr_v
                    A_acc[i, j, t] = acc_obs

                    # Euler-Maruyama integration (using true physical acceleration)
                    curr_x = curr_x + curr_v * self.dt
                    curr_v = curr_v + acc_true * self.dt

                    # Auxiliary variable evolution: dz = -A z dt - ap v dt + B sqrt(dt) dW
                    curr_z = curr_z - (self.true_A @ curr_z) * self.dt \
                             - (self.true_ap * curr_v.view(1, 1)) * self.dt \
                             + self.true_B * math.sqrt(self.dt) * torch.randn_like(curr_z)

        return X, V, A_acc, x_grid, F_fields


# ============================================================
# 3. SVGP force + KF likelihood for auxiliary OU (GLE embedding)
# ============================================================
class GLESVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, n_replicas, config=Config, n_aux=1):
        self.n_replicas = n_replicas
        self.n_aux = n_aux
        self.obs_noise_var = config.OBS_NOISE_STD ** 2

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([n_replicas]),
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_replicas]))

        # Apply the custom Force Kernel to directly model the spatial force field
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFForceKernel())

        # Learnable GLE parameters
        self.raw_A = torch.nn.Parameter(torch.randn(n_aux))
        self.raw_ap = torch.nn.Parameter(torch.randn(n_aux))
        self.raw_B = torch.nn.Parameter(torch.randn(n_aux))

        # Initialize the Force Kernel parameters
        init_ls = torch.rand(1)
        init_os = torch.rand(1)

        self.covar_module.base_kernel.lengthscale = init_ls
        self.covar_module.outputscale = init_os

        self.covar_module.base_kernel.raw_lengthscale.requires_grad = True
        self.covar_module.raw_outputscale.requires_grad = True

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def A(self):
        return torch.diag(torch.exp(self.raw_A))

    @property
    def ap(self):
        return torch.exp(self.raw_ap.unsqueeze(-1))

    @property
    def B(self):
        return torch.diag(torch.exp(self.raw_B))

    @property
    def noise(self):
        # Fixed observation noise variance to prevent variance collapse
        return torch.tensor([self.obs_noise_var], device=device)

    def kalman_filter_likelihood_batch(self, x, v, a, m, dt=0.01):
        device = x.device
        N, M, T = x.shape
        n_aux = self.ap.shape[0]

        # GP prediction for F_c(x)
        x_in = x.reshape(N, M * T).unsqueeze(-1)
        gp_dist = self(x_in)
        mu_f = gp_dist.mean.reshape(N, M, T)
        var_f = gp_dist.variance.reshape(N, M, T)

        # KF Matrices
        I0 = torch.eye(n_aux, device=device)
        I = I0.expand(N, M, n_aux, n_aux)

        Phi0 = (torch.eye(n_aux, device=device) - self.A * dt)
        Phi = Phi0.expand(N, M, n_aux, n_aux)

        Q0 = (self.B @ self.B.T) * dt
        Q = Q0.expand(N, M, n_aux, n_aux)

        H0 = self.ap.T
        H = H0.expand(N, M, 1, n_aux)
        HT = H.transpose(-1, -2)

        # Init state
        z_mean = torch.zeros(N, M, n_aux, 1, device=device)
        z_cov = torch.eye(n_aux, device=device).expand(N, M, n_aux, n_aux).clone() * 0.01

        if not torch.is_tensor(m):
            m = torch.tensor(m, device=device, dtype=x.dtype)
        m_b = m if m.ndim == 0 else m.view(N, 1, 1)

        log2pi = math.log(2.0 * math.pi)
        log_likelihood = 0.0

        ap_col = self.ap.view(1, 1, n_aux, 1)

        for t in range(T):
            y_obs = m_b * a[:, :, t] - mu_f[:, :, t]

            # Using the fixed observation noise variance
            R = var_f[:, :, t] + self.noise

            # KF Predict step
            z_mean = torch.matmul(Phi, z_mean) - ap_col * v[:, :, t].view(N, M, 1, 1) * dt
            z_cov = torch.matmul(torch.matmul(Phi, z_cov), Phi.transpose(-1, -2)) + Q

            # KF Update step
            Hz = torch.matmul(H, z_cov)
            S = torch.matmul(Hz, HT).squeeze(-1).squeeze(-1) + R

            zcovHT = torch.matmul(z_cov, HT)
            K = zcovHT / S.view(N, M, 1, 1)

            Hzmean = torch.matmul(H, z_mean).squeeze(-1).squeeze(-1)
            innov = y_obs - Hzmean

            log_likelihood = log_likelihood + (-0.5 * (innov ** 2 / S + torch.log(S) + log2pi)).sum()

            z_mean = z_mean + K * innov.view(N, M, 1, 1)
            KH = torch.matmul(K, H)
            z_cov = torch.matmul(I - KH, z_cov)

        return log_likelihood


# ============================================================
# 4. Training Loop
# ============================================================
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize data generator using the Config class
    sim = GLESimulator(config=Config)
    X, V, A_acc, x_grid, F_fields = sim.generate_data()

    # Build inducing points initialized on a uniform grid
    z_base = torch.linspace(-5, 5, Config.M_INDUCING, device=device).unsqueeze(-1)
    inducing_points = z_base.unsqueeze(0).repeat(Config.N_REPLICAS, 1, 1).contiguous()

    # Initialize the SD-GLE SVGP model
    model = GLESVGP(inducing_points=inducing_points, n_replicas=Config.N_REPLICAS, config=Config, n_aux=1).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Optimization loop (Maximizing ELBO)
    for ep in range(1, Config.EPOCHS + 1):
        optimizer.zero_grad()

        ll = model.kalman_filter_likelihood_batch(X, V, A_acc, m=sim.m, dt=Config.DT)
        kl = model.variational_strategy.kl_divergence().sum()

        loss = -(ll - kl)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Print progress every 20 epochs
        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                A_est = torch.diag(model.A).detach().cpu().numpy()
                ap_est = model.ap.detach().cpu().numpy().reshape(-1)
                B_est = torch.diag(model.B).detach().cpu().numpy()
                ls_val = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().item()
                os_val = model.covar_module.outputscale.detach().cpu().numpy().item()
                kl_val = kl.detach().cpu().item()
                ll_val = ll.detach().cpu().item()
            print(
                f"Epoch {ep:4d} | loss={loss.item():.6f} | ll={ll_val:.2f} | kl={kl_val:.2f}")

    # ============================================================
    # 5. Visualization: Memory Kernel and Spatial Kernel
    # ============================================================
    model.eval()

    # Extract learned parameters
    with torch.no_grad():
        A_learned = model.A.detach().cpu().numpy()[0, 0]
        ap_learned = model.ap.detach().cpu().numpy()[0, 0]
        ls_learned = model.covar_module.base_kernel.lengthscale.detach().cpu().item()
        os_learned = model.covar_module.outputscale.detach().cpu().item()

    # Ground truth parameters
    true_A = sim.true_A.cpu().item()
    true_ap = sim.true_ap.cpu().item()
    true_ls = sim.true_ls
    true_os = sim.true_os

    # --- Setup Data for Plotting ---
    # 1. Memory Kernel: K(t) = ap^2 * exp(-A * t)
    t_arr = np.linspace(0, 10, 200)
    K_true = (true_ap ** 2) * np.exp(-true_A * t_arr)
    K_learned = (ap_learned ** 2) * np.exp(-A_learned * t_arr)

    # 2. Spatial Kernel: C(r) = os * exp(-r^2 / (2 * ls^2))
    r_arr = np.linspace(0, 5, 200)
    C_true = true_os * np.exp(-(r_arr ** 2) / (2 * true_ls ** 2))
    C_learned = os_learned * np.exp(-(r_arr ** 2) / (2 * ls_learned ** 2))

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Memory Kernel
    axs[0].plot(t_arr, K_true, 'k--', lw=2.5, label='True $K(t)$')
    axs[0].plot(t_arr, K_learned, 'r-', lw=2.5, alpha=0.7, label='Learned $K(t)$')
    axs[0].set_xlabel('Time $t$', fontsize=12)
    axs[0].set_ylabel('Memory Kernel $K(t)$', fontsize=12)
    axs[0].set_title('Temporal Memory Kernel Comparison', fontsize=14)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Spatial Kernel (RBF of Potential U)
    axs[1].plot(r_arr, C_true, 'k--', lw=2.5, label='True $C(r)$')
    axs[1].plot(r_arr, C_learned, 'b-', lw=2.5, alpha=0.7, label='Learned $C(r)$')
    axs[1].set_xlabel('Distance $r$', fontsize=12)
    axs[1].set_ylabel('Spatial Covariance $C(r)$', fontsize=12)
    axs[1].set_title('Spatial Potential Kernel Comparison', fontsize=14)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
