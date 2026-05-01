import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import homework4


# ─── Data Collection ──────────────────────────────────────────────────────────

def collect_demonstrations(n_trajectories=200, steps=100, render_mode="offscreen"):
    """
    Each trajectory yields an array of shape (steps, 5):
        columns: [ee_y, ee_z, obj_y, obj_z, obj_height]
    obj_height is constant within a trajectory (random per reset).
    """
    env = homework4.Hw5Env(render_mode=render_mode)
    trajectories = []

    for i in range(n_trajectories):
        env.reset()
        p1 = np.array([0.5,  0.30, 1.04])
        p2 = np.array([0.5,  0.15, np.random.uniform(1.04, 1.4)])
        p3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p4 = np.array([0.5, -0.30, 1.04])
        curve = homework4.bezier(np.stack([p1, p2, p3, p4]), steps=steps)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180],
                                  n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())

        trajectories.append(np.array(states))   # (steps, 5)
        print(f"Collected {i+1}/{n_trajectories}", end="\r")

    print()
    return trajectories


def _load_demonstrations(path):
    try:
        return list(np.load(path))
    except ValueError:
        # legacy file saved with dtype=object — migrate to float64 and re-save
        print("Migrating demonstrations file to float64…")
        data = np.load(path, allow_pickle=True)
        trajectories = [t.astype(np.float64) for t in data]
        np.save(path, np.array(trajectories, dtype=np.float64))
        return trajectories


# ─── CNMP ─────────────────────────────────────────────────────────────────────

class CNMP(torch.nn.Module):
    """
    Conditional Neural Movement Primitive.

    Encoder input : (t, e_y, e_z, o_y, o_z)   shape d_x + d_y
    Aggregation   : mean over context points
    Decoder input : (r, t, h)                   shape hidden + d_x + d_c
    Decoder output: (mean, log_std) for (e_y, e_z, o_y, o_z)
    """

    def __init__(self, d_x=1, d_c=1, d_y=4, hidden_size=128,
                 num_hidden_layers=3, min_std=0.1):
        super().__init__()
        self.d_x, self.d_c, self.d_y = d_x, d_c, d_y
        self.min_std = min_std

        def mlp(in_dim, out_dim):
            layers = [torch.nn.Linear(in_dim, hidden_size), torch.nn.ReLU()]
            for _ in range(num_hidden_layers - 1):
                layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
            layers.append(torch.nn.Linear(hidden_size, out_dim))
            return torch.nn.Sequential(*layers)

        self.encoder = mlp(d_x + d_y, hidden_size)
        self.decoder = mlp(hidden_size + d_x + d_c, 2 * d_y)

    def forward(self, obs, query_t, cond_h, obs_mask=None):
        """
        obs     : (B, n_ctx, d_x+d_y)
        query_t : (B, n_tgt, d_x)
        cond_h  : (B, d_c)
        Returns mean, std each (B, n_tgt, d_y)
        """
        h = self.encoder(obs)                              # (B, n_ctx, hidden)
        if obs_mask is not None:
            r = (h * obs_mask.unsqueeze(2)).sum(1) / obs_mask.sum(1, keepdim=True)
        else:
            r = h.mean(1)                                  # (B, hidden)

        n_tgt = query_t.shape[1]
        r_exp = r.unsqueeze(1).expand(-1, n_tgt, -1)      # (B, n_tgt, hidden)
        c_exp = cond_h.unsqueeze(1).expand(-1, n_tgt, -1) # (B, n_tgt, d_c)
        dec_in = torch.cat([r_exp, query_t, c_exp], dim=-1)
        out = self.decoder(dec_in)
        mean = out[..., :self.d_y]
        std  = torch.nn.functional.softplus(out[..., self.d_y:]) + self.min_std
        return mean, std

    def nll_loss(self, obs, query_t, cond_h, target_y, obs_mask=None):
        mean, std = self.forward(obs, query_t, cond_h, obs_mask)
        return -torch.distributions.Normal(mean, std).log_prob(target_y).mean()


# ─── Normalisation helpers ────────────────────────────────────────────────────

def compute_stats(trajectories):
    """Compute mean/std for state dims (ey,ez,oy,oz) and height h."""
    all_states = np.concatenate(trajectories)     # (N*T, 5)
    state_mean = all_states[:, :4].mean(0)
    state_std  = all_states[:, :4].std(0)  + 1e-8
    h_mean     = all_states[:,  4].mean()
    h_std      = all_states[:,  4].std()   + 1e-8
    return state_mean, state_std, h_mean, h_std


def normalize(trajectories, state_mean, state_std, h_mean, h_std):
    out = []
    for traj in trajectories:
        n = traj.copy()
        n[:, :4] = (traj[:, :4] - state_mean) / state_std
        n[:,  4] = (traj[:,  4] - h_mean)      / h_std
        out.append(n)
    return out


# ─── Batch sampling ───────────────────────────────────────────────────────────

def sample_episode(trajectories, t_grid, n_ctx_max, n_tgt_max, device):
    """
    Returns one training instance (batch_size=1):
        obs    : (1, n_ctx, 5)  — (t, ey, ez, oy, oz)
        tgt_t  : (1, n_tgt, 1) — query times
        cond   : (1, 1)         — normalised object height h
        tgt_y  : (1, n_tgt, 4) — target (ey, ez, oy, oz)
    """
    traj = trajectories[np.random.randint(len(trajectories))]
    T    = len(t_grid)

    n_ctx = np.random.randint(1, n_ctx_max + 1)
    n_tgt = np.random.randint(1, n_tgt_max + 1)
    ci    = np.random.choice(T, n_ctx, replace=False)
    ti    = np.random.choice(T, n_tgt, replace=False)

    obs   = np.c_[t_grid[ci],   traj[ci, :4]]  # (n_ctx, 5)
    tgt_t = t_grid[ti, None]                    # (n_tgt, 1)
    cond  = traj[0, 4:5][None]                  # (1, 1) — h constant per traj
    tgt_y = traj[ti, :4]                        # (n_tgt, 4)

    f = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    return f(obs)[None], f(tgt_t)[None], f(cond), f(tgt_y)[None]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_cnmp(model, train_trajs, n_iters=50000, lr=1e-4,
               n_ctx_max=10, n_tgt_max=20, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t_grid    = np.linspace(0, 1, train_trajs[0].shape[0])
    losses    = []

    model.train()
    for i in range(1, n_iters + 1):
        obs, tgt_t, cond, tgt_y = sample_episode(
            train_trajs, t_grid, n_ctx_max, n_tgt_max, device)
        optimizer.zero_grad()
        loss = model.nll_loss(obs, tgt_t, cond, tgt_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 5000 == 0:
            print(f"  [{i:5d}/{n_iters}] loss {np.mean(losses[-500:]):.4f}")

    return losses


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_cnmp(model, test_trajs, state_std, n_tests=100,
                  n_ctx_max=10, n_tgt_max=20, device="cpu"):
    """
    Returns MSE arrays (length n_tests) in original units.
    Note: MSE in original units = state_std^2 * MSE_normalised because
          the mean shift cancels in squared differences.
    """
    model.eval()
    t_grid      = np.linspace(0, 1, test_trajs[0].shape[0])
    ee_mses, obj_mses = [], []

    with torch.no_grad():
        for _ in range(n_tests):
            obs, tgt_t, cond, tgt_y = sample_episode(
                test_trajs, t_grid, n_ctx_max, n_tgt_max, device)
            mean, _ = model(obs, tgt_t, cond)

            # scale back to original units (shift cancels in MSE)
            pred  = mean.squeeze(0).cpu().numpy() * state_std
            truth = tgt_y.squeeze(0).cpu().numpy() * state_std

            ee_mses .append(np.mean((pred[:, :2] - truth[:, :2]) ** 2))
            obj_mses.append(np.mean((pred[:, 2:] - truth[:, 2:]) ** 2))

    return np.array(ee_mses), np.array(obj_mses)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(losses, ee_mses, obj_mses, save_path="results.png"):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss (smoothed)
    window = min(500, len(losses) // 10)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    axes[0].plot(smooth, color="steelblue")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("NLL Loss")
    axes[0].set_title("Training Loss Curve")
    axes[0].grid(True)

    # MSE bar plot
    labels = ["End-Effector", "Object"]
    means  = [ee_mses.mean(), obj_mses.mean()]
    stds   = [ee_mses.std(),  obj_mses.std()]
    x      = np.arange(len(labels))
    axes[1].bar(x, means, yerr=stds, capsize=8,
                color=["steelblue", "tomato"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("MSE (original units)")
    axes[1].set_title(f"Prediction MSE — {len(ee_mses)} tests")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")


# ─── Test visualisation ───────────────────────────────────────────────────────

def visualize_test(model, trajectories, state_mean, state_std, h_mean, h_std,
                   n_ctx=5, device="cpu"):
    """
    Pick one random trajectory from `trajectories` (raw, un-normalised).
    Sample n_ctx random context points, then query the model across all
    timesteps and plot predicted vs ground-truth for all four output dims.
    """
    model.eval()
    traj_raw = trajectories[np.random.randint(len(trajectories))]  # (T, 5)
    T        = traj_raw.shape[0]
    t_grid   = np.linspace(0, 1, T)

    # normalise for model input
    traj_norm        = traj_raw.copy()
    traj_norm[:, :4] = (traj_raw[:, :4] - state_mean) / state_std
    traj_norm[:,  4] = (traj_raw[:,  4] - h_mean)      / h_std

    # random context indices
    ctx_idx = np.sort(np.random.choice(T, n_ctx, replace=False))

    obs   = np.c_[t_grid[ctx_idx], traj_norm[ctx_idx, :4]]  # (n_ctx, 5)
    cond  = traj_norm[0, 4:5][None]                          # (1, 1)
    q_t   = t_grid[:, None]                                  # (T, 1)

    f = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean_n, std_n = model(f(obs)[None], f(q_t)[None], f(cond))

    # denormalise predictions back to original units
    mean_pred = mean_n.squeeze(0).cpu().numpy() * state_std + state_mean  # (T, 4)
    std_pred  = std_n .squeeze(0).cpu().numpy() * state_std               # (T, 4)

    labels   = ["$e_y$", "$e_z$", "$o_y$", "$o_z$"]
    colors   = ["steelblue", "steelblue", "tomato", "tomato"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()

    h_val = traj_raw[0, 4]
    fig.suptitle(f"CNMP prediction  |  object height h = {h_val:.3f} m  |  "
                 f"{n_ctx} context points", fontsize=12)

    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        gt = traj_raw[:, i]

        ax.plot(t_grid, gt,         color="black",  lw=1.5, label="ground truth")
        ax.plot(t_grid, mean_pred[:, i], color=col, lw=1.5, ls="--", label="predicted mean")
        ax.fill_between(t_grid,
                        mean_pred[:, i] - std_pred[:, i],
                        mean_pred[:, i] + std_pred[:, i],
                        color=col, alpha=0.20, label="±1 std")
        ax.scatter(t_grid[ctx_idx], traj_raw[ctx_idx, i],
                   color="black", zorder=5, s=40, label="context" if i == 0 else None)
        ax.set_ylabel(lbl)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("test_visualization.png", dpi=150)
    plt.show()
    print("Saved test_visualization.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_FILE  = "demonstrations.npy"
    MODEL_FILE = "cnmp_model.pt"
    STATS_FILE = "cnmp_stats.npz"

    # ── TEST MODE ──────────────────────────────────────────────────────────────
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        for f in (MODEL_FILE, STATS_FILE, DATA_FILE):
            if not os.path.exists(f):
                raise FileNotFoundError(f"{f} not found — train the model first.")

        stats = np.load(STATS_FILE)
        state_mean = stats["state_mean"]
        state_std  = stats["state_std"]
        h_mean     = float(stats["h_mean"])
        h_std      = float(stats["h_std"])

        model = CNMP(d_x=1, d_c=1, d_y=4, hidden_size=128, num_hidden_layers=3).to(device)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))

        trajectories = _load_demonstrations(DATA_FILE)
        split        = int(0.8 * len(trajectories))
        test_raw     = trajectories[split:]

        n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"Visualising on {len(test_raw)} held-out trajectories "
              f"with {n_ctx} context points…")
        visualize_test(model, test_raw, state_mean, state_std, h_mean, h_std,
                       n_ctx=n_ctx, device=device)

    # ── TRAIN MODE ─────────────────────────────────────────────────────────────
    else:
        # 1. Collect or load demonstrations ------------------------------------
        if os.path.exists(DATA_FILE):
            print("Loading saved demonstrations...")
            trajectories = _load_demonstrations(DATA_FILE)
        else:
            print("Collecting 200 demonstrations (offscreen)...")
            trajectories = collect_demonstrations(200, render_mode="offscreen")
            np.save(DATA_FILE, np.array(trajectories, dtype=np.float64))
            print(f"Saved to {DATA_FILE}")

        T = trajectories[0].shape[0]
        print(f"Dataset: {len(trajectories)} trajectories × {T} steps")

        # 2. Train / test split (80 / 20) --------------------------------------
        split     = int(0.8 * len(trajectories))
        train_raw = trajectories[:split]
        test_raw  = trajectories[split:]

        # 3. Normalise (stats from train set only) -----------------------------
        state_mean, state_std, h_mean, h_std = compute_stats(train_raw)
        train_trajs = normalize(train_raw, state_mean, state_std, h_mean, h_std)
        test_trajs  = normalize(test_raw,  state_mean, state_std, h_mean, h_std)

        # 4. Build model -------------------------------------------------------
        model = CNMP(d_x=1, d_c=1, d_y=4, hidden_size=128,
                     num_hidden_layers=3).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"CNMP — {n_params:,} parameters | device: {device}")

        # 5. Train -------------------------------------------------------------
        print("Training…")
        losses = train_cnmp(model, train_trajs, n_iters=50000, lr=1e-4,
                             n_ctx_max=10, n_tgt_max=20, device=device)

        torch.save(model.state_dict(), MODEL_FILE)
        np.savez(STATS_FILE, state_mean=state_mean, state_std=state_std,
                 h_mean=h_mean, h_std=h_std)
        print(f"Saved {MODEL_FILE} and {STATS_FILE}")

        # 6. Evaluate ----------------------------------------------------------
        print("Evaluating on test trajectories…")
        ee_mses, obj_mses = evaluate_cnmp(
            model, test_trajs, state_std,
            n_tests=100, n_ctx_max=10, n_tgt_max=20, device=device)

        print(f"End-effector MSE : {ee_mses.mean():.6f} ± {ee_mses.std():.6f}")
        print(f"Object       MSE : {obj_mses.mean():.6f} ± {obj_mses.std():.6f}")

        # 7. Plot --------------------------------------------------------------
        plot_results(losses, ee_mses, obj_mses)
