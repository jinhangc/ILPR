from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_methods import (
    OneOffKernelEstimator,
    pav_isotonic_increasing,
    SimConfig,
    TrueModel,
    invert_monotone_grid,
)


def run_trial(cfg: SimConfig, exploration_scale: float, episode_index: int, monotone_mode: str) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(cfg.seed)
    model = TrueModel(beta=cfg.beta)

    def m_true(x: np.ndarray) -> np.ndarray:
        return cfg.theta_true * np.asarray(x, dtype=float)

    remaining = cfg.T
    block_len = 200
    for _ in range(episode_index):
        remaining -= min(block_len, remaining)
        if remaining <= 0:
            break
        block_len *= 2
    block_len = min(block_len, max(remaining, 1))
    explore_len = max(1, min(block_len, int(np.floor(exploration_scale * np.sqrt(block_len)))))
    x_exp = rng.uniform(cfg.x_low, cfg.x_high, size=explore_len)
    p_exp = rng.uniform(cfg.P_low, cfg.P_high, size=explore_len)
    u_obs = p_exp - m_true(x_exp)
    D_exp = (rng.uniform(size=explore_len) < (1.0 - model.F(u_obs))).astype(float)

    h = 0.5 * max(explore_len, 5) ** (-1.0 / (2.0 * cfg.beta + 1.0))
    ker = OneOffKernelEstimator(u_obs, D_exp, h=h)

    umin = float(np.min(u_obs))
    umax = float(np.max(u_obs))
    pad = 0.05 * (umax - umin + 1e-12)
    grid_u = np.linspace(umin - pad, umax + pad, 401)

    S_hat = ker.survival_hat(grid_u)
    S_prime_hat = ker.survival_prime_hat(grid_u)
    F_hat = np.clip(1.0 - S_hat, 0.0, 1.0)
    F_prime_hat = np.maximum(-S_prime_hat, 1e-4)
    phi_raw = grid_u + S_hat / np.where(np.abs(S_prime_hat) < 1e-4, np.sign(S_prime_hat + 1e-12) * 1e-4, S_prime_hat)
    if monotone_mode == "none":
        phi_hat = phi_raw.copy()
    elif monotone_mode == "full":
        phi_hat = np.maximum.accumulate(phi_raw)
    elif monotone_mode == "interior":
        phi_hat = phi_raw.copy()
        mask = (grid_u >= -0.18) & (grid_u <= 0.18)
        phi_hat[mask] = np.maximum.accumulate(phi_hat[mask])
    elif monotone_mode == "interior_extrap":
        phi_hat = phi_raw.copy()
        left_u, right_u = -0.15, 0.15
        mask = (grid_u >= left_u) & (grid_u <= right_u)
        if np.any(mask):
            phi_hat[mask] = pav_isotonic_increasing(phi_hat[mask])
            idx = np.where(mask)[0]
            left_idx = int(idx[0])
            right_idx = int(idx[-1])
            small_slope = 0.15
            for i in range(0, left_idx):
                phi_hat[i] = phi_hat[left_idx] + small_slope * (grid_u[i] - grid_u[left_idx])
            for i in range(right_idx + 1, len(grid_u)):
                phi_hat[i] = phi_hat[right_idx] + small_slope * (grid_u[i] - grid_u[right_idx])
            phi_hat = np.maximum.accumulate(phi_hat)
    else:
        raise ValueError(f"Unknown monotone_mode: {monotone_mode}")

    x_plot = np.linspace(cfg.x_low, cfg.x_high, 200)
    m_vals = m_true(x_plot)
    p_star = model.optimal_price(m_vals, (cfg.P_low, cfg.P_high), grid_n=600)
    if monotone_mode == "interior_extrap":
        inv_mask = (grid_u >= -0.15) & (grid_u <= 0.15)
        z_hat = invert_monotone_grid(grid_u[inv_mask], phi_hat[inv_mask], -m_vals)
        z_hat = np.clip(z_hat, -0.15, 0.15)
    else:
        z_hat = invert_monotone_grid(grid_u, phi_hat, -m_vals)
    p_hat = np.clip(z_hat + m_vals, cfg.P_low, cfg.P_high)

    return {
        "grid_u": grid_u,
        "u_obs": u_obs,
        "F_hat": F_hat,
        "F_prime_hat": F_prime_hat,
        "phi_raw": phi_raw,
        "phi_hat": phi_hat,
        "x_plot": x_plot,
        "p_star": p_star,
        "p_hat": p_hat,
        "h": h,
        "explore_len": explore_len,
        "block_len": block_len,
        "episode_index": episode_index,
        "monotone_mode": monotone_mode,
        "model": model,
    }


def make_plot(result: dict[str, np.ndarray | float], cfg: SimConfig, output: Path) -> None:
    model = result["model"]
    grid_u = result["grid_u"]
    true_F = model.F(grid_u)
    true_Fp = model.Fprime(grid_u)
    true_phi = model.phi(grid_u)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    l1, = axes[0].plot(grid_u, true_F, "k--", label="True F(u)")
    l2, = axes[0].plot(grid_u, result["F_hat"], "r-", alpha=0.75, label="Kernel F_hat(u)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Demand CDF")
    ax0b = axes[0].twinx()
    ax0b.hist(result["u_obs"], bins=40, range=(grid_u.min(), grid_u.max()), density=True, color="blue", alpha=0.15)
    axes[0].legend([l1, l2, plt.Rectangle((0, 0), 1, 1, color="blue", alpha=0.15)], ["True F(u)", "Kernel F_hat(u)", "Design density"], loc="upper left")

    axes[1].plot(grid_u, true_Fp, "k--", label="True F'(u)")
    axes[1].plot(grid_u, result["F_prime_hat"], "r-", alpha=0.75, label="Kernel F'_hat(u)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Density")
    axes[1].legend()

    axes[2].plot(grid_u, true_phi, "k--", label="True phi(u)")
    axes[2].plot(grid_u, result["phi_raw"], color="orange", alpha=0.5, label="Raw kernel phi")
    axes[2].plot(grid_u, result["phi_hat"], "r-", alpha=0.85, label="Monotone kernel phi")
    axes[2].set_ylim(-8, 5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Virtual Value")
    axes[2].legend()

    axes[3].plot(result["x_plot"], result["p_star"], "k--", label="p*(x)")
    axes[3].plot(result["x_plot"], result["p_hat"], "r-", alpha=0.85, label="Kernel p_hat(x)")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title("Pricing Policy")
    axes[3].legend()

    fig.suptitle(
        f"Kernel Baseline Diagnostic | beta={cfg.beta}, T={cfg.T}, seed={cfg.seed}, episode={int(result['episode_index'])}, block={int(result['block_len'])}, explore={int(result['explore_len'])}, h={float(result['h']):.4f}, mono={result['monotone_mode']}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--episode", type=int, default=-1)
    parser.add_argument("--monotone", choices=["none", "full", "interior", "interior_extrap"], default="full")
    parser.add_argument("--output", default="outputs/kernel_diagnostic_beta2_T10000_seed0.png")
    args = parser.parse_args()

    cfg = SimConfig(T=args.T, beta=args.beta, seed=args.seed, known_utility=True)
    if args.episode >= 0:
        episode_index = args.episode
    else:
        remaining = cfg.T
        block_len = 200
        episode_index = 0
        while remaining > block_len:
            remaining -= block_len
            block_len *= 2
            episode_index += 1
    result = run_trial(cfg, exploration_scale=args.c, episode_index=episode_index, monotone_mode=args.monotone)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(result, cfg, output)


if __name__ == "__main__":
    main()
