from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


def clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def epanechnikov(u: np.ndarray) -> np.ndarray:
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    out[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return out


def pav_isotonic_increasing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    level = y.copy()
    weight = np.ones(len(y), dtype=int)

    i = 0
    while i < len(level) - 1:
        if level[i] <= level[i + 1] + 1e-12:
            i += 1
            continue

        new_w = weight[i] + weight[i + 1]
        new_level = (weight[i] * level[i] + weight[i + 1] * level[i + 1]) / new_w
        level[i] = new_level
        weight[i] = new_w
        level = np.delete(level, i + 1)
        weight = np.delete(weight, i + 1)
        if i > 0:
            i -= 1

    out = np.empty(len(y), dtype=float)
    idx = 0
    for lev, w in zip(level, weight):
        out[idx : idx + w] = lev
        idx += w
    return out


def invert_monotone_grid(x_grid: np.ndarray, y_grid: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    y_grid = np.asarray(y_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_query = np.asarray(y_query, dtype=float)
    y_mono = np.maximum.accumulate(y_grid)
    yq = np.clip(y_query, y_mono[0], y_mono[-1])
    return np.interp(yq, y_mono, x_grid)


def fit_logistic_working_model(
    X: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
    max_iter: int = 80,
    ridge: float = 1e-6,
) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    p = np.asarray(p, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([np.ones(len(y), dtype=float), X[:, 0], -p[:, 0]])
    beta = np.zeros(Z.shape[1], dtype=float)

    for _ in range(max_iter):
        eta = Z @ beta
        eta = np.clip(eta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-6)
        grad = Z.T @ (y - mu) - ridge * beta
        hess = (Z.T * w) @ Z + ridge * np.eye(Z.shape[1], dtype=float)
        step = np.linalg.solve(hess, grad)
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-8:
            beta = beta_next
            break
        beta = beta_next

    return beta


@dataclass
class TrueModel:
    beta: float
    umin: float = -0.25
    umax: float = 0.25
    rho: float = 20.0
    K: int = 10

    def __post_init__(self) -> None:
        bump_min, bump_max = -0.2, 0.2
        if self.K == 0:
            self.u_bump_centers = np.array([])
            self.signs = np.array([])
            self.bump_width = 0.0
            self.bump_amplitude = 0.0
            return
        if self.K == 1:
            self.u_bump_centers = np.array([(bump_min + bump_max) / 2.0])
            self.signs = np.array([1])
            self.bump_width = (bump_max - bump_min) / 4.0
        else:
            self.u_bump_centers = np.linspace(bump_min, bump_max, self.K)
            self.signs = (-1) ** np.arange(self.K)
            self.bump_width = 0.5 * (bump_max - bump_min) / (self.K - 1)
        self.bump_amplitude = self.rho * (self.bump_width ** self.beta) if self.bump_width > 0 else 0.0

    @staticmethod
    def _smooth_bump(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = np.abs(x) < 1.0
        out[mask] = np.exp(-1.0 / (1.0 - x[mask] ** 2))
        return out

    def _smooth_bump_prime(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = np.abs(x) < 1.0
        bump = self._smooth_bump(x[mask])
        derivative_inner = -2.0 * x[mask] / ((1.0 - x[mask] ** 2) ** 2)
        out[mask] = bump * derivative_inner
        return out

    def F(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        width = self.umax - self.umin
        t = np.clip((u - self.umin) / width, 0.0, 1.0)
        base = t**5 * (126.0 + t * (-420.0 + t * (540.0 + t * (-315.0 + t * 70.0))))

        total_bump = np.zeros_like(u, dtype=float)
        for i, center in enumerate(self.u_bump_centers):
            bump_mask = (u >= center - self.bump_width) & (u <= center + self.bump_width)
            if np.any(bump_mask):
                arg = (u[bump_mask] - center) / self.bump_width
                total_bump[bump_mask] += self.signs[i] * self.bump_amplitude * self._smooth_bump(arg)
        return clip01(base + total_bump)

    def Fprime(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        width = self.umax - self.umin
        t = (u - self.umin) / width
        inside = (t > 0.0) & (t < 1.0)
        up = np.zeros_like(u, dtype=float)
        if np.any(inside):
            t_in = t[inside]
            up[inside] = 630.0 * (t_in**4) * ((1.0 - t_in) ** 4) / width

        total_dbump = np.zeros_like(u, dtype=float)
        for i, center in enumerate(self.u_bump_centers):
            bump_mask = (u >= center - self.bump_width) & (u <= center + self.bump_width)
            if np.any(bump_mask):
                arg = (u[bump_mask] - center) / self.bump_width
                total_dbump[bump_mask] += (
                    self.signs[i] * self.bump_amplitude * self._smooth_bump_prime(arg) / self.bump_width
                )
        return np.maximum(up + total_dbump, 0.0)

    def phi(self, u: np.ndarray) -> np.ndarray:
        Fu = self.F(u)
        Fp = np.maximum(self.Fprime(u), 1e-4)
        return u - (1.0 - Fu) / Fp

    def optimal_price(self, u_level: np.ndarray, price_range: tuple[float, float], grid_n: int = 600) -> np.ndarray:
        u_level = np.asarray(u_level, dtype=float)
        p_grid = np.linspace(price_range[0], price_range[1], grid_n)
        U = u_level.reshape(-1, 1)
        rev = p_grid.reshape(1, -1) * (1.0 - self.F(p_grid.reshape(1, -1) - U))
        return p_grid[np.argmax(rev, axis=1)]


@dataclass
class SimConfig:
    T: int
    beta: float
    seed: int
    known_utility: bool
    T0: int = 100
    T0m: int = 400
    x_low: float = 0.35
    x_high: float = 0.65
    theta_true: float = 1.0
    P_low: float = 0.0
    P_high: float = 1.0
    degree: int = 2
    gridN: int = 301
    C_delta: float = 2.5
    C_v: float = 3.0
    kappa: float = 0.0
    band: float = 0.5
    baseline_band: float = 0.5
    baseline_stepsize: float = 0.35
    baseline_explore_c: float = 5.0
    dip_lambda: float = 0.1
    dip_ucb_c: float = 1.0 / 40.0
    dip_discretization_c: float = 20.0
    dip_init_exponent: int = 7


def pad_cdf_on_known_support(
    grid_u: np.ndarray,
    Fhat: np.ndarray,
    Fprime_hat: np.ndarray,
    support: tuple[float, float] = (-0.3, 0.3),
    min_density: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    F = np.asarray(Fhat, float).copy()
    Fp = np.asarray(Fprime_hat, float).copy()
    left = grid_u <= support[0]
    right = grid_u >= support[1]
    inside = (~left) & (~right)
    F[inside] = np.clip(F[inside], 0.0, 1.0)
    F[left] = 0.0
    F[right] = 1.0
    Fp = np.maximum(Fp, min_density)
    Fp[left | right] = min_density
    return F, Fp


def estimate_m_hat_ols_from_demand(
    T0m: int,
    rng: np.random.Generator,
    cfg: SimConfig,
    model: TrueModel,
) -> tuple[Callable[[np.ndarray], np.ndarray], float, float]:
    x = rng.uniform(cfg.x_low, cfg.x_high, size=T0m)
    p = rng.uniform(cfg.P_low, cfg.P_high, size=T0m)
    u_true = p - cfg.theta_true * x
    buy_prob = 1.0 - model.F(u_true)
    D = (rng.uniform(size=T0m) < buy_prob).astype(float)
    denom = float(x @ x) + 1e-12
    theta_hat = float((x @ D) / denom) * (cfg.P_high - cfg.P_low)

    def mhat(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return theta_hat * z

    grid = np.linspace(cfg.x_low, cfg.x_high, 400)
    sup_err = float(np.max(np.abs(theta_hat * grid - cfg.theta_true * grid)))
    return mhat, sup_err, theta_hat


def lpr_estimate_F_and_derivative(
    u: np.ndarray,
    D: np.ndarray,
    x_grid: np.ndarray,
    h: float,
    degree: int,
    ridge: float = 1e-8,
    support_filter: tuple[float, float] = (-0.3, 0.3),
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    D = np.asarray(D, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    mask = (u >= support_filter[0]) & (u <= support_filter[1])
    if np.sum(mask) > 10:
        u = u[mask]
        D = D[mask]

    q = degree
    G = len(x_grid)
    yhat = np.zeros(G, dtype=float)
    yprime = np.zeros(G, dtype=float)
    fact = np.array([math.factorial(k) for k in range(q + 1)], dtype=float)

    for gi, x0 in enumerate(x_grid):
        t = (u - x0) / (h + 1e-12)
        w = epanechnikov(t)
        mask_local = w > 0
        if np.sum(mask_local) < (q + 2):
            if np.sum(mask_local) == 0:
                yhat[gi] = np.mean(D) if len(D) else 0.5
                continue
            yhat[gi] = np.sum(w[mask_local] * D[mask_local]) / (np.sum(w[mask_local]) + 1e-12)
            continue

        tm = t[mask_local]
        wm = w[mask_local]
        Ym = D[mask_local]
        X = np.vstack([(tm**k) / fact[k] for k in range(q + 1)]).T
        XT_W = X.T * wm
        A = XT_W @ X
        A.flat[:: A.shape[0] + 1] += ridge
        theta = np.linalg.solve(A, XT_W @ Ym)
        yhat[gi] = theta[0]
        yprime[gi] = theta[1] / (h + 1e-12) if q >= 1 else 0.0

    Fhat = clip01(1.0 - yhat)
    Fprime_hat = np.maximum(-yprime, 1e-3)
    return Fhat, Fprime_hat


def alpha_rel(z: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    denom = (zmax - zmin) + 1e-12
    return np.minimum((z - zmin) / denom, (zmax - z) / denom)


def variable_bandwidth_smooth(grid_u: np.ndarray, phiI: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid_u), dtype=float)
    for i in range(len(grid_u)):
        dx = float(delta_x[i])
        t = (grid_u - grid_u[i]) / (dx + 1e-12)
        w = epanechnikov(t)
        s = float(w.sum())
        out[i] = phiI[i] if s <= 1e-12 else float(w @ phiI) / (s + 1e-12)
    return out


def apply_extrapolation_and_isotonic(grid_u: np.ndarray, phiS: np.ndarray, v: float, c1: float) -> np.ndarray:
    zmin, zmax = -0.2, -0.05
    v = float(np.clip(v, 0.0, 0.01))
    v1 = zmin + v * (zmax - zmin)
    v2 = zmin + (1.0 - v) * (zmax - zmin)
    i1 = int(np.clip(np.searchsorted(grid_u, v1), 0, len(grid_u) - 1))
    i2 = int(np.clip(np.searchsorted(grid_u, v2), 0, len(grid_u) - 1))
    out = phiS.copy()
    slope = float(max(c1, 1e-3)) / 2.0
    for i in range(0, i1):
        out[i] = out[i1] + slope * (grid_u[i] - grid_u[i1])
    for i in range(i2 + 1, len(grid_u)):
        out[i] = out[i2] + slope * (grid_u[i] - grid_u[i2])
    return pav_isotonic_increasing(out)


def build_phi_hat_from_data_paperstyle(
    u_obs: np.ndarray,
    D: np.ndarray,
    cfg: SimConfig,
    eps_m: float,
    stage_n: int,
) -> dict[str, np.ndarray]:
    umin = float(np.min(u_obs)) if len(u_obs) else -0.5
    umax = float(np.max(u_obs)) if len(u_obs) else 0.5
    pad = 0.02 * (umax - umin + 1e-12)
    zmin = umin - pad
    zmax = umax + pad
    grid_u = np.linspace(zmin, zmax, cfg.gridN)
    n = max(stage_n, 5)
    h = n ** (-1.0 / (2.0 * cfg.beta + 1.0)) * cfg.band
    Fhat, Fprime_hat = lpr_estimate_F_and_derivative(u_obs, D, grid_u, h, cfg.degree)
    Fhat, Fprime_hat = pad_cdf_on_known_support(grid_u, Fhat, Fprime_hat)
    phiI = grid_u - (1.0 - Fhat) / Fprime_hat
    logT = np.log(max(cfg.T, 3))
    rate = n ** (-(cfg.beta - 1.0) / (2.0 * cfg.beta + 1.0))
    alpha = np.maximum(alpha_rel(grid_u, zmin, zmax), 1e-6)
    delta_x = cfg.C_delta * rate * np.sqrt(logT / (alpha ** cfg.kappa)) + eps_m
    delta_x = np.maximum(delta_x / 100.0, 1e-4)
    phiS = variable_bandwidth_smooth(grid_u, phiI, delta_x)
    c1 = 1.0
    v = ((cfg.C_v**2) * rate * np.sqrt(logT)) ** (2.0 / (cfg.kappa + 2.0)) + cfg.C_v * eps_m
    phi_hat = apply_extrapolation_and_isotonic(grid_u, phiS, v=float(v), c1=c1)
    return {
        "grid_u": grid_u,
        "phi_hat": phi_hat,
        "phiI": phiI,
        "Fhat": Fhat,
        "Fprime_hat": Fprime_hat,
    }


def pricing_policy_from_phi(
    grid_u: np.ndarray,
    phi_grid: np.ndarray,
    m_vals: np.ndarray,
    price_range: tuple[float, float],
) -> np.ndarray:
    u_target = -np.asarray(m_vals, dtype=float)
    u_hat = invert_monotone_grid(grid_u, phi_grid, u_target)
    return np.clip(u_hat + m_vals, price_range[0], price_range[1])


class OneOffKernelEstimator:
    def __init__(self, u_obs: np.ndarray, D: np.ndarray, h: float) -> None:
        self.u_obs = np.asarray(u_obs, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.n = max(len(self.u_obs), 1)
        self.h = max(float(h), 1e-4)

    @staticmethod
    def kernel(x: np.ndarray) -> np.ndarray:
        mask = (np.abs(x) <= 1.0).astype(float)
        return (1.0 - 11.0 * x**2 / 3.0) * (1.0 - x**2) ** 3 * mask

    @staticmethod
    def kernel2(x: np.ndarray) -> np.ndarray:
        mask = (np.abs(x) <= 1.0).astype(float)
        return (-22.0 * x * (1.0 - x**2) ** 3 / 3.0 + (1.0 - 11.0 * x**2 / 3.0) * 3.0 * (1.0 - x**2) ** 2 * (-2.0 * x)) * mask

    @staticmethod
    def kernel3(x: np.ndarray) -> np.ndarray:
        mask = (np.abs(x) <= 1.0).astype(float)
        return (-4.0 * x * (1.0 - x**2) * (88.0 * x**3 / 3.0 - 40.0 * x / 3.0) + (1.0 - x**2) ** 2 * (88.0 * x**2 - 40.0 / 3.0)) * mask

    def _loc(self, t: float, deriv: int, weighted: bool) -> float:
        z = (self.u_obs - t) / self.h
        values = [self.kernel(z), self.kernel2(z), self.kernel3(z)][deriv]
        weights = self.D if weighted else 1.0
        scale = 1.0 / (self.n * self.h ** (deriv + 1))
        sign = -1.0 if deriv == 1 else 1.0
        return float(sign * scale * np.sum(values * weights))

    def whole1(self, t: float) -> float:
        den = self._loc(t, 0, False)
        return self._loc(t, 0, True) / max(den, 1e-6)

    def whole2(self, t: float) -> float:
        h1 = self._loc(t, 1, True)
        f0 = self._loc(t, 0, False)
        h0 = self._loc(t, 0, True)
        f1 = self._loc(t, 1, False)
        den = max(f0**2, 1e-6)
        out = (h1 * f0 - h0 * f1) / den
        if abs(out) < 1e-6:
            return 1e-6 if out >= 0 else -1e-6
        return float(out)

    def whole3(self, t: float) -> float:
        kh1 = self._loc(t, 1, True)
        kf0 = self._loc(t, 0, False)
        kh0 = self._loc(t, 0, True)
        kf1 = self._loc(t, 1, False)
        kh2 = self._loc(t, 2, True)
        kf2 = self._loc(t, 2, False)
        den = (kh1 * kf0 - kh0 * kf1) ** 2
        den = max(den, 1e-6)
        num = (kh1 * kf0) ** 2 - (kh0 * kf1) ** 2 - kh0 * kh2 * kf0**2 + kh0**2 * kf0 * kf2
        return float(num / den)

    def phi(self, t: float, target: float) -> float:
        return t + self.whole1(t) / self.whole2(t) + target

    def survival_hat(self, grid_u: np.ndarray) -> np.ndarray:
        return np.array([self.whole1(float(t)) for t in grid_u], dtype=float)

    def survival_prime_hat(self, grid_u: np.ndarray) -> np.ndarray:
        return np.array([self.whole2(float(t)) for t in grid_u], dtype=float)

    def base_phi_grid(self, grid_u: np.ndarray) -> np.ndarray:
        vals = np.empty(len(grid_u), dtype=float)
        for i, t in enumerate(grid_u):
            vals[i] = t + self.whole1(float(t)) / self.whole2(float(t))
        return vals

    def phi_p(self, t: float) -> float:
        return 1.0 + self.whole3(t)

    def root(self, target: float, init: float = -0.2, stepsize: float = 0.35) -> float:
        y = init
        x = (-1.7) * math.exp(y) / (1.0 + math.exp(y)) + 0.3
        for _ in range(250):
            phi_x = self.phi(x, target)
            if abs(phi_x) < 1e-4:
                break
            dxdy = (-1.7) * math.exp(y) / (1.0 + math.exp(y)) ** 2
            denom = self.phi_p(x) * dxdy
            if abs(denom) < 1e-6:
                break
            y = y - stepsize * phi_x / denom
            x = (-1.7) * math.exp(y) / (1.0 + math.exp(y)) + 0.3
            x = float(np.clip(x, -0.49, 0.29))
        return x


def regret_from_history(
    model: TrueModel,
    x_play: np.ndarray,
    p_play: np.ndarray,
    cfg: SimConfig,
) -> float:
    u_level = cfg.theta_true * x_play
    p_star = model.optimal_price(u_level, (cfg.P_low, cfg.P_high), grid_n=600)
    rev_star = p_star * (1.0 - model.F(p_star - u_level))
    rev_play = p_play * (1.0 - model.F(p_play - u_level))
    return float(np.sum(rev_star - rev_play))


def simulate_mymethod(cfg: SimConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    model = TrueModel(beta=cfg.beta)

    def m_true(x: np.ndarray) -> np.ndarray:
        return cfg.theta_true * np.asarray(x, dtype=float)

    if cfg.known_utility:
        mhat = lambda x: m_true(x)
        eps_m = 0.0
        theta_stage = 0
    else:
        mhat, eps_m, _ = estimate_m_hat_ols_from_demand(cfg.T0m, rng, cfg, model)
        theta_stage = cfg.T0m

    x_hist = np.zeros(cfg.T)
    p_hist = np.zeros(cfg.T)
    D_hist = np.zeros(cfg.T)
    t_ptr = 0

    if not cfg.known_utility:
        x = rng.uniform(cfg.x_low, cfg.x_high, size=cfg.T0m)
        p = rng.uniform(cfg.P_low, cfg.P_high, size=cfg.T0m)
        u_true = p - m_true(x)
        buy_prob = 1.0 - model.F(u_true)
        D = (rng.uniform(size=cfg.T0m) < buy_prob).astype(float)
        x_hist[: cfg.T0m] = x
        p_hist[: cfg.T0m] = p
        D_hist[: cfg.T0m] = D
        t_ptr += cfg.T0m

    for _ in range(cfg.T0):
        x = rng.uniform(cfg.x_low, cfg.x_high)
        p = rng.uniform(cfg.P_low, cfg.P_high)
        u_true = p - m_true(x)
        D = float(rng.uniform() < (1.0 - model.F(u_true)))
        x_hist[t_ptr] = x
        p_hist[t_ptr] = p
        D_hist[t_ptr] = D
        t_ptr += 1

    pack = build_phi_hat_from_data_paperstyle(p_hist[:t_ptr] - mhat(x_hist[:t_ptr]), D_hist[:t_ptr], cfg, eps_m, t_ptr)
    grid_u = pack["grid_u"]
    phi_hat = pack["phi_hat"]

    while t_ptr < cfg.T:
        x = rng.uniform(cfg.x_low, cfg.x_high)
        z_hat = invert_monotone_grid(grid_u, phi_hat, np.array([-mhat(x)]))[0]
        p = float(np.clip(z_hat + float(mhat(x)), cfg.P_low, cfg.P_high))
        u_true = p - m_true(x)
        D = float(rng.uniform() < (1.0 - model.F(u_true)))
        x_hist[t_ptr] = x
        p_hist[t_ptr] = p
        D_hist[t_ptr] = D
        t_ptr += 1

        t_eff = t_ptr - theta_stage
        if t_eff > 0 and (t_eff % cfg.T0 == 0):
            blk = t_eff // cfg.T0
            if blk >= 2 and (blk & (blk - 1)) == 0:
                pack = build_phi_hat_from_data_paperstyle(
                    p_hist[:t_ptr] - mhat(x_hist[:t_ptr]),
                    D_hist[:t_ptr],
                    cfg,
                    eps_m,
                    t_ptr,
                )
                grid_u = pack["grid_u"]
                phi_hat = pack["phi_hat"]

    start_theta = theta_stage
    start_phi = theta_stage + cfg.T0
    x_play = np.concatenate((x_hist[:start_theta], x_hist[start_phi:]))
    p_play = np.concatenate((p_hist[:start_theta], p_hist[start_phi:]))
    return regret_from_history(model, x_play, p_play, cfg)


def code_exploration_length(block_len: int, known_utility: bool, beta: float, c: float) -> int:
    if known_utility:
        return max(1, int(math.floor(c * (block_len ** 0.5))))
    exponent = (2.0 * beta + 1.0) / (4.0 * beta - 1.0)
    return max(1, int(math.floor(c * (block_len ** exponent))))


def simulate_code_methodology(cfg: SimConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    model = TrueModel(beta=cfg.beta)

    def m_true(x: np.ndarray) -> np.ndarray:
        return cfg.theta_true * np.asarray(x, dtype=float)

    regrets: list[float] = []
    block_base = 200
    total_steps = 0
    episode = 0

    while total_steps < cfg.T:
        block_len = min(block_base * (2**episode), cfg.T - total_steps)
        explore_len = min(code_exploration_length(block_len, cfg.known_utility, cfg.beta, cfg.baseline_explore_c), block_len)

        x_exp = rng.uniform(cfg.x_low, cfg.x_high, size=explore_len)
        p_exp = rng.uniform(cfg.P_low, cfg.P_high, size=explore_len)
        u_true_exp = p_exp - m_true(x_exp)
        buy_prob = 1.0 - model.F(u_true_exp)
        D_exp = (rng.uniform(size=explore_len) < buy_prob).astype(float)

        p_star_exp = model.optimal_price(m_true(x_exp), (cfg.P_low, cfg.P_high), grid_n=600)
        rev_star_exp = p_star_exp * (1.0 - model.F(p_star_exp - m_true(x_exp)))
        rev_play_exp = p_exp * (1.0 - model.F(p_exp - m_true(x_exp)))
        regrets.extend((rev_star_exp - rev_play_exp).tolist())

        if cfg.known_utility:
            mhat = lambda x: m_true(x)
            eps_m = 0.0
        else:
            denom = float(x_exp @ x_exp) + 1e-12
            theta_hat = float((x_exp @ D_exp) / denom) * (cfg.P_high - cfg.P_low)

            def mhat(x: np.ndarray, theta_hat: float = theta_hat) -> np.ndarray:
                return theta_hat * np.asarray(x, dtype=float)

            grid = np.linspace(cfg.x_low, cfg.x_high, 400)
            eps_m = float(np.max(np.abs(mhat(grid) - m_true(grid))))

        u_obs = p_exp - mhat(x_exp)
        pack = build_phi_hat_from_data_paperstyle(u_obs, D_exp, cfg, eps_m, explore_len)

        exploit_len = block_len - explore_len
        if exploit_len > 0:
            x_exploit = rng.uniform(cfg.x_low, cfg.x_high, size=exploit_len)
            mhat_vals = mhat(x_exploit)
            p_exploit = pricing_policy_from_phi(
                pack["grid_u"],
                pack["phi_hat"],
                mhat_vals,
                (cfg.P_low, cfg.P_high),
            )
            p_star = model.optimal_price(m_true(x_exploit), (cfg.P_low, cfg.P_high), grid_n=600)
            rev_star = p_star * (1.0 - model.F(p_star - m_true(x_exploit)))
            rev_play = p_exploit * (1.0 - model.F(p_exploit - m_true(x_exploit)))
            regrets.extend((rev_star - rev_play).tolist())

        total_steps += block_len
        episode += 1

    return float(np.sum(np.asarray(regrets[: cfg.T], dtype=float)))


def simulate_kernel_baseline(cfg: SimConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    model = TrueModel(beta=cfg.beta)

    def m_true(x: np.ndarray) -> np.ndarray:
        return cfg.theta_true * np.asarray(x, dtype=float)

    regrets: list[float] = []
    block_base = 200
    total_steps = 0
    episode = 0

    while total_steps < cfg.T:
        block_len = min(block_base * (2**episode), cfg.T - total_steps)
        explore_len = min(code_exploration_length(block_len, cfg.known_utility, cfg.beta, cfg.baseline_explore_c), block_len)

        x_exp = rng.uniform(cfg.x_low, cfg.x_high, size=explore_len)
        p_exp = rng.uniform(cfg.P_low, cfg.P_high, size=explore_len)
        u_true_exp = p_exp - m_true(x_exp)
        D_exp = (rng.uniform(size=explore_len) < (1.0 - model.F(u_true_exp))).astype(float)

        p_star_exp = model.optimal_price(m_true(x_exp), (cfg.P_low, cfg.P_high), grid_n=600)
        rev_star_exp = p_star_exp * (1.0 - model.F(p_star_exp - m_true(x_exp)))
        rev_play_exp = p_exp * (1.0 - model.F(p_exp - m_true(x_exp)))
        regrets.extend((rev_star_exp - rev_play_exp).tolist())

        if cfg.known_utility:
            theta_hat = cfg.theta_true
            mhat = lambda x: m_true(x)
        else:
            denom = float(x_exp @ x_exp) + 1e-12
            theta_hat = float((x_exp @ D_exp) / denom) * (cfg.P_high - cfg.P_low)
            mhat = lambda x, theta_hat=theta_hat: theta_hat * np.asarray(x, dtype=float)

        u_obs = p_exp - mhat(x_exp)
        h = cfg.baseline_band * max(explore_len, 5) ** (-1.0 / (2 * cfg.beta + 1))
        ker = OneOffKernelEstimator(u_obs, D_exp, h=h)
        umin = float(np.min(u_obs))
        umax = float(np.max(u_obs))
        pad = 0.05 * (umax - umin + 1e-12)
        grid_u = np.linspace(umin - pad, umax + pad, 401)
        phi_raw = ker.base_phi_grid(grid_u)
        phi_grid = phi_raw.copy()
        left_u, right_u = -0.15, 0.15
        mask = (grid_u >= left_u) & (grid_u <= right_u)
        if np.any(mask):
            phi_grid[mask] = pav_isotonic_increasing(phi_grid[mask])
            idx = np.where(mask)[0]
            left_idx = int(idx[0])
            right_idx = int(idx[-1])
            small_slope = 0.15
            for i in range(0, left_idx):
                phi_grid[i] = phi_grid[left_idx] + small_slope * (grid_u[i] - grid_u[left_idx])
            for i in range(right_idx + 1, len(grid_u)):
                phi_grid[i] = phi_grid[right_idx] + small_slope * (grid_u[i] - grid_u[right_idx])
            phi_grid = np.maximum.accumulate(phi_grid)

        exploit_len = block_len - explore_len
        if exploit_len > 0:
            x_exploit = rng.uniform(cfg.x_low, cfg.x_high, size=exploit_len)
            target_vals = -mhat(x_exploit)
            inv_mask = (grid_u >= -0.15) & (grid_u <= 0.15)
            if np.sum(inv_mask) >= 2:
                z_hat = invert_monotone_grid(grid_u[inv_mask], phi_grid[inv_mask], target_vals)
                z_hat = np.clip(z_hat, -0.15, 0.15)
            else:
                z_hat = invert_monotone_grid(grid_u, phi_grid, target_vals)
            p_exploit = np.clip(z_hat + mhat(x_exploit), cfg.P_low, cfg.P_high)
            p_star = model.optimal_price(m_true(x_exploit), (cfg.P_low, cfg.P_high), grid_n=600)
            rev_star = p_star * (1.0 - model.F(p_star - m_true(x_exploit)))
            rev_play = p_exploit * (1.0 - model.F(p_exploit - m_true(x_exploit)))
            regrets.extend((rev_star - rev_play).tolist())

        total_steps += block_len
        episode += 1

    return float(np.sum(np.asarray(regrets[: cfg.T], dtype=float)))


def _dip_episode_exponents(T: int, init_exponent: int) -> list[int]:
    exponents: list[int] = []
    total = 0
    j = init_exponent
    while total < T:
        exponents.append(j)
        total += 2**j
        j += 1
    return exponents


def _dip_theta_from_block(
    x_block: np.ndarray,
    p_block: np.ndarray,
    d_block: np.ndarray,
) -> tuple[float, float]:
    coef = fit_logistic_working_model(x_block, p_block, d_block)
    price_coef = float(coef[-1])
    if abs(price_coef) < 1e-6:
        price_coef = 1e-6 if price_coef >= 0 else -1e-6
    intercept = float(coef[0] / price_coef)
    theta_hat = float(coef[1] / price_coef)
    return intercept, theta_hat


def simulate_dip_policy(cfg: SimConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    model = TrueModel(beta=cfg.beta)

    def m_true(x: np.ndarray) -> np.ndarray:
        return cfg.theta_true * np.asarray(x, dtype=float)

    x_hist = np.zeros(cfg.T, dtype=float)
    p_hist = np.zeros(cfg.T, dtype=float)
    D_hist = np.zeros(cfg.T, dtype=float)
    t_ptr = 0

    init_len = min(2**cfg.dip_init_exponent, cfg.T)
    for _ in range(init_len):
        x = float(rng.uniform(cfg.x_low, cfg.x_high))
        p = float(rng.uniform(cfg.P_low, cfg.P_high))
        u_true = p - float(m_true(np.array([x]))[0])
        D = float(rng.uniform() < (1.0 - model.F(np.array([u_true]))[0]))
        x_hist[t_ptr] = x
        p_hist[t_ptr] = p
        D_hist[t_ptr] = D
        t_ptr += 1

    if cfg.known_utility:
        intercept_hat = 0.0
        theta_hat = cfg.theta_true
    else:
        intercept_hat, theta_hat = _dip_theta_from_block(x_hist[:t_ptr], p_hist[:t_ptr], D_hist[:t_ptr])

    for exponent in _dip_episode_exponents(cfg.T, cfg.dip_init_exponent):
        block_len = min(2**exponent, cfg.T - t_ptr)
        if block_len <= 0:
            break
        intv = max(2, int(cfg.dip_discretization_c * math.ceil(block_len ** (1.0 / 6.0))))
        me0 = np.zeros(intv, dtype=float)
        ti0 = np.zeros(intv, dtype=float)
        u1 = cfg.P_high
        u2 = abs(theta_hat) * cfg.x_high + abs(intercept_hat)
        u = u1 + 2.0 * u2
        ku = u / intv

        block_start = t_ptr
        for i in range(block_len):
            x = float(rng.uniform(cfg.x_low, cfg.x_high))
            cx = theta_hat * x + intercept_hat
            dex1 = int(math.floor((-cx + u2 + ku / 2.0) / ku) + 1)
            dex2 = int(math.floor((u1 - cx + u2 + ku / 2.0) / ku))
            dex1 = max(1, min(dex1, intv))
            dex2 = max(dex1, min(dex2, intv))
            num = dex2 - dex1 + 1
            rma = (2.0 * dex1 - 1.0) * ku / 2.0 - u2 + cx
            if i == 0:
                bc = int(rng.integers(1, num + 1))
            else:
                me = me0[dex1 - 1 : dex2]
                ti = ti0[dex1 - 1 : dex2]
                unseen = np.flatnonzero(ti <= 1e-12)
                if len(unseen) > 0:
                    bc = int(unseen[rng.integers(0, len(unseen))]) + 1
                else:
                    beta_t = cfg.dip_ucb_c * max(
                        1.0,
                        (
                            math.sqrt(cfg.dip_lambda * intv) / max(u1, 1e-8)
                            + math.sqrt(
                                2.0 * math.log(max(block_len, 2))
                                + intv
                                * math.log(
                                    (
                                        cfg.dip_lambda * intv
                                        + i * (u1**2)
                                    )
                                    / (cfg.dip_lambda * intv)
                                )
                            )
                        )
                        ** 2,
                    )
                    inde = np.empty(num, dtype=float)
                    for i1 in range(num):
                        inde[i1] = ((i1 * ku) + rma) * (me[i1] + math.sqrt(beta_t / (cfg.dip_lambda + ti[i1])))
                    best = np.flatnonzero(inde >= np.max(inde) - 1e-12)
                    bc = int(best[rng.integers(0, len(best))]) + 1

            p = float(np.clip((bc - 1) * ku + rma, cfg.P_low, cfg.P_high))
            u_true = p - float(m_true(np.array([x]))[0])
            D = float(rng.uniform() < (1.0 - model.F(np.array([u_true]))[0]))

            idx = dex1 - 1 + bc - 1
            me0[idx] = (me0[idx] * (cfg.dip_lambda + ti0[idx]) + p * D) / (
                cfg.dip_lambda + ti0[idx] + p**2
            )
            ti0[idx] += p**2

            x_hist[t_ptr] = x
            p_hist[t_ptr] = p
            D_hist[t_ptr] = D
            t_ptr += 1
            if t_ptr >= cfg.T:
                break

        if not cfg.known_utility:
            x_block = x_hist[block_start:t_ptr]
            p_block = p_hist[block_start:t_ptr]
            d_block = D_hist[block_start:t_ptr]
            if len(x_block) >= 8:
                intercept_hat, theta_hat = _dip_theta_from_block(x_block, p_block, d_block)

        if t_ptr >= cfg.T:
            break

    return regret_from_history(model, x_hist[:t_ptr], p_hist[:t_ptr], cfg)


def run_trials(
    T_values: list[int],
    beta_values: list[float],
    trials: int,
    known_utility: bool,
) -> list[dict[str, float | int | str]]:
    tasks: list[tuple[int, float, int, bool]] = []
    for T in T_values:
        for beta in beta_values:
            for seed in range(trials):
                tasks.append((T, beta, seed, known_utility))

    rows: dict[tuple[int, float, str], list[float]] = {}
    max_workers = min(6, max(1, (os.cpu_count() or 1) - 1))
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for T, beta, mymethod_regret, baseline_regret, dip_regret in ex.map(run_single_trial, tasks):
                rows.setdefault((T, beta, "mymethod"), []).append(mymethod_regret)
                rows.setdefault((T, beta, "kernel_baseline"), []).append(baseline_regret)
                rows.setdefault((T, beta, "dip"), []).append(dip_regret)
    except PermissionError:
        for task in tasks:
            T, beta, mymethod_regret, baseline_regret, dip_regret = run_single_trial(task)
            rows.setdefault((T, beta, "mymethod"), []).append(mymethod_regret)
            rows.setdefault((T, beta, "kernel_baseline"), []).append(baseline_regret)
            rows.setdefault((T, beta, "dip"), []).append(dip_regret)

    out: list[dict[str, float | int | str]] = []
    for (T, beta, method), regrets in sorted(rows.items()):
        out.append(
            {
                "scenario": "known" if known_utility else "unknown",
                "method": method,
                "T": T,
                "beta": beta,
                "avg_regret": float(np.mean(regrets)),
                "std_regret": float(np.std(regrets)),
                "n_trials": trials,
            }
        )
    return out


def run_single_trial(task: tuple[int, float, int, bool]) -> tuple[int, float, float, float, float]:
    T, beta, seed, known_utility = task
    T0m = 0 if known_utility else int(np.sqrt(T) * 4)
    cfg = SimConfig(T=T, T0=100, T0m=T0m, beta=beta, seed=seed, known_utility=known_utility)
    mymethod_regret = simulate_mymethod(cfg)
    baseline_regret = simulate_kernel_baseline(cfg)
    dip_regret = simulate_dip_policy(cfg)
    scenario = "known" if known_utility else "unknown"
    print(
        f"[done] scenario={scenario} T={T} beta={beta} seed={seed} "
        f"mymethod={mymethod_regret:.6f} baseline={baseline_regret:.6f} dip={dip_regret:.6f}",
        flush=True,
    )
    return T, beta, mymethod_regret, baseline_regret, dip_regret


def _group_rows(rows: list[dict[str, float | int | str]], scenario: str, method: str) -> list[dict[str, float]]:
    grouped: dict[int, list[tuple[float, float, int]]] = {}
    for row in rows:
        row_method = row["method"]
        if row_method == "notebook":
            row_method = "mymethod"
        if row["scenario"] != scenario or row_method != method:
            continue
        T = int(row["T"])
        grouped.setdefault(T, []).append((float(row["avg_regret"]), float(row["std_regret"]), int(row["n_trials"])))
    out = []
    for T in sorted(grouped):
        vals = grouped[T]
        out.append(
            {
                "T": float(T),
                "avg_regret": float(np.mean([v[0] for v in vals])),
                "stderr": float(np.mean([v[1] / np.sqrt(max(v[2], 1)) for v in vals])),
            }
        )
    return out


def _svg_polyline(points: list[tuple[float, float]], color: str, width: int = 3) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{coords}" />'


def _svg_polygon(points: list[tuple[float, float]], color: str, opacity: float = 0.18) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon fill="{color}" fill-opacity="{opacity}" stroke="none" points="{coords}" />'


def _scale(vals: list[float], lo: float, hi: float, log_scale: bool) -> list[float]:
    arr = np.asarray(vals, dtype=float)
    if log_scale:
        arr = np.log(np.maximum(arr, 1e-6))
        lo = math.log(max(lo, 1e-6))
        hi = math.log(max(hi, 1e-6))
    denom = max(hi - lo, 1e-12)
    return ((arr - lo) / denom).tolist()


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def plot_comparison(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    width = 900
    height = 620
    panel_w = 690
    panel_h = 430
    margin_left = 90
    margin_top = 110
    gap_x = 90
    colors = {"mymethod": "#1f77b4", "kernel_baseline": "#d62728", "dip": "#2a9d8f"}
    labels = {"mymethod": "ILPR", "kernel_baseline": "Kernel-based policy", "dip": "DIP"}

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]

    legend_x = 120
    legend_y = 68
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="680" height="34" fill="white" stroke="#444" />')
    legend_positions = {
        "mymethod": legend_x + 18,
        "kernel_baseline": legend_x + 195,
        "dip": legend_x + 545,
    }
    for method in ["mymethod", "kernel_baseline", "dip"]:
        x = legend_positions[method]
        y = legend_y + 18
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x + 38}" y2="{y}" stroke="{colors[method]}" stroke-width="4" />')
        svg.append(f'<circle cx="{x + 19}" cy="{y}" r="4.5" fill="{colors[method]}" />')
        svg.append(f'<text x="{x + 50}" y="{y + 5}" font-size="17" font-family="Helvetica">{labels[method]}</text>')

    for col_idx, scenario in enumerate(["unknown"]):
            x0 = margin_left + col_idx * (panel_w + gap_x)
            y0 = margin_top
            plot_x0 = x0 + 55
            plot_y0 = y0 + 25
            plot_w = panel_w - 95
            plot_h = panel_h - 85

            grouped = {m: _group_rows(rows, scenario, m) for m in ["mymethod", "kernel_baseline", "dip"]}
            all_x = [pt["T"] for pts in grouped.values() for pt in pts]
            all_y = [max(pt["avg_regret"] + pt["stderr"], 1e-6) for pts in grouped.values() for pt in pts]
            xmin = 0.0
            xmax = max(all_x)
            ymin = 0.0
            raw_ymax = max(all_y)
            y_step = _nice_step(raw_ymax / 4.0)
            ymax = y_step * 4.0
            y_ticks = [0.0, y_step, 2.0 * y_step, 3.0 * y_step, 4.0 * y_step]
            x_ticks = np.linspace(xmin, xmax, 5)

            svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1.5" />')
            svg.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#333" />')
            svg.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#333" />')

            for x_tick in x_ticks[1:-1]:
                gx = plot_x0 + (x_tick - xmin) / max(xmax - xmin, 1e-12) * plot_w
                svg.append(f'<line x1="{gx:.2f}" y1="{plot_y0}" x2="{gx:.2f}" y2="{plot_y0 + plot_h}" stroke="#ddd" stroke-dasharray="4 4" />')
            for y_tick in y_ticks[1:-1]:
                gy = plot_y0 + plot_h - (y_tick - ymin) / max(ymax - ymin, 1e-12) * plot_h
                svg.append(f'<line x1="{plot_x0}" y1="{gy:.2f}" x2="{plot_x0 + plot_w}" y2="{gy:.2f}" stroke="#ddd" stroke-dasharray="4 4" />')

            for method in ["mymethod", "kernel_baseline", "dip"]:
                pts = grouped[method]
                x_vals = [pt["T"] for pt in pts]
                y_vals = [pt["avg_regret"] for pt in pts]
                lower = [max(pt["avg_regret"] - pt["stderr"], 1e-6) for pt in pts]
                upper = [pt["avg_regret"] + pt["stderr"] for pt in pts]
                xs = _scale(x_vals, xmin, xmax, False)
                ys = _scale(y_vals, ymin, ymax, False)
                lows = _scale(lower, ymin, ymax, False)
                highs = _scale(upper, ymin, ymax, False)

                line_pts = []
                upper_pts = []
                lower_pts = []
                for sx, sy, sl, sh in zip(xs, ys, lows, highs):
                    px = plot_x0 + sx * plot_w
                    py = plot_y0 + plot_h - sy * plot_h
                    pl = plot_y0 + plot_h - sl * plot_h
                    ph = plot_y0 + plot_h - sh * plot_h
                    line_pts.append((px, py))
                    upper_pts.append((px, ph))
                    lower_pts.append((px, pl))

                band_pts = upper_pts + list(reversed(lower_pts))
                svg.append(_svg_polygon(band_pts, colors[method]))
                svg.append(_svg_polyline(line_pts, colors[method]))
                for px, py in line_pts:
                    svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{colors[method]}" />')

            svg.append(f'<text x="{x0 + panel_w/2:.1f}" y="{y0 + panel_h - 10}" text-anchor="middle" font-size="18" font-family="Helvetica">T</text>')
            svg.append(
                f'<text x="{x0 - 36}" y="{y0 + panel_h/2:.1f}" text-anchor="middle" font-size="18" font-family="Helvetica" transform="rotate(-90 {x0 - 36} {y0 + panel_h/2:.1f})">Regret</text>'
            )

            for x_val in x_ticks:
                tx = plot_x0 + (x_val - xmin) / max(xmax - xmin, 1e-12) * plot_w
                x_label = int(round(x_val / 1000.0) * 1000) if x_val > 0 else 0
                svg.append(f'<text x="{tx:.2f}" y="{plot_y0 + plot_h + 24}" text-anchor="middle" font-size="14" font-family="Helvetica">{x_label}</text>')
            for y_val in y_ticks:
                ty = plot_y0 + plot_h - (y_val - ymin) / max(ymax - ymin, 1e-12) * plot_h
                y_label = int(round(y_val))
                svg.append(f'<text x="{plot_x0 - 10}" y="{ty + 5:.2f}" text-anchor="end" font-size="14" font-family="Helvetica">{y_label}</text>')

    svg.append("</svg>")
    (output_dir / "simulation_comparison.svg").write_text("\n".join(svg), encoding="utf-8")


def parse_csv_numbers(raw: str, cast: Callable[[str], float | int]) -> list[float | int]:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]
