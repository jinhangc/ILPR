from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = WORKSPACE_ROOT / "data/competition_data_2023_09_25-2.csv"


def clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def parse_csv_numbers(text: str, caster: Callable[[str], float | int]) -> list[float | int]:
    return [caster(part.strip()) for part in text.split(",") if part.strip()]


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


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    return pav_isotonic_increasing(np.asarray(y, dtype=float)[::-1])[::-1]


def monotone_decreasing_projection(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.maximum.accumulate(y[::-1])[::-1]


def invert_monotone_grid(x_grid: np.ndarray, y_grid: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    y_query = np.asarray(y_query, dtype=float)
    y_mono = np.maximum.accumulate(y_grid)
    yq = np.clip(y_query, y_mono[0], y_mono[-1])
    return np.interp(yq, y_mono, x_grid)


@dataclass
class CalibratedSKU:
    sku: str
    feature_names: list[str]
    contexts: np.ndarray
    min_price: np.ndarray
    max_price: np.ndarray
    price_mean: float
    price_std: float
    bounds_low: float
    bounds_high: float
    m_coef: np.ndarray
    m_intercept: float
    u_grid: np.ndarray
    surv_grid: np.ndarray
    demand_rate: float
    n_obs: int

    def _oracle_tail_end(self) -> float:
        span = max(float(self.u_grid[-1] - self.u_grid[0]), 1e-2)
        return float(self.u_grid[-1] + span)

    def m_true(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.m_intercept + x @ self.m_coef

    def survival(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        tail_u = self._oracle_tail_end()
        grid = np.append(self.u_grid, tail_u)
        surv = np.append(self.surv_grid, 0.0)
        return clip01(np.interp(u, grid, surv, left=self.surv_grid[0], right=0.0))

    def demand_probability(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.survival(np.asarray(p, dtype=float) - self.m_true(x))

    def optimal_price(self, x: np.ndarray, p_low: np.ndarray, p_high: np.ndarray, grid_n: int = 240) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p_low = np.asarray(p_low, dtype=float)
        out = np.empty(len(x), dtype=float)
        u_high = self._oracle_tail_end()
        for i in range(len(x)):
            p_upper = max(float(p_low[i]) + 1e-6, float(self.m_true(x[i])) + u_high)
            grid = np.linspace(p_low[i], p_upper, grid_n)
            surv = self.survival(grid - self.m_true(x[i]))
            out[i] = grid[int(np.argmax(grid * surv))]
        return out


def _safe_float(text: str, default: float | None = None) -> float | None:
    value = text.strip()
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_date_weekday(text: str) -> int:
    return datetime.strptime(text, "%m/%d/%Y").weekday()


def _fill_missing_with_median(values: list[float | None], fallback: float) -> np.ndarray:
    observed = np.array([v for v in values if v is not None], dtype=float)
    median = float(np.median(observed)) if len(observed) else fallback
    return np.array([median if v is None else float(v) for v in values], dtype=float)


def _fit_ridge_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, max_iter: int = 80) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    eye = np.eye(X.shape[1], dtype=float)
    eye[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-5, None)
        z = eta + (y - p) / w
        WX = X * w[:, None]
        A = X.T @ WX + ridge * eye
        b = X.T @ (w * z)
        beta_new = np.linalg.solve(A, b)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _fit_ridge_logistic_with_offset(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    ridge: float = 1.0,
    max_iter: int = 80,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    eye = np.eye(X.shape[1], dtype=float)
    eye[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(offset + X @ beta, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-5, None)
        z = eta + (y - p) / w
        target = z - offset
        WX = X * w[:, None]
        A = X.T @ WX + ridge * eye
        b = X.T @ (w * target)
        beta_new = np.linalg.solve(A, b)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _gaussian_smooth_on_grid(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    bandwidth = max(float(bandwidth), 1e-8)
    dist = (grid[:, None] - grid[None, :]) / bandwidth
    weights = np.exp(-0.5 * dist**2)
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-12
    return weights @ values


def _calibrate_survival(u: np.ndarray, D: np.ndarray, grid_n: int = 401) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(u)
    u_sorted = np.asarray(u, dtype=float)[order]
    d_sorted = np.asarray(D, dtype=float)[order]
    s_sorted = isotonic_decreasing(d_sorted)
    if len(u_sorted) == 1:
        grid = np.array([u_sorted[0] - 1e-3, u_sorted[0] + 1e-3], dtype=float)
        surv = np.array([s_sorted[0], s_sorted[0]], dtype=float)
        return grid, clip01(surv)
    pad = 0.05 * max(float(u_sorted[-1] - u_sorted[0]), 1e-2)
    grid = np.linspace(float(u_sorted[0] - pad), float(u_sorted[-1] + pad), grid_n)
    surv = np.interp(grid, u_sorted, s_sorted, left=s_sorted[0], right=s_sorted[-1])

    # Build a smooth monotone semi-synthetic truth instead of using the raw
    # isotonic step fit directly.
    span = max(float(grid[-1] - grid[0]), 1e-2)
    grid_step = span / max(grid_n - 1, 1)
    bandwidth = max(0.9 * span * (len(u_sorted) ** (-1.0 / 5.0)), 2.0 * grid_step)
    surv = _gaussian_smooth_on_grid(surv, grid, bandwidth)
    surv = _gaussian_smooth_on_grid(surv, grid, bandwidth)
    surv = monotone_decreasing_projection(surv)
    return grid, clip01(surv)


def load_calibrated_skus(
    dataset_path: str | Path = DEFAULT_DATASET,
    min_rows: int = 300,
    max_skus: int | None = None,
    max_purchase_rate: float = 0.95,
    max_upper_bound_oracle_share: float = 1.0,
) -> list[CalibratedSKU]:
    dataset_path = Path(dataset_path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sku = row["sku"].strip()
            if sku:
                groups[sku].append(row)

    ordered_skus = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    calibrated_candidates: list[tuple[float, float, CalibratedSKU]] = []

    for sku, rows in ordered_skus:
        if len(rows) < min_rows:
            continue

        weekdays: list[int] = []
        comp_min_raw: list[float | None] = []
        comp_max_raw: list[float | None] = []
        stock_raw: list[float | None] = []
        prices: list[float] = []
        units: list[float] = []
        min_price: list[float] = []
        max_price: list[float] = []

        bad = False
        for row in rows:
            salesdate = row["salesdate"].strip()
            price = _safe_float(row["price"], None)
            unitsordered = _safe_float(row["unitsordered"], None)
            p_low = _safe_float(row["min_price"], None)
            p_high = _safe_float(row["max_price"], None)
            if not salesdate or price is None or unitsordered is None or p_low is None or p_high is None:
                bad = True
                break
            if p_high <= p_low:
                bad = True
                break
            weekdays.append(_parse_date_weekday(salesdate))
            comp_min_raw.append(_safe_float(row["comp_data_min_price"], price))
            comp_max_raw.append(_safe_float(row["comp_data_max_price"], price))
            stock_raw.append(_safe_float(row["managed_fba_stock_level"], 0.0))
            prices.append(price)
            units.append(unitsordered)
            min_price.append(p_low)
            max_price.append(p_high)
        if bad:
            continue

        y = (np.asarray(units, dtype=float) > 0.0).astype(float)
        if y.mean() < 0.05 or y.mean() > max_purchase_rate:
            continue

        price_arr = np.asarray(prices, dtype=float)
        min_price_arr = np.asarray(min_price, dtype=float)
        max_price_arr = np.asarray(max_price, dtype=float)
        comp_min_arr = _fill_missing_with_median(comp_min_raw, float(np.median(price_arr)))
        comp_max_arr = _fill_missing_with_median(comp_max_raw, float(np.median(price_arr)))
        stock_arr = _fill_missing_with_median(stock_raw, 0.0)

        price_mean = float(np.mean(price_arr))
        price_std = float(np.std(price_arr)) + 1e-6
        comp_min_norm = (comp_min_arr - price_mean) / price_std
        comp_max_norm = (comp_max_arr - price_mean) / price_std
        stock_norm = (stock_arr - float(np.mean(stock_arr))) / (float(np.std(stock_arr)) + 1e-6)

        X = np.zeros((len(rows), 9), dtype=float)
        for i, wd in enumerate(weekdays):
            if wd < 6:
                X[i, wd] = 1.0
        X[:, 6] = comp_min_norm
        X[:, 7] = comp_max_norm
        X[:, 8] = stock_norm

        # Estimate the price slope instead of fixing it to -1; otherwise the
        # calibrated oracle often degenerates to the upper price bound.
        Z = np.column_stack([np.ones(len(rows), dtype=float), X, -price_arr])
        coef = _fit_ridge_logistic(Z, y, ridge=1.0)
        price_coef = float(coef[-1])
        if abs(price_coef) < 1e-6:
            price_coef = 1e-6 if price_coef >= 0 else -1e-6
        m_intercept = float(coef[0] / price_coef)
        m_coef = np.asarray(coef[1:-1], dtype=float) / price_coef
        u = price_arr - (m_intercept + X @ m_coef)
        u_grid, surv_grid = _calibrate_survival(u, y)

        calibrated_sku = CalibratedSKU(
            sku=sku,
            feature_names=[
                "dow_mon",
                "dow_tue",
                "dow_wed",
                "dow_thu",
                "dow_fri",
                "dow_sat",
                "comp_min_norm",
                "comp_max_norm",
                "stock_norm",
            ],
            contexts=X,
            min_price=min_price_arr,
            max_price=max_price_arr,
            price_mean=price_mean,
            price_std=price_std,
            bounds_low=float(np.min(min_price_arr)),
            bounds_high=float(np.max(max_price_arr)),
            m_coef=m_coef,
            m_intercept=m_intercept,
            u_grid=u_grid,
            surv_grid=surv_grid,
            demand_rate=float(np.mean(y)),
            n_obs=len(rows),
        )
        oracle_prices = calibrated_sku.optimal_price(calibrated_sku.contexts, calibrated_sku.min_price, calibrated_sku.max_price)
        upper_share = float(np.mean(np.isclose(oracle_prices, calibrated_sku.max_price, atol=1e-9)))
        if upper_share > max_upper_bound_oracle_share:
            continue

        calibrated_candidates.append((upper_share, abs(calibrated_sku.demand_rate - 0.5), calibrated_sku))

    calibrated_candidates.sort(key=lambda item: (item[0], item[1], -item[2].n_obs, item[2].sku))
    calibrated = [item[2] for item in calibrated_candidates]
    if max_skus is not None:
        calibrated = calibrated[:max_skus]

    if not calibrated:
        raise ValueError(f"No usable products found in {dataset_path}")
    return calibrated


@dataclass
class SimConfigReal:
    T: int
    seed: int
    sku: CalibratedSKU
    T0: int = 80
    T0m: int = 200
    degree: int = 2
    gridN: int = 301
    beta: float = 2.0
    C_delta: float = 2.5
    C_v: float = 3.0
    kappa: float = 0.0
    band: float = 0.6
    baseline_band: float = 0.6
    baseline_stepsize: float = 0.35
    baseline_explore_c: float = 4.0
    dip_lambda: float = 0.1
    dip_ucb_c: float = 1.0 / 40.0
    dip_discretization_c: float = 20.0
    dip_init_exponent: int = 7


class SemiSyntheticEnv:
    def __init__(self, sku: CalibratedSKU, rng: np.random.Generator) -> None:
        self.sku = sku
        self.rng = rng

    def draw(self, n: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.sku.n_obs, size=n)
        return (
            self.sku.contexts[idx],
            self.sku.min_price[idx],
            self.sku.max_price[idx],
        )

    def sample_demand(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        prob = self.sku.demand_probability(x, p)
        return (self.rng.uniform(size=len(prob)) < prob).astype(float)


def lpr_estimate_F_and_derivative(
    u: np.ndarray,
    D: np.ndarray,
    x_grid: np.ndarray,
    h: float,
    degree: int,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    D = np.asarray(D, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    q = degree
    fact = np.array([math.factorial(k) for k in range(q + 1)], dtype=float)
    yhat = np.zeros(len(x_grid), dtype=float)
    yprime = np.zeros(len(x_grid), dtype=float)

    for gi, x0 in enumerate(x_grid):
        t = (u - x0) / (h + 1e-12)
        w = epanechnikov(t)
        mask = w > 0
        if np.sum(mask) < q + 2:
            if np.sum(mask) == 0:
                yhat[gi] = float(np.mean(D)) if len(D) else 0.5
            else:
                yhat[gi] = float(np.sum(w[mask] * D[mask]) / (np.sum(w[mask]) + 1e-12))
            continue
        tm = t[mask]
        wm = w[mask]
        Ym = D[mask]
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
    lower = float(np.quantile(grid_u, 0.15))
    upper = float(np.quantile(grid_u, 0.85))
    v = float(np.clip(v, 0.0, 0.03))
    v1 = lower + v * (upper - lower)
    v2 = lower + (1.0 - v) * (upper - lower)
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
    cfg: SimConfigReal,
    eps_m: float,
    stage_n: int,
) -> dict[str, np.ndarray]:
    umin = float(np.min(u_obs))
    umax = float(np.max(u_obs))
    pad = 0.03 * (umax - umin + 1e-12)
    zmin = umin - pad
    zmax = umax + pad
    grid_u = np.linspace(zmin, zmax, cfg.gridN)
    n = max(stage_n, 5)
    h = n ** (-1.0 / (2.0 * cfg.beta + 1.0)) * cfg.band
    Fhat, Fprime_hat = lpr_estimate_F_and_derivative(u_obs, D, grid_u, h, cfg.degree)
    phiI = grid_u - (1.0 - Fhat) / Fprime_hat
    logT = np.log(max(cfg.T, 3))
    rate = n ** (-(cfg.beta - 1.0) / (2.0 * cfg.beta + 1.0))
    alpha = np.maximum(alpha_rel(grid_u, zmin, zmax), 1e-6)
    delta_x = cfg.C_delta * rate * np.sqrt(logT / (alpha ** cfg.kappa)) + eps_m
    delta_x = np.maximum(delta_x / 100.0, 1e-4)
    phiS = variable_bandwidth_smooth(grid_u, phiI, delta_x)
    v = ((cfg.C_v**2) * rate * np.sqrt(logT)) ** (2.0 / (cfg.kappa + 2.0)) + cfg.C_v * eps_m
    phi_hat = apply_extrapolation_and_isotonic(grid_u, phiS, v=float(v), c1=1.0)
    return {"grid_u": grid_u, "phi_hat": phi_hat}


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
        return (
            -22.0 * x * (1.0 - x**2) ** 3 / 3.0
            + (1.0 - 11.0 * x**2 / 3.0) * 3.0 * (1.0 - x**2) ** 2 * (-2.0 * x)
        ) * mask

    @staticmethod
    def kernel3(x: np.ndarray) -> np.ndarray:
        mask = (np.abs(x) <= 1.0).astype(float)
        return (
            -4.0 * x * (1.0 - x**2) * (88.0 * x**3 / 3.0 - 40.0 * x / 3.0)
            + (1.0 - x**2) ** 2 * (88.0 * x**2 - 40.0 / 3.0)
        ) * mask

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

    def phi_p(self, t: float) -> float:
        return 1.0 + self.whole3(t)

    def root(self, target: float, init: float = -0.2, stepsize: float = 0.35) -> float:
        y = init
        ey = math.exp(float(np.clip(y, -40.0, 40.0)))
        x = (-1.7) * ey / (1.0 + ey) + 0.3
        for _ in range(250):
            phi_x = self.phi(x, target)
            if abs(phi_x) < 1e-4:
                break
            ey = math.exp(float(np.clip(y, -40.0, 40.0)))
            dxdy = (-1.7) * ey / (1.0 + ey) ** 2
            denom = self.phi_p(x) * dxdy
            if abs(denom) < 1e-6:
                break
            y = y - stepsize * phi_x / denom
            ey = math.exp(float(np.clip(y, -40.0, 40.0)))
            x = (-1.7) * ey / (1.0 + ey) + 0.3
        return x


def _fit_m_hat_from_history(
    x: np.ndarray,
    p: np.ndarray,
    D: np.ndarray,
    ridge: float = 1.0,
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    X = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    D = np.asarray(D, dtype=float)
    X_model = np.column_stack([np.ones(len(X)), X])
    beta = _fit_ridge_logistic_with_offset(X_model, D, offset=-p, ridge=ridge)
    intercept = float(beta[0])
    coef = np.asarray(beta[1:], dtype=float)

    def mhat(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return intercept + z @ coef

    return mhat, float(np.linalg.norm(coef))


def _fit_dip_linear_utility(
    x: np.ndarray,
    p: np.ndarray,
    D: np.ndarray,
    ridge: float = 1.0,
) -> tuple[float, np.ndarray]:
    X = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1, 1)
    D = np.asarray(D, dtype=float)
    Z = np.column_stack([np.ones(len(X), dtype=float), X, -p[:, 0]])
    coef = _fit_ridge_logistic(Z, D, ridge=ridge)
    price_coef = float(coef[-1])
    if abs(price_coef) < 1e-6:
        price_coef = 1e-6 if price_coef >= 0 else -1e-6
    intercept = float(coef[0] / price_coef)
    utility_coef = np.asarray(coef[1:-1], dtype=float) / price_coef
    return intercept, utility_coef


def revenue_stats_from_history(
    sku: CalibratedSKU,
    x_play: np.ndarray,
    p_play: np.ndarray,
    p_low_play: np.ndarray,
    p_high_play: np.ndarray,
) -> tuple[float, float, float]:
    p_star = sku.optimal_price(x_play, p_low_play, p_high_play)
    rev_star = p_star * sku.demand_probability(x_play, p_star)
    rev_play = p_play * sku.demand_probability(x_play, p_play)
    oracle = float(np.sum(rev_star))
    realized = float(np.sum(rev_play))
    regret = oracle - realized
    return regret, oracle, realized


def _prefix_stats(
    sku: CalibratedSKU,
    x_hist: np.ndarray,
    p_hist: np.ndarray,
    lo_hist: np.ndarray,
    hi_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_star = sku.optimal_price(x_hist, lo_hist, hi_hist)
    rev_star = p_star * sku.demand_probability(x_hist, p_star)
    rev_play = p_hist * sku.demand_probability(x_hist, p_hist)
    cum_oracle = np.cumsum(rev_star)
    cum_realized = np.cumsum(rev_play)
    cum_regret = cum_oracle - cum_realized
    return cum_regret, cum_oracle, cum_realized


def simulate_mymethod_path(cfg: SimConfigReal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    env = SemiSyntheticEnv(cfg.sku, rng)

    x_hist = np.zeros((cfg.T, cfg.sku.contexts.shape[1]), dtype=float)
    p_hist = np.zeros(cfg.T, dtype=float)
    d_hist = np.zeros(cfg.T, dtype=float)
    lo_hist = np.zeros(cfg.T, dtype=float)
    hi_hist = np.zeros(cfg.T, dtype=float)
    t_ptr = 0

    t0m = min(cfg.T0m, cfg.T)
    if t0m <= 0:
        zeros = np.zeros(cfg.T, dtype=float)
        return zeros, zeros, zeros

    x0, lo0, hi0 = env.draw(t0m)
    p0 = rng.uniform(lo0, hi0)
    d0 = env.sample_demand(x0, p0)
    x_hist[:t0m] = x0
    p_hist[:t0m] = p0
    d_hist[:t0m] = d0
    lo_hist[:t0m] = lo0
    hi_hist[:t0m] = hi0
    t_ptr = t0m

    mhat, coef_norm = _fit_m_hat_from_history(x_hist[:t_ptr], p_hist[:t_ptr], d_hist[:t_ptr])
    eps_m = 0.05 * coef_norm / math.sqrt(max(t0m, 1))

    t0 = min(cfg.T0, cfg.T - t_ptr)
    if t0 > 0:
        x1, lo1, hi1 = env.draw(t0)
        p1 = rng.uniform(lo1, hi1)
        d1 = env.sample_demand(x1, p1)
        x_hist[t_ptr : t_ptr + t0] = x1
        p_hist[t_ptr : t_ptr + t0] = p1
        d_hist[t_ptr : t_ptr + t0] = d1
        lo_hist[t_ptr : t_ptr + t0] = lo1
        hi_hist[t_ptr : t_ptr + t0] = hi1
        t_ptr += t0

    pack = build_phi_hat_from_data_paperstyle(p_hist[:t_ptr] - mhat(x_hist[:t_ptr]), d_hist[:t_ptr], cfg, eps_m, t_ptr)
    grid_u = pack["grid_u"]
    phi_hat = pack["phi_hat"]

    while t_ptr < cfg.T:
        x, lo, hi = env.draw(1)
        x_row = x[0]
        target = np.array([-mhat(x_row.reshape(1, -1))[0]])
        z_hat = invert_monotone_grid(grid_u, phi_hat, target)[0]
        p = float(np.clip(z_hat + target[0] * -1.0, lo[0], hi[0]))
        d = float(env.sample_demand(x, np.array([p]))[0])
        x_hist[t_ptr] = x_row
        p_hist[t_ptr] = p
        d_hist[t_ptr] = d
        lo_hist[t_ptr] = lo[0]
        hi_hist[t_ptr] = hi[0]
        t_ptr += 1

        if t_ptr % cfg.T0 == 0:
            blk = t_ptr // cfg.T0
            if blk >= 2 and (blk & (blk - 1)) == 0:
                mhat, coef_norm = _fit_m_hat_from_history(x_hist[:t_ptr], p_hist[:t_ptr], d_hist[:t_ptr])
                eps_m = 0.05 * coef_norm / math.sqrt(max(t_ptr, 1))
                pack = build_phi_hat_from_data_paperstyle(
                    p_hist[:t_ptr] - mhat(x_hist[:t_ptr]),
                    d_hist[:t_ptr],
                    cfg,
                    eps_m,
                    t_ptr,
                )
                grid_u = pack["grid_u"]
                phi_hat = pack["phi_hat"]

    return _prefix_stats(cfg.sku, x_hist, p_hist, lo_hist, hi_hist)


def code_exploration_length(block_len: int, c: float) -> int:
    return max(1, int(math.floor(c * (block_len ** 0.5))))


def simulate_kernel_baseline_path(cfg: SimConfigReal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    env = SemiSyntheticEnv(cfg.sku, rng)

    x_hist = np.zeros((cfg.T, cfg.sku.contexts.shape[1]), dtype=float)
    p_hist = np.zeros(cfg.T, dtype=float)
    lo_hist = np.zeros(cfg.T, dtype=float)
    hi_hist = np.zeros(cfg.T, dtype=float)
    block_base = 160
    total_steps = 0
    episode = 0

    while total_steps < cfg.T:
        block_len = min(block_base * (2**episode), cfg.T - total_steps)
        explore_len = min(code_exploration_length(block_len, cfg.baseline_explore_c), block_len)
        x_exp, lo_exp, hi_exp = env.draw(explore_len)
        p_exp = rng.uniform(lo_exp, hi_exp)
        d_exp = env.sample_demand(x_exp, p_exp)

        mhat, _ = _fit_m_hat_from_history(x_exp, p_exp, d_exp)
        u_obs = p_exp - mhat(x_exp)
        h = cfg.baseline_band * max(explore_len, 5) ** (-1.0 / (2.0 * cfg.beta + 1.0))
        ker = OneOffKernelEstimator(u_obs, d_exp, h)

        x_hist[total_steps : total_steps + explore_len] = x_exp
        p_hist[total_steps : total_steps + explore_len] = p_exp
        lo_hist[total_steps : total_steps + explore_len] = lo_exp
        hi_hist[total_steps : total_steps + explore_len] = hi_exp

        exploit_len = block_len - explore_len
        if exploit_len > 0:
            x_exploit, lo_exploit, hi_exploit = env.draw(exploit_len)
            m_vals = mhat(x_exploit)
            p_exploit = np.empty(exploit_len, dtype=float)
            for i in range(exploit_len):
                root = ker.root(target=-float(m_vals[i]), stepsize=cfg.baseline_stepsize)
                p_exploit[i] = float(np.clip(root + m_vals[i], lo_exploit[i], hi_exploit[i]))
            start = total_steps + explore_len
            end = start + exploit_len
            x_hist[start:end] = x_exploit
            p_hist[start:end] = p_exploit
            lo_hist[start:end] = lo_exploit
            hi_hist[start:end] = hi_exploit

        total_steps += block_len
        episode += 1

    return _prefix_stats(cfg.sku, x_hist, p_hist, lo_hist, hi_hist)


def _dip_episode_exponents(T: int, init_exponent: int) -> list[int]:
    exponents: list[int] = []
    total = 0
    j = init_exponent
    while total < T:
        exponents.append(j)
        total += 2**j
        j += 1
    return exponents


def simulate_dip_path(cfg: SimConfigReal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    env = SemiSyntheticEnv(cfg.sku, rng)

    d = cfg.sku.contexts.shape[1]
    x_hist = np.zeros((cfg.T, d), dtype=float)
    p_hist = np.zeros(cfg.T, dtype=float)
    d_hist = np.zeros(cfg.T, dtype=float)
    lo_hist = np.zeros(cfg.T, dtype=float)
    hi_hist = np.zeros(cfg.T, dtype=float)
    t_ptr = 0

    init_len = min(2**cfg.dip_init_exponent, cfg.T)
    if init_len <= 0:
        zeros = np.zeros(cfg.T, dtype=float)
        return zeros, zeros, zeros

    x0, lo0, hi0 = env.draw(init_len)
    p0 = rng.uniform(lo0, hi0)
    d0 = env.sample_demand(x0, p0)
    x_hist[:init_len] = x0
    p_hist[:init_len] = p0
    d_hist[:init_len] = d0
    lo_hist[:init_len] = lo0
    hi_hist[:init_len] = hi0
    t_ptr = init_len

    intercept_hat, coef_hat = _fit_dip_linear_utility(x_hist[:t_ptr], p_hist[:t_ptr], d_hist[:t_ptr])

    for exponent in _dip_episode_exponents(cfg.T, cfg.dip_init_exponent):
        block_len = min(2**exponent, cfg.T - t_ptr)
        if block_len <= 0:
            break

        intv = max(2, int(cfg.dip_discretization_c * math.ceil(block_len ** (1.0 / 6.0))))
        me0 = np.zeros(intv, dtype=float)
        ti0 = np.zeros(intv, dtype=float)
        block_start = t_ptr

        x_blk, lo_blk, hi_blk = env.draw(block_len)
        for i in range(block_len):
            x_row = x_blk[i]
            lo = float(lo_blk[i])
            hi = float(hi_blk[i])
            cx = float(intercept_hat + x_row @ coef_hat)
            u1 = hi
            u2 = max(abs(cx - lo), abs(cx), abs(cx - hi))
            u = max(u1 + 2.0 * u2, 1e-6)
            ku = u / intv
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
                                * math.log((cfg.dip_lambda * intv + i * (u1**2)) / (cfg.dip_lambda * intv))
                            )
                        )
                        ** 2,
                    )
                    inde = np.empty(num, dtype=float)
                    for i1 in range(num):
                        inde[i1] = ((i1 * ku) + rma) * (me[i1] + math.sqrt(beta_t / (cfg.dip_lambda + ti[i1])))
                    best = np.flatnonzero(inde >= np.max(inde) - 1e-12)
                    bc = int(best[rng.integers(0, len(best))]) + 1

            p = float(np.clip((bc - 1) * ku + rma, lo, hi))
            demand = float(env.sample_demand(x_row.reshape(1, -1), np.array([p]))[0])

            idx = dex1 - 1 + bc - 1
            me0[idx] = (me0[idx] * (cfg.dip_lambda + ti0[idx]) + p * demand) / (
                cfg.dip_lambda + ti0[idx] + p**2
            )
            ti0[idx] += p**2

            x_hist[t_ptr] = x_row
            p_hist[t_ptr] = p
            d_hist[t_ptr] = demand
            lo_hist[t_ptr] = lo
            hi_hist[t_ptr] = hi
            t_ptr += 1
            if t_ptr >= cfg.T:
                break

        x_block = x_hist[block_start:t_ptr]
        p_block = p_hist[block_start:t_ptr]
        d_block = d_hist[block_start:t_ptr]
        if len(x_block) >= 8:
            intercept_hat, coef_hat = _fit_dip_linear_utility(x_block, p_block, d_block)

        if t_ptr >= cfg.T:
            break

    return _prefix_stats(cfg.sku, x_hist[:t_ptr], p_hist[:t_ptr], lo_hist[:t_ptr], hi_hist[:t_ptr])


def run_single_trial(
    task: tuple[tuple[int, ...], int, CalibratedSKU]
) -> tuple[str, list[tuple[int, float, float, float, float, float, float, float, float, float]]]:
    horizons, seed, sku = task
    max_T = max(horizons)
    cfg = SimConfigReal(T=max_T, seed=seed, sku=sku)
    my_regret_path, my_oracle_path, my_realized_path = simulate_mymethod_path(cfg)
    base_regret_path, base_oracle_path, base_realized_path = simulate_kernel_baseline_path(cfg)
    dip_regret_path, dip_oracle_path, dip_realized_path = simulate_dip_path(cfg)
    print(
        f"[trial] sku={sku.sku} Tmax={max_T} seed={seed} "
        f"mymethod={my_regret_path[max_T - 1]:.6f} baseline={base_regret_path[max_T - 1]:.6f} dip={dip_regret_path[max_T - 1]:.6f}",
        flush=True,
    )
    out: list[tuple[int, float, float, float, float, float, float, float, float, float]] = []
    for T in horizons:
        idx = T - 1
        out.append(
            (
                T,
                float(my_regret_path[idx]),
                float(base_regret_path[idx]),
                float(dip_regret_path[idx]),
                float(my_oracle_path[idx]),
                float(my_realized_path[idx]),
                float(base_oracle_path[idx]),
                float(base_realized_path[idx]),
                float(dip_oracle_path[idx]),
                float(dip_realized_path[idx]),
            )
        )
    return sku.sku, out


def run_trials(
    horizons: list[int],
    trials: int,
    dataset_path: str | Path = DEFAULT_DATASET,
    max_skus: int | None = None,
    workers: int | None = None,
    max_purchase_rate: float = 0.95,
    max_upper_bound_oracle_share: float = 1.0,
) -> list[dict[str, float | int | str]]:
    horizons = sorted(set(int(T) for T in horizons))
    skus = load_calibrated_skus(
        dataset_path=dataset_path,
        max_skus=max_skus,
        max_purchase_rate=max_purchase_rate,
        max_upper_bound_oracle_share=max_upper_bound_oracle_share,
    )
    tasks: list[tuple[tuple[int, ...], int, CalibratedSKU]] = []
    for sku in skus:
        for seed in range(trials):
            tasks.append((tuple(horizons), seed, sku))

    rows: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    if workers is None:
        workers = min(max(os.cpu_count() or 1, 1), max(len(tasks), 1), 6)

    used_parallel = False
    if workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for sku_name, trial_rows in ex.map(run_single_trial, tasks):
                    for (
                        T,
                        mymethod_regret,
                        baseline_regret,
                        dip_regret,
                        my_oracle,
                        my_realized,
                        base_oracle,
                        base_realized,
                        dip_oracle,
                        dip_realized,
                    ) in trial_rows:
                        rows.setdefault((sku_name, T, "mymethod"), []).append((mymethod_regret, my_oracle, my_realized))
                        rows.setdefault((sku_name, T, "kernel_baseline"), []).append((baseline_regret, base_oracle, base_realized))
                        rows.setdefault((sku_name, T, "dip"), []).append((dip_regret, dip_oracle, dip_realized))
            used_parallel = True
        except PermissionError:
            print("[warn] multiprocessing unavailable; falling back to sequential execution", flush=True)

    if not used_parallel:
        for task in tasks:
            sku_name, trial_rows = run_single_trial(task)
            for (
                T,
                mymethod_regret,
                baseline_regret,
                dip_regret,
                my_oracle,
                my_realized,
                base_oracle,
                base_realized,
                dip_oracle,
                dip_realized,
            ) in trial_rows:
                rows.setdefault((sku_name, T, "mymethod"), []).append((mymethod_regret, my_oracle, my_realized))
                rows.setdefault((sku_name, T, "kernel_baseline"), []).append((baseline_regret, base_oracle, base_realized))
                rows.setdefault((sku_name, T, "dip"), []).append((dip_regret, dip_oracle, dip_realized))

    out: list[dict[str, float | int | str]] = []
    grouped_for_plot: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    for (sku_name, T, method), stats in sorted(rows.items()):
        regrets = np.array([triple[0] for triple in stats], dtype=float)
        oracles = np.array([triple[1] for triple in stats], dtype=float)
        realizeds = np.array([triple[2] for triple in stats], dtype=float)
        avg = float(np.mean(regrets))
        std = float(np.std(regrets))
        avg_oracle = float(np.mean(oracles))
        avg_realized = float(np.mean(realizeds))
        out.append(
            {
                "scenario": "real_semisynth",
                "sku": sku_name,
                "method": method,
                "T": T,
                "avg_regret": avg,
                "std_regret": std,
                "avg_regret_per_period": avg / max(T, 1),
                "avg_oracle_revenue": avg_oracle,
                "avg_realized_revenue": avg_realized,
                "relative_regret": avg / max(avg_oracle, 1e-9),
                "n_trials": len(regrets),
            }
        )
        grouped_for_plot.setdefault((T, method), []).append((avg, avg_oracle, avg_realized))

    for (T, method), values in sorted(grouped_for_plot.items()):
        regrets = np.array([triple[0] for triple in values], dtype=float)
        oracles = np.array([triple[1] for triple in values], dtype=float)
        realizeds = np.array([triple[2] for triple in values], dtype=float)
        avg = float(np.mean(regrets))
        avg_oracle = float(np.mean(oracles))
        avg_realized = float(np.mean(realizeds))
        out.append(
            {
                "scenario": "real_semisynth",
                "sku": "__aggregate__",
                "method": method,
                "T": T,
                "avg_regret": avg,
                "std_regret": float(np.std(regrets)),
                "avg_regret_per_period": avg / max(T, 1),
                "avg_oracle_revenue": avg_oracle,
                "avg_realized_revenue": avg_realized,
                "relative_regret": avg / max(avg_oracle, 1e-9),
                "n_trials": len(regrets),
            }
        )
    return out


def _group_plot_rows(rows: list[dict[str, float | int | str]], method: str) -> list[dict[str, float]]:
    out = []
    for row in rows:
        if row["sku"] != "__aggregate__" or row["method"] != method:
            continue
        n = int(row["n_trials"])
        std = float(row["std_regret"])
        out.append(
            {
                "T": float(row["T"]),
                "avg_regret": float(row["avg_regret"]),
                "stderr": std / math.sqrt(max(n, 1)),
            }
        )
    return sorted(out, key=lambda item: item["T"])


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3.2" points="{pts}" />'


def _svg_polygon(points: list[tuple[float, float]], color: str) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon fill="{color}" fill-opacity="0.15" stroke="none" points="{pts}" />'


def plot_comparison(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    width, height = 980, 620
    left, right, top, bottom = 90, 40, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_pts = {m: _group_plot_rows(rows, m) for m in ["mymethod", "kernel_baseline", "dip"]}
    all_pts = [pt for pts in x_pts.values() for pt in pts]
    if not all_pts:
        raise ValueError("No aggregate rows found for plotting.")
    xmin = min(pt["T"] for pt in all_pts)
    xmax = max(pt["T"] for pt in all_pts)
    ymin = min(0.0, min(max(pt["avg_regret"] - pt["stderr"], 0.0) for pt in all_pts))
    ymax = max(pt["avg_regret"] + pt["stderr"] for pt in all_pts)

    def xmap(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin + 1e-12) * plot_w

    def ymap(y: float) -> float:
        return top + (1.0 - (y - ymin) / (ymax - ymin + 1e-12)) * plot_h

    colors = {"mymethod": "#1f77b4", "kernel_baseline": "#d62728", "dip": "#2a9d8f"}
    labels = {"mymethod": "ILPR", "kernel_baseline": "Kernel-based policy", "dip": "DIP"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="28" font-family="Helvetica">Real-data regret comparison</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5" />',
        f'<text x="{left + plot_w/2:.1f}" y="{height - 18}" text-anchor="middle" font-size="20" font-family="Helvetica">Horizon T</text>',
        f'<text x="24" y="{top + plot_h/2:.1f}" text-anchor="middle" font-size="20" font-family="Helvetica" transform="rotate(-90, 24, {top + plot_h/2:.1f})">Average cumulative regret across products</text>',
    ]

    svg.append(f'<rect x="{width - 455}" y="{top - 8}" width="405" height="92" fill="white" stroke="#444" />')
    for idx, method in enumerate(["mymethod", "kernel_baseline", "dip"]):
        y = top + 28 * idx
        x = width - 430
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x + 48}" y2="{y}" stroke="{colors[method]}" stroke-width="5" />')
        svg.append(f'<circle cx="{x + 24}" cy="{y}" r="5.5" fill="{colors[method]}" />')
        svg.append(f'<text x="{x + 62}" y="{y + 6}" font-size="21" font-family="Helvetica">{labels[method]}</text>')

    xticks = sorted({pt["T"] for pt in all_pts})
    for xt in xticks:
        x = xmap(xt)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#ddd" stroke-width="1" />')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="14" font-family="Helvetica">{int(xt)}</text>')

    ytick_vals = np.linspace(ymin, ymax, num=6)
    for yt in ytick_vals:
        y = ymap(float(yt))
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eee" stroke-width="1" />')
        svg.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="15" font-family="Helvetica">{yt:.0f}</text>')

    for method in ["mymethod", "kernel_baseline", "dip"]:
        pts = x_pts[method]
        if not pts:
            continue
        line_pts = [(xmap(pt["T"]), ymap(pt["avg_regret"])) for pt in pts]
        upper = [(xmap(pt["T"]), ymap(pt["avg_regret"] + pt["stderr"])) for pt in pts]
        lower = [(xmap(pt["T"]), ymap(max(pt["avg_regret"] - pt["stderr"], 1e-6))) for pt in reversed(pts)]
        svg.append(_svg_polygon(upper + lower, colors[method]))
        svg.append(_svg_polyline(line_pts, colors[method]))
        for x, y in line_pts:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{colors[method]}" />')

    svg.append("</svg>")
    (output_dir / "real_method_comparison.svg").write_text("\n".join(svg), encoding="utf-8")
