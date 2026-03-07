"""
cov_estimator.py
----------------
Estimate daily factor covariance and idiosyncratic variance for the
multi-factor risk model.

Method
------
Given the factor exposure matrix X_t (output of RiskFactorEngine) and daily
stock returns R_t, this module performs:

Step 1 — Cross-sectional WLS regression (per trading day):
    R_t = X_t f_t + ε_t,  W_t = diag(sqrt(total_mv))
    → daily factor returns f_t ∈ R^K
    → daily idiosyncratic residuals ε_t ∈ R^n_t

Step 2 — Rolling 60-day factor covariance:
    F_t = sample_cov(f_{t-59:t})    (K × K)
    Regularised: F_t += ridge * I to ensure positive definiteness.
    Cholesky: F_t = L_t L_t^T  → save L_t^T (upper triangular, K × K).

Step 3 — Rolling 60-day idiosyncratic variance (per stock):
    Δ_{ii,t} = var(ε_{i,t-59:t})   (rolling 60 days)
    Floor at delta_floor.  Stocks with < min_periods observations use the
    cross-sectional median as a fallback.

Outputs
-------
data/risk_cov_F.parquet
    Long format: [trade_date, f_i, f_j, value]
    Each row is one entry of L_t^T (the upper-triangular Cholesky factor).
    Reconstruct: pivot on (f_i, f_j), values → K×K numpy array.

data/risk_delta.parquet
    Long format: [trade_date, ts_code, delta_std]
    delta_std = sqrt(Δ_{ii}) — the per-stock idiosyncratic standard deviation.

Usage
-----
    Called from risk_model_main.py.  Not intended for standalone execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class CovarianceEstimator:
    """Estimate factor covariance and idiosyncratic variance from returns.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Long format [trade_date, ts_code, <factor_cols>].
        Output of RiskFactorEngine.compute().
    prices_df : pd.DataFrame
        Flat [trade_date, ts_code, close, tradable, ...].
    meta_df : pd.DataFrame
        Flat [trade_date, ts_code, total_mv, ...].
    cov_window : int
        Rolling lookback for factor cov and idiosyncratic var (default 60).
    min_periods : int
        Minimum valid observations in the rolling window before outputting
        delta_std; otherwise the cross-sectional median is used (default 30).
    delta_floor : float
        Minimum allowed value for Δ_{ii} (variance, not std).
        Default (0.005)^2 = 2.5e-5 (0.5% daily return floor).
    ridge : float
        Ridge regularisation added to F before Cholesky (default 1e-6).
    """

    def __init__(
        self,
        exposure_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        cov_window: int = 60,
        min_periods: int = 30,
        delta_floor: float = 2.5e-5,
        ridge: float = 1e-6,
    ) -> None:
        self.cov_window  = cov_window
        self.min_periods = min_periods
        self.delta_floor = delta_floor
        self.ridge       = ridge

        # Factor columns (everything except the key columns)
        key_cols = {"trade_date", "ts_code"}
        self.factor_cols: list[str] = [
            c for c in exposure_df.columns if c not in key_cols
        ]
        self.K = len(self.factor_cols)

        # Pre-index exposure by date for O(1) daily lookup
        exposure_df = exposure_df.copy()
        exposure_df["trade_date"] = pd.to_datetime(exposure_df["trade_date"])
        self._exposure_by_date: dict[pd.Timestamp, pd.DataFrame] = {
            date: grp.set_index("ts_code")[self.factor_cols]
            for date, grp in exposure_df.groupby("trade_date")
        }

        # Pivot prices / meta to wide format for fast daily slicing
        prices_df = prices_df.copy()
        prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"])
        meta_df = meta_df.copy()
        meta_df["trade_date"] = pd.to_datetime(meta_df["trade_date"])

        close_wide = (
            prices_df.pivot(index="trade_date", columns="ts_code", values="close")
            .sort_index()
        )
        tradable_wide = (
            prices_df.pivot(index="trade_date", columns="ts_code", values="tradable")
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
        mv_wide = (
            meta_df.pivot(index="trade_date", columns="ts_code", values="total_mv")
            .sort_index()
            .reindex(close_wide.index)
        )

        self._returns_wide  = close_wide.pct_change()
        self._tradable_wide = tradable_wide.reindex(self._returns_wide.index)
        self._mv_wide       = mv_wide.reindex(self._returns_wide.index)
        self._all_dates     = self._returns_wide.index
        self._all_stocks    = self._returns_wide.columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full covariance estimation pipeline.

        Returns
        -------
        f_half_df : pd.DataFrame
            [trade_date, f_i, f_j, value] — entries of L_t^T (Cholesky factor).
        delta_df : pd.DataFrame
            [trade_date, ts_code, delta_std] — per-stock idiosyncratic std.
        """
        # Step 1: WLS regression per day
        print("  Step 1: Running cross-sectional WLS regressions ...")
        f_returns, eps_wide = self._run_all_regressions()

        # Step 2: Rolling factor covariance → Cholesky
        print("  Step 2: Computing rolling factor covariance (Cholesky) ...")
        f_half_df = self._compute_rolling_F_half(f_returns)

        # Step 3: Rolling idiosyncratic variance → delta_std
        print("  Step 3: Computing rolling idiosyncratic variance ...")
        delta_df = self._compute_rolling_delta(eps_wide)

        return f_half_df, delta_df

    # ------------------------------------------------------------------
    # Step 1: WLS regression
    # ------------------------------------------------------------------

    def _run_all_regressions(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run WLS regression for every trading day.

        Returns
        -------
        f_returns : pd.DataFrame  (index=trade_date, columns=factor_cols)
            Factor return series; NaN row on days when regression fails.
        eps_wide : pd.DataFrame  (index=trade_date, columns=ts_code)
            Per-stock residuals; NaN when stock not in that day's regression.
        """
        eps_wide = pd.DataFrame(
            np.nan,
            index=self._all_dates,
            columns=self._all_stocks,
            dtype=float,
        )
        f_records = []

        n_total = len(self._all_dates)
        for i, date in enumerate(self._all_dates):
            if (i + 1) % 200 == 0:
                print(f"    ... processed {i + 1}/{n_total} days")

            result = self._wls_one_day(date)
            if result is None:
                continue

            f_t, eps_t, stocks_t = result
            f_records.append({"trade_date": date, **dict(zip(self.factor_cols, f_t))})
            eps_wide.loc[date, stocks_t] = eps_t

        if f_records:
            f_returns = (
                pd.DataFrame(f_records)
                .set_index("trade_date")
                .reindex(self._all_dates)
            )
        else:
            f_returns = pd.DataFrame(
                np.nan, index=self._all_dates, columns=self.factor_cols
            )

        return f_returns, eps_wide

    def _wls_one_day(
        self, date: pd.Timestamp
    ) -> Optional[tuple[np.ndarray, np.ndarray, list]]:
        """Run WLS regression R_t = X_t f_t + ε_t for a single day.

        Returns (f_t, eps_t, stock_codes) or None if regression cannot run.
        """
        # Tradability and return validity mask
        tradable = self._tradable_wide.loc[date]
        ret_row  = self._returns_wide.loc[date]
        mv_row   = self._mv_wide.loc[date]

        mask = tradable & ret_row.notna() & mv_row.notna() & (mv_row > 0)
        stocks = list(self._all_stocks[mask])

        if len(stocks) < self.K + 1:
            return None

        # Factor exposures for this day's tradable universe
        exp_day = self._exposure_by_date.get(date)
        if exp_day is None:
            return None

        exp_sub = exp_day.reindex(stocks)[self.factor_cols]
        valid   = ~exp_sub.isna().any(axis=1)  # stocks with all-valid exposures
        stocks  = list(exp_sub.index[valid])

        if len(stocks) < self.K + 1:
            return None

        X = exp_sub.loc[stocks].values.astype(float)   # (n, K)
        R = ret_row[stocks].values.astype(float)        # (n,)
        W = np.sqrt(mv_row[stocks].values.astype(float))  # (n,) WLS weights

        # Weighted system: multiply rows by W
        Xw = X * W[:, np.newaxis]   # (n, K)
        Rw = R * W                   # (n,)

        try:
            f_t, _, _, _ = np.linalg.lstsq(Xw, Rw, rcond=None)  # (K,)
        except Exception:
            return None

        eps_t = R - X @ f_t  # (n,) residuals

        return f_t, eps_t, stocks

    # ------------------------------------------------------------------
    # Step 2: Rolling factor covariance → Cholesky
    # ------------------------------------------------------------------

    def _compute_rolling_F_half(self, f_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling cov of factor returns and return Cholesky L^T.

        For date index i (within f_returns), uses f_returns[i-window+1 : i+1].
        Requires at least K+1 non-NaN rows; otherwise the date is skipped.

        Returns
        -------
        pd.DataFrame
            Long format [trade_date, f_i, f_j, value].
        """
        K = self.K
        fi_names = [self.factor_cols[i] for i in range(K) for _ in range(K)]
        fj_names = [self.factor_cols[j] for _ in range(K) for j in range(K)]
        # Pre-flatten into a template to avoid repeated list comprehensions
        fi_flat  = [self.factor_cols[r] for r in range(K) for c in range(K)]
        fj_flat  = [self.factor_cols[c] for r in range(K) for c in range(K)]

        records = []
        dates   = f_returns.index
        vals    = f_returns.values  # (T, K) — may contain NaN

        for i in range(len(dates)):
            date  = dates[i]
            start = max(0, i - self.cov_window + 1)
            chunk = vals[start : i + 1]              # (≤window, K)

            # Drop rows with any NaN
            chunk_clean = chunk[~np.any(np.isnan(chunk), axis=1)]
            if len(chunk_clean) < max(self.min_periods, K + 1):
                continue

            F = np.cov(chunk_clean, rowvar=False)    # (K, K)
            F = (F + F.T) / 2                        # symmetrise numerical noise

            # Positive-definite regularisation
            F += self.ridge * np.eye(K)

            try:
                L = np.linalg.cholesky(F)            # F = L L^T (lower triangular)
            except np.linalg.LinAlgError:
                # Extra regularisation to ensure PD
                eig_min = np.linalg.eigvalsh(F).min()
                F      += (-eig_min + 1e-8) * np.eye(K)
                try:
                    L = np.linalg.cholesky(F)
                except np.linalg.LinAlgError:
                    continue

            L_T = L.T  # upper triangular, K × K; satisfies F = L @ L.T

            batch = pd.DataFrame({
                "trade_date": date,
                "f_i":        fi_flat,
                "f_j":        fj_flat,
                "value":      L_T.ravel(),           # row-major flatten
            })
            records.append(batch)

        if records:
            return pd.concat(records, ignore_index=True)
        else:
            return pd.DataFrame(columns=["trade_date", "f_i", "f_j", "value"])

    # ------------------------------------------------------------------
    # Step 3: Rolling idiosyncratic variance → delta_std
    # ------------------------------------------------------------------

    def _compute_rolling_delta(self, eps_wide: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling per-stock residual std (sqrt Δ_{ii}).

        Parameters
        ----------
        eps_wide : pd.DataFrame  (index=trade_date, columns=ts_code)

        Returns
        -------
        pd.DataFrame
            Long format [trade_date, ts_code, delta_std].
        """
        # Rolling variance, then apply floor, then sqrt
        delta_var = eps_wide.rolling(
            self.cov_window, min_periods=self.min_periods
        ).var()
        delta_var = delta_var.clip(lower=self.delta_floor)

        # Fallback: for stocks with insufficient history, use cross-sectional median
        cross_median = delta_var.median(axis=1)           # (T,)
        for col in delta_var.columns:
            missing = delta_var[col].isna()
            delta_var.loc[missing, col] = cross_median[missing]

        delta_std = np.sqrt(delta_var)

        # Melt to long format
        delta_long = (
            delta_std.stack(future_stack=True)
            .reset_index()
        )
        delta_long.columns = ["trade_date", "ts_code", "delta_std"]
        delta_long = delta_long.dropna(subset=["delta_std"]).reset_index(drop=True)

        return delta_long
