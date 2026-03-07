"""
risk_factor_engine.py
---------------------
Compute daily risk factor exposure matrices for the multi-factor risk model.

Factor definitions (K = 5 style + N_ind industry dummies)
----------------------------------------------------------
Style factors (cross-sectionally winsorized ±3σ then z-scored per day):
    size       : log(total_mv)
    beta       : 60-day rolling CAPM beta vs CSI 300 index
    momentum   : cumulative return from t-252 to t-21 (skip reversal window)
    volatility : 20-day rolling std of daily close-to-close returns
    value      : EP = 1/PE  (NaN when PE <= 0)

Industry factors (binary 0/1, NOT z-scored):
    ind_<name> : one-hot encoding of SW Level-1 industry classification

Output
------
data/risk_exposure.parquet
    Long format: [trade_date, ts_code, size, beta, momentum, volatility, value,
                  ind_<industry_1>, ..., ind_<industry_N>]

Usage
-----
    Called from risk_model_main.py.  Not intended for standalone execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

STYLE_FACTORS  = ["size", "beta", "momentum", "volatility", "value"]
BETA_WINDOW    = 60     # trading days for rolling CAPM OLS
MOMENTUM_LONG  = 252    # long lookback for momentum (days)
MOMENTUM_SHORT = 21     # short lookback skip (days); avoids short-term reversal
VOLATILITY_WIN = 20     # lookback for realised volatility (days)
WINSORIZE_SIG  = 3.0    # winsorization clip level (in cross-sectional std units)


class RiskFactorEngine:
    """Compute daily risk factor exposures for all stocks.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Flat format [trade_date, ts_code, open, high, low, close, vol, tradable, ...].
        The 'close' column should be adjusted (forward-adjusted) prices.
    meta_df : pd.DataFrame
        Flat format [trade_date, ts_code, pe, total_mv, industry, ...].
        'total_mv' in any consistent unit (万元); 'industry' is SW L1 name (str).
    index_df : pd.DataFrame or None
        CSI 300 index prices, flat format [trade_date, close].
        If None, beta is set to NaN for all stocks.
    beta_window : int
        Rolling window for CAPM beta regression (default 60 trading days).
    momentum_long : int
        Long lookback for momentum signal (default 252 trading days ≈ 1 year).
    momentum_short : int
        Short lookback skip to avoid reversal (default 21 trading days ≈ 1 month).
    volatility_window : int
        Rolling window for realised volatility (default 20 trading days ≈ 1 month).
    winsorize_sigma : float
        Winsorization threshold in cross-sectional std units (default 3.0).
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        beta_window: int = BETA_WINDOW,
        momentum_long: int = MOMENTUM_LONG,
        momentum_short: int = MOMENTUM_SHORT,
        volatility_window: int = VOLATILITY_WIN,
        winsorize_sigma: float = WINSORIZE_SIG,
    ) -> None:
        self.beta_window      = beta_window
        self.momentum_long    = momentum_long
        self.momentum_short   = momentum_short
        self.volatility_window = volatility_window
        self.winsorize_sigma  = winsorize_sigma

        # Pivot prices to wide format (trade_date × ts_code)
        prices_df = prices_df.copy()
        prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"])
        meta_df = meta_df.copy()
        meta_df["trade_date"] = pd.to_datetime(meta_df["trade_date"])

        self._close = (
            prices_df.pivot(index="trade_date", columns="ts_code", values="close")
            .sort_index()
        )

        self._total_mv = (
            meta_df.pivot(index="trade_date", columns="ts_code", values="total_mv")
            .sort_index()
            .reindex(self._close.index)
        )

        self._pe = (
            meta_df.pivot(index="trade_date", columns="ts_code", values="pe")
            .sort_index()
            .reindex(self._close.index)
        )

        # Industry classification for each (trade_date, ts_code) pair
        self._industry_long = (
            meta_df[["trade_date", "ts_code", "industry"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Daily close-to-close returns (used for beta and volatility)
        self._returns = self._close.pct_change()

        # Market (CSI 300) daily returns for beta computation
        if index_df is not None:
            idx_close = (
                index_df.set_index("trade_date")["close"]
                .sort_index()
                .reindex(self._close.index)
                .ffill()
            )
            self._mkt_ret: Optional[pd.Series] = idx_close.pct_change()
        else:
            self._mkt_ret = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> pd.DataFrame:
        """Compute all risk factor exposures and return long-format DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: [trade_date, ts_code, size, beta, momentum, volatility,
                      value, ind_<industry_1>, ..., ind_<industry_N>]
            Style factor values are cross-sectionally winsorized and z-scored.
            Industry dummies are binary (0 or 1).
        """
        print("  Computing style factors ...")

        size_z       = self._cs_zscore_winsorize(self._compute_size(),       self.winsorize_sigma)
        beta_z       = self._cs_zscore_winsorize(self._compute_beta(),       self.winsorize_sigma)
        momentum_z   = self._cs_zscore_winsorize(self._compute_momentum(),   self.winsorize_sigma)
        volatility_z = self._cs_zscore_winsorize(self._compute_volatility(), self.winsorize_sigma)
        value_z      = self._cs_zscore_winsorize(self._compute_value(),      self.winsorize_sigma)

        style_panels = {
            "size":       size_z,
            "beta":       beta_z,
            "momentum":   momentum_z,
            "volatility": volatility_z,
            "value":      value_z,
        }

        # Stack each style factor to long format
        pieces = []
        for name, df in style_panels.items():
            s = (
                df.reindex(index=self._close.index, columns=self._close.columns)
                .stack(future_stack=True)
            )
            s.name = name
            pieces.append(s)

        style_long = pd.concat(pieces, axis=1)
        style_long.index.names = ["trade_date", "ts_code"]
        style_long = style_long.reset_index()

        # Industry dummies (long format)
        print("  Computing industry dummies ...")
        ind_dummies = self._compute_industry_dummies()

        # Merge style + industry on (trade_date, ts_code)
        result = style_long.merge(ind_dummies, on=["trade_date", "ts_code"], how="left")

        # Fill missing industry dummies with 0 (e.g. stocks missing from meta on some days)
        ind_cols = [c for c in result.columns if c.startswith("ind_")]
        result[ind_cols] = result[ind_cols].fillna(0.0)

        result["trade_date"] = pd.to_datetime(result["trade_date"])

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Factor computation (wide format: trade_date × ts_code)
    # ------------------------------------------------------------------

    def _compute_size(self) -> pd.DataFrame:
        """Size factor: log(total market capitalisation)."""
        return np.log(self._total_mv.clip(lower=1.0))

    def _compute_beta(self) -> pd.DataFrame:
        """Market beta: 60-day rolling OLS coefficient vs CSI 300 returns."""
        if self._mkt_ret is None:
            return pd.DataFrame(
                np.nan, index=self._close.index, columns=self._close.columns
            )

        mkt   = self._mkt_ret.values       # (T,)
        ret   = self._returns.values        # (T, N)
        T, N  = ret.shape
        beta  = np.full((T, N), np.nan)
        w     = self.beta_window

        print(f"    Computing rolling {w}-day CAPM beta ...")
        for t in range(w - 1, T):
            r_m = mkt[t - w + 1 : t + 1]   # (w,)
            if np.any(np.isnan(r_m)):
                continue

            r_s = ret[t - w + 1 : t + 1, :]  # (w, N)
            valid = ~np.any(np.isnan(r_s), axis=0)  # (N,) boolean mask
            if not valid.any():
                continue

            X = np.column_stack([np.ones(w), r_m])     # (w, 2)
            Y = r_s[:, valid]                            # (w, n_valid)

            try:
                # lstsq: rows of coef are [intercept, beta] for each stock
                coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (2, n_valid)
            except Exception:
                continue

            beta[t, valid] = coef[1, :]   # slope = market beta

        return pd.DataFrame(beta, index=self._close.index, columns=self._close.columns)

    def _compute_momentum(self) -> pd.DataFrame:
        """Momentum: cumulative return from t-252 to t-21 (skips 1 month)."""
        s = self.momentum_short
        l = self.momentum_long
        return self._close.shift(s) / self._close.shift(l) - 1.0

    def _compute_volatility(self) -> pd.DataFrame:
        """Realised volatility: rolling std of daily returns over past 20 days."""
        return self._returns.rolling(self.volatility_window).std()

    def _compute_value(self) -> pd.DataFrame:
        """Value factor: earnings yield EP = 1/PE.  NaN when PE <= 0 or missing."""
        ep = 1.0 / self._pe
        # Mask out non-positive PE (which makes EP economically meaningless)
        ep[self._pe <= 0] = np.nan
        return ep

    def _compute_industry_dummies(self) -> pd.DataFrame:
        """Create industry one-hot dummy variables.

        Returns
        -------
        pd.DataFrame
            Columns: [trade_date, ts_code, ind_<industry_1>, ..., ind_<industry_N>]
            Binary 0/1, one column per unique industry found in meta_df.
        """
        ind_df = self._industry_long.copy()
        ind_df["industry"] = ind_df["industry"].fillna("OTHER").astype(str)

        # Consistent column ordering: sort alphabetically
        dummies = pd.get_dummies(ind_df["industry"], prefix="ind", dtype=float)

        result = pd.concat(
            [ind_df[["trade_date", "ts_code"]].reset_index(drop=True),
             dummies.reset_index(drop=True)],
            axis=1,
        )
        result["trade_date"] = pd.to_datetime(result["trade_date"])

        return result

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def _cs_zscore_winsorize(
        df: pd.DataFrame, n_sigma: float = WINSORIZE_SIG
    ) -> pd.DataFrame:
        """Per-row (cross-sectional) winsorize then z-score.

        Steps:
        1. Winsorize: clip values outside mean ± n_sigma * std.
        2. Z-score: subtract (clipped) mean, divide by (clipped) std.
        3. Fill NaN rows (all NaN inputs, zero std) with 0.
        """
        mean = df.mean(axis=1)
        std  = df.std(axis=1)

        # Clip rows to [mean - k*std, mean + k*std]
        clipped = df.clip(
            lower=(mean - n_sigma * std),
            upper=(mean + n_sigma * std),
            axis=0,
        )

        c_mean = clipped.mean(axis=1)
        c_std  = clipped.std(axis=1).replace(0.0, np.nan)

        z = clipped.sub(c_mean, axis=0).div(c_std, axis=0)

        return z.fillna(0.0)
