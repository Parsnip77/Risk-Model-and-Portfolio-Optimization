"""
risk_model_main.py
------------------
Stage 3 entry script: multi-factor risk model construction.

Workflow
--------
1.  Load prices.parquet, meta.parquet, and (optionally) index.parquet from data/.
2.  Instantiate RiskFactorEngine to compute daily factor exposures X_t:
      - Style factors: Size, Beta, Momentum, Volatility, Value (cross-section z-scored)
      - Industry dummies: SW Level-1 classification (binary 0/1)
    Save to data/risk_exposure.parquet.
3.  Instantiate CovarianceEstimator to:
      a. Run cross-sectional WLS regression per day: R_t = X_t f_t + ε_t
      b. Compute rolling 60-day factor covariance F_t with Cholesky decomposition
      c. Compute rolling 60-day per-stock idiosyncratic variance Δ_t
    Save Cholesky factors L_t^T to data/risk_cov_F.parquet.
    Save idiosyncratic stds sqrt(Δ_{ii}) to data/risk_delta.parquet.

Output files (in data/)
-----------------------
  risk_exposure.parquet  — [trade_date, ts_code, size, beta, momentum,
                             volatility, value, ind_<industry>, ...]
  risk_cov_F.parquet     — [trade_date, f_i, f_j, value]  (L_t^T entries)
  risk_delta.parquet     — [trade_date, ts_code, delta_std]

Prerequisites
-------------
Run src/data_preparation/data_preparation_main.py first to generate:
    data/prices.parquet, data/meta.parquet, data/index.parquet

Usage
-----
    python src/risk_model/risk_model_main.py

Estimated runtime
-----------------
    Beta regression      : ~2-5 min  (loop over T days, vectorised across N stocks)
    WLS regression loop  : ~3-8 min  (loop over T days)
    Covariance estimation: ~1 min    (numpy/pandas rolling ops)
    Total                : ~6-14 min depending on data size
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup: src/risk_model/ → project root is two levels up
# ---------------------------------------------------------------------------
_RISK_DIR = pathlib.Path(__file__).parent          # src/risk_model/
_ROOT     = _RISK_DIR.parent.parent                # project root
sys.path.insert(0, str(_RISK_DIR))

from risk_factor_engine import RiskFactorEngine    # noqa: E402
from cov_estimator      import CovarianceEstimator # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = _ROOT / "data"

# Rolling window for covariance and delta estimation
COV_WINDOW   = 60   # trading days
MIN_PERIODS  = 30   # minimum observations before outputting delta_std
DELTA_FLOOR  = 2.5e-5   # (0.5% / day)^2 — lower bound on Δ_{ii}
RIDGE        = 1e-6     # ridge regularisation added to F before Cholesky


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _step(msg: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {msg}")
    print(f"{'=' * 65}")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _err(msg: str) -> None:
    print(f"  [ERROR] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    _step("Stage 3 · Multi-Factor Risk Model Construction")

    # -----------------------------------------------------------------------
    # 1. Check required files
    # -----------------------------------------------------------------------
    required = {
        "prices": DATA_DIR / "prices.parquet",
        "meta":   DATA_DIR / "meta.parquet",
    }
    for name, path in required.items():
        if not path.exists():
            _err(f"{name}.parquet not found at {path}")
            _err("Run src/data_preparation/data_preparation_main.py first.")
            sys.exit(1)
        _ok(f"Found {path.name}")

    index_path = DATA_DIR / "index.parquet"
    if not index_path.exists():
        print("  [WARN] index.parquet not found — beta factor will be set to NaN.")

    # -----------------------------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------------------------
    _step("Step 1 / 3  —  Loading data")

    prices_df = pd.read_parquet(required["prices"])
    meta_df   = pd.read_parquet(required["meta"])

    # Flatten MultiIndex if present (data_preparation_main may produce MultiIndex)
    for df in (prices_df, meta_df):
        if isinstance(df.index, pd.MultiIndex):
            df.reset_index(inplace=True)

    index_df: pd.DataFrame | None = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)
        if isinstance(index_df.index, pd.MultiIndex):
            index_df = index_df.reset_index()
        _ok(f"Loaded index.parquet  ({len(index_df)} rows)")

    _ok(f"Loaded prices.parquet  ({len(prices_df)} rows)")
    _ok(f"Loaded meta.parquet    ({len(meta_df)} rows)")

    # -----------------------------------------------------------------------
    # 3. Compute risk factor exposures
    # -----------------------------------------------------------------------
    _step("Step 2 / 3  —  Computing risk factor exposures")

    rfe = RiskFactorEngine(
        prices_df=prices_df,
        meta_df=meta_df,
        index_df=index_df,
    )
    exposure_df = rfe.compute()

    # Save
    out_exposure = DATA_DIR / "risk_exposure.parquet"
    exposure_df.to_parquet(out_exposure, index=False)
    _ok(f"Saved risk_exposure.parquet  ({len(exposure_df)} rows × {len(exposure_df.columns)} cols)")

    style_cols = ["size", "beta", "momentum", "volatility", "value"]
    ind_cols   = [c for c in exposure_df.columns if c.startswith("ind_")]
    print(f"  Factor columns: {len(style_cols)} style + {len(ind_cols)} industry = K={len(style_cols)+len(ind_cols)}")
    if ind_cols:
        print(f"  Industries: {', '.join(ind_cols[:5])}{'...' if len(ind_cols) > 5 else ''}")

    # -----------------------------------------------------------------------
    # 4. Estimate factor covariance and idiosyncratic variance
    # -----------------------------------------------------------------------
    _step("Step 3 / 3  —  Estimating covariance components")

    cov_est = CovarianceEstimator(
        exposure_df=exposure_df,
        prices_df=prices_df,
        meta_df=meta_df,
        cov_window=COV_WINDOW,
        min_periods=MIN_PERIODS,
        delta_floor=DELTA_FLOOR,
        ridge=RIDGE,
    )
    f_half_df, delta_df = cov_est.compute()

    # Save F_half (Cholesky L^T)
    out_f = DATA_DIR / "risk_cov_F.parquet"
    f_half_df.to_parquet(out_f, index=False)
    _ok(f"Saved risk_cov_F.parquet   ({len(f_half_df)} rows)")

    # Save delta_std
    out_delta = DATA_DIR / "risk_delta.parquet"
    delta_df.to_parquet(out_delta, index=False)
    _ok(f"Saved risk_delta.parquet   ({len(delta_df)} rows)")

    # Quick sanity check
    n_dates_F = f_half_df["trade_date"].nunique() if not f_half_df.empty else 0
    n_dates_d = delta_df["trade_date"].nunique()  if not delta_df.empty  else 0
    print(f"\n  Dates with F_half  : {n_dates_F}")
    print(f"  Dates with delta   : {n_dates_d}")

    elapsed = time.time() - t0
    _step(f"Done  —  total time {elapsed / 60:.1f} min")

    print(
        "\n  Output files:\n"
        f"    {out_exposure}\n"
        f"    {out_f}\n"
        f"    {out_delta}"
    )
    print(
        "\n  Next step:\n"
        "    Set USE_RISK_MODEL = True in src/portfolio/optimization_main.py\n"
        "    and re-run to incorporate risk model in the portfolio optimiser."
    )


if __name__ == "__main__":
    main()
