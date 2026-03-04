"""
data_preparation_main.py — Phase 1 end-to-end pipeline.

Execution order:
    1. Verify that the local SQLite database exists.
    2. Load raw data via DataEngine.
    3. Compute tradable mask (suspended / delisted / limit-hit / ST / new-listing).
    4. Compute raw factors via FactorEngine (13 factors across groups A, B, C).
    5. Clean factors via FactorCleaner (industry-median fill + percentile rank).
       Override non-tradable cells to NaN.
    6. Export four Parquet files to ./data/:
         prices.parquet        — raw OHLCV + adj_factor + tradable flag
         meta.parquet          — 15 daily fundamentals + static industry
         factors_raw.parquet   — raw factor values (NaN preserved)
         factors_clean.parquet — cleaned percentile ranks; non-tradable cells = NaN

All four tables share the logical primary key (trade_date, ts_code).

Usage:
    python src/data_preparation/data_preparation_main.py
"""

import sys
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent.parent          # project root
sys.path.insert(0, str(ROOT / "src"))               # for config
sys.path.insert(0, str(ROOT / "src" / "data_preparation"))

import config  # noqa: E402
from data_loader  import DataEngine
from factors      import FactorEngine
from preprocessor import FactorCleaner


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _step(msg: str) -> None:
    print(f"\n{'=' * 62}\n  {msg}\n{'=' * 62}")

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _info(msg: str) -> None:
    print(f"  [--]  {msg}")

def _err(msg: str) -> None:
    print(f"  [ERR] {msg}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Reset MultiIndex (date, code) → flat columns (trade_date, ts_code)."""
    return (
        df.reset_index()
        .rename(columns={"date": "trade_date", "code": "ts_code"})
    )


def _compute_tradable(
    df_price: pd.DataFrame,
    df_st: pd.DataFrame,
    list_date_series: pd.Series,
) -> pd.Series:
    """
    Return a boolean Series (MultiIndex date×code) marking tradable stock-days.

    A stock-day is NOT tradable if ANY of:
        1. Suspended   — vol == 0
        2. Delisted    — close is NaN or 0
        3. Limit hit   — |pct_chg| > 9.5% AND close is flush against high or low
        4. ST / *ST    — date falls within any ST interval in stock_st
        5. New listing — listed for fewer than 180 calendar days
    """
    close = df_price["close"]
    high  = df_price["high"]
    low   = df_price["low"]
    vol   = df_price["vol"]

    suspended = vol == 0
    delisted  = close.isna() | (close == 0)

    pct_chg = (
        close.unstack("code")
             .pct_change()
             .stack(future_stack=True)
             .reindex(close.index)
    )
    large_move = pct_chg.abs() > 0.095
    at_high    = (high - close).abs() <= high.abs() * 1e-4
    at_low     = (close - low).abs() <= low.abs() * 1e-4
    limit_hit  = large_move & (at_high | at_low)

    # ST expansion
    idx_flat = df_price.index.to_frame(index=False)
    df_st_clean = df_st.copy()
    df_st_clean["end_date"] = df_st_clean["end_date"].fillna(config.END_DATE)
    merged = idx_flat.merge(df_st_clean, on="code", how="left")
    is_st_flat = (
        (merged["date"] >= merged["start_date"])
        & (merged["date"] <= merged["end_date"])
    ).fillna(False)
    is_st = is_st_flat.groupby([merged["date"], merged["code"]]).any()
    is_st_series = pd.Series(
        is_st.reindex(pd.MultiIndex.from_frame(idx_flat), fill_value=False).values,
        index=df_price.index,
    )

    # IPO filter: < 180 calendar days since listing
    price_dates = pd.to_datetime(df_price.index.get_level_values("date"))
    list_dates  = pd.to_datetime(
        df_price.index.get_level_values("code").map(list_date_series),
        errors="coerce",
    )
    ipo_flag = pd.Series(
        ((price_dates - list_dates).days < 180),
        index=df_price.index,
    ).fillna(False)

    return ~(suspended | delisted | limit_hit | is_st_series | ipo_flag)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = (ROOT / "src" / config.DB_PATH).resolve()

    # ----------------------------------------------------------------
    # Step 1: Database check
    # ----------------------------------------------------------------
    _step("Step 1 / 6  —  Database check")
    if not db_path.exists():
        _err(f"Database not found: {db_path}")
        _err("Run DataEngine().init_db() then engine.download_data() first.")
        sys.exit(1)
    _ok(f"Database: {db_path}")

    # ----------------------------------------------------------------
    # Step 2: Load data
    # ----------------------------------------------------------------
    _step("Step 2 / 6  —  Loading data from SQLite")
    engine = DataEngine()
    data   = engine.load_data()

    df_price      = data["df_price"]
    df_mv         = data["df_mv"]
    df_basic      = data["df_basic"]
    df_industry   = data["df_industry"]
    df_adj        = data["df_adj"]
    df_st         = data["df_st"]
    df_index      = data["df_index"]
    df_financials = data["df_financials"]

    _ok(f"daily_price  : {df_price.shape[0]:>7,} rows")
    _ok(f"daily_basic  : {df_basic.shape[0]:>7,} rows")
    _ok(f"stock_info   : {len(df_industry):>7,} stocks | "
        f"{df_industry['industry'].nunique()} industries")
    _ok(f"adj_factor   : {df_adj.shape[0]:>7,} rows")
    _ok(f"index_daily  : {len(df_index):>7,} rows (CSI 300)")
    _ok(f"quarterly fin: {len(df_financials):>7,} rows")

    # ----------------------------------------------------------------
    # Step 3: Tradable mask
    # ----------------------------------------------------------------
    _step("Step 3 / 6  —  Computing tradable mask")
    list_date_series = (
        df_industry["list_date"] if "list_date" in df_industry.columns
        else pd.Series(dtype=str)
    )
    tradable_mask = _compute_tradable(df_price, df_st, list_date_series)
    n_total    = len(tradable_mask)
    n_tradable = tradable_mask.sum()
    _ok(f"Total stock-days : {n_total:>10,}")
    _ok(f"Tradable         : {n_tradable:>10,}  ({n_tradable / n_total * 100:.2f}%)")
    _info(f"Non-tradable     : {n_total - n_tradable:>10,}")

    # ----------------------------------------------------------------
    # Step 4: Raw factors
    # ----------------------------------------------------------------
    _step("Step 4 / 6  —  Computing raw factors (FactorEngine, forward-adj)")
    factor_engine = FactorEngine(data, adj_type="forward")
    df_factors_raw = factor_engine.get_all_factors()

    factor_cols = df_factors_raw.columns.tolist()
    nan_pct = df_factors_raw.isna().mean().mean() * 100
    _ok(f"Factors  : {len(factor_cols)}  [{', '.join(factor_cols)}]")
    _ok(f"Shape    : {df_factors_raw.shape[0]:,} rows × {df_factors_raw.shape[1]} factors")
    _info(f"Avg NaN  : {nan_pct:.2f}%  (expected; warmup period + quarterly data gaps)")

    # ----------------------------------------------------------------
    # Step 5: Clean factors
    # ----------------------------------------------------------------
    _step("Step 5 / 6  —  Cleaning factors (FactorCleaner: fill + percentile rank)")
    cleaner          = FactorCleaner(data)
    df_factors_clean = cleaner.process_all(df_factors_raw)

    # Override non-tradable cells with NaN so downstream models skip them
    tradable_aligned = tradable_mask.reindex(df_factors_clean.index, fill_value=False)
    df_factors_clean.loc[~tradable_aligned] = np.nan

    nan_pct_clean = df_factors_clean.isna().mean().mean() * 100
    _ok(f"Shape    : {df_factors_clean.shape[0]:,} rows × {df_factors_clean.shape[1]} factors")
    _info(f"NaN %    : {nan_pct_clean:.2f}%  (non-tradable + unfillable cells)")

    # ----------------------------------------------------------------
    # Step 6: Export Parquet files
    # ----------------------------------------------------------------
    _step("Step 6 / 6  —  Exporting Parquet files  →  ./data/")

    # -- prices.parquet -----------------------------------------------
    df_prices = (
        df_price[["open", "high", "low", "close", "vol"]]
        .rename(columns={"vol": "volume"})
        .join(df_adj["adj_factor"], how="left")
    )
    df_prices["tradable"] = tradable_mask.reindex(df_prices.index, fill_value=False)
    prices_out = _flatten(df_prices)
    prices_out.to_parquet(out_dir / "prices.parquet", index=False)
    _ok(f"prices.parquet        : {prices_out.shape[0]:>7,} rows | {prices_out.columns.tolist()}")

    # -- meta.parquet -------------------------------------------------
    _all_basic_cols = [
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm",
        "total_share", "float_share", "free_share", "total_mv", "circ_mv",
    ]
    available = [c for c in _all_basic_cols if c in df_basic.columns]
    df_meta = df_basic[available].copy()
    df_meta["industry"] = df_meta.index.get_level_values("code").map(
        df_industry["industry"]
    )
    df_meta = df_meta[["industry"] + available]
    meta_out = _flatten(df_meta)
    meta_out.to_parquet(out_dir / "meta.parquet", index=False)
    _ok(f"meta.parquet          : {meta_out.shape[0]:>7,} rows | {len(meta_out.columns)} cols")

    # -- factors_raw.parquet ------------------------------------------
    raw_out = _flatten(df_factors_raw)
    raw_out.to_parquet(out_dir / "factors_raw.parquet", index=False)
    _ok(f"factors_raw.parquet   : {raw_out.shape[0]:>7,} rows | {len(factor_cols)} factor cols")

    # -- factors_clean.parquet ----------------------------------------
    clean_out = _flatten(df_factors_clean)
    clean_out.to_parquet(out_dir / "factors_clean.parquet", index=False)
    _ok(f"factors_clean.parquet : {clean_out.shape[0]:>7,} rows | {len(factor_cols)} factor cols"
        f" | non-tradable = NaN")

    _step("Complete")
    _ok(f"All files saved to: {out_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
