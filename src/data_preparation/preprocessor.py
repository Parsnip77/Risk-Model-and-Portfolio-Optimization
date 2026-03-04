"""
FactorCleaner: convert raw factor values into model-ready cross-sectional features.

Pipeline (applied per factor column, per trading date):
    Step 1 — Sanity check    : replace ±inf with NaN
    Step 2 — NaN fill        : replace NaN with SW L1 industry cross-section median;
                               any remaining NaN falls back to global (all-stock) median
    Step 3 — Percentile rank : convert each cross-section to [0, 1] percentile ranks

Design rationale:
    - Filling before ranking ensures every tradable stock has a valid rank,
      avoiding information loss from pure listwise deletion.
    - Industry median (not mean) is used because factor distributions are often
      skewed; the median is less affected by extreme values within an industry.
    - Percentile rank is distribution-free and automatically handles outliers;
      an extreme value simply receives rank ≈ 1.0 or ≈ 0.0 without distorting
      the rest of the cross-section.

Non-tradable cells are NOT modified here.  The calling script is responsible
for applying the tradable mask after process_all() returns.

Output values are in (0, 1]; NaN is preserved for cells that could not be
filled (e.g. entire industry is NaN on that date AND global median is also NaN).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class FactorCleaner:
    """
    Clean raw factor values into model-ready cross-sectional percentile ranks.

    Parameters
    ----------
    data_dict : dict
        Dictionary from DataEngine.load_data().  Must contain 'df_industry'
        (indexed by code, column 'industry' with SW L1 industry names).
    """

    def __init__(self, data_dict: dict):
        df_industry = data_dict["df_industry"]
        # Static SW L1 industry assignment per stock (code → industry string)
        self._industry: pd.Series = df_industry["industry"]

    # ------------------------------------------------------------------
    # Individual steps (public — can be called standalone)
    # ------------------------------------------------------------------

    def fill_industry_median(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values with the SW L1 industry cross-section median.

        For each trading date (row):
            1. For each NaN cell, compute the median of all non-NaN stocks
               in the same SW L1 industry group on that date.
            2. If the entire industry group is NaN (or the stock's industry
               is unknown), fall back to the global median of all non-NaN
               stocks on that date.
            3. If the global median is also NaN (all stocks are NaN on that
               date), the cell remains NaN.

        Parameters
        ----------
        factor_df : DataFrame
            Wide form: index = trade date, columns = stock code.

        Returns
        -------
        DataFrame of the same shape; NaN reduced as described above.
        """
        industry_map = self._industry.reindex(factor_df.columns)

        def fill_row(row: pd.Series) -> pd.Series:
            if not row.isna().any():
                return row
            row = row.copy()

            # Vectorized industry median via groupby.transform
            # groupby maps each code → its industry; transform returns a Series
            # aligned to row's index with the group's median for each code.
            ind_med = row.groupby(industry_map, sort=False).transform("median")
            row = row.fillna(ind_med)

            # Fallback: global median for any still-NaN cells
            if row.isna().any():
                global_med = row.dropna().median()
                if pd.notna(global_med):
                    row = row.fillna(global_med)

            return row

        return factor_df.apply(fill_row, axis=1)

    def percentile_rank(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert each cross-section to [0, 1] percentile ranks.

        rank(pct=True, axis=1) assigns:
            - rank 1/N to the smallest value
            - rank 1.0 to the largest value
        NaN values are excluded from ranking and remain NaN in the output.

        Parameters
        ----------
        factor_df : DataFrame
            Wide form: index = trade date, columns = stock code.
        """
        return factor_df.rank(axis=1, pct=True, na_option="keep")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_all(self, raw_factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete cleaning pipeline on a raw factor DataFrame.

        Steps
        -----
        1. Sanity check  : ±inf → NaN
        2. NaN fill      : industry median → global median fallback
        3. Percentile rank : cross-section rank in (0, 1]

        Parameters
        ----------
        raw_factors_df : pd.DataFrame
            MultiIndex (date, code) × factor columns, as returned by
            FactorEngine.get_all_factors().  NaN and inf are expected.

        Returns
        -------
        pd.DataFrame
            Same MultiIndex (date, code) and column structure.
            Values in (0, 1]; NaN only where fill was impossible.
            Non-tradable cells are NOT overwritten here.
        """
        factor_names = raw_factors_df.columns.tolist()

        wide: dict = {
            col: raw_factors_df[col].unstack(level="code")
            for col in factor_names
        }

        cleaned: dict = {}
        for name, df in wide.items():
            # Step 1: replace ±inf
            df = df.replace([np.inf, -np.inf], np.nan)
            # Step 2: fill NaN (industry median → global median)
            df = self.fill_industry_median(df)
            # Step 3: cross-sectional percentile rank
            df = self.percentile_rank(df)
            cleaned[name] = df

        frames = []
        for name, df in cleaned.items():
            s = df.stack(dropna=False)
            s.name = name
            frames.append(s)

        out = pd.concat(frames, axis=1)
        out.index.names = ["date", "code"]
        return out
