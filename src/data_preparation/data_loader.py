"""
DataEngine: data acquisition and persistence layer.

Extends the original data_loader with two additional tables:
    index_daily          : CSI 300 index daily close prices
    quarterly_financials : per-stock quarterly ROE, OCF/Revenue, single-quarter net income

Data flow:
    Tushare Pro API  -->  DataEngine.download_data()  -->  SQLite (stock_data.db)
    SQLite           -->  DataEngine.load_data()       -->  pandas DataFrames

All existing tables and interfaces are preserved unchanged.
"""

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tushare as ts

# config.py lives in src/; this file is in src/data_preparation/
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Module-level helper for single-quarter net income conversion
# ---------------------------------------------------------------------------

def _to_single_quarter(df: pd.DataFrame, ni_col: str = "n_income_cumulative") -> pd.Series:
    """
    Convert cumulative quarterly net income to single-quarter values.

    Tushare income API returns cumulative values within each fiscal year:
        Q1  = Q1
        H1  = Q1 + Q2   (reported at Q2)
        9M  = Q1+Q2+Q3  (reported at Q3)
        FY  = Q1+Q2+Q3+Q4 (reported at Q4 / annual)

    Strategy:
        - Sort by (code, end_date), derive fiscal year and quarter from end_date.
        - Within each (code, fiscal_year) group, compute shifted cumulative and subtract.
        - Q1 is always taken as-is (no prior quarter within the year).
        - If the prior quarter is missing, the result is NaN (safer than using 0).

    Parameters
    ----------
    df : DataFrame
        Must contain columns: [end_date, ni_col].
        end_date format: YYYYMMDD string.
    ni_col : str
        Column name for cumulative net income.

    Returns
    -------
    pd.Series
        Single-quarter net income aligned to df's index.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    tmp = df[[ni_col, "end_date"]].copy()
    tmp["end_dt"] = pd.to_datetime(tmp["end_date"], format="%Y%m%d", errors="coerce")
    # Use float (not Int64) so that NaT-derived NaN stays as numpy NaN.
    # Int64 would produce pd.NA, which raises "boolean value of NA is ambiguous"
    # inside np.where comparisons below.
    tmp["year"]    = tmp["end_dt"].dt.year.astype(float)
    tmp["quarter"] = ((tmp["end_dt"].dt.month - 1) // 3 + 1).astype(float)

    tmp["prev_q"] = tmp.groupby(["year"])["quarter"].shift(1)
    tmp["prev_cum"] = tmp.groupby(["year"])[ni_col].shift(1)

    # Valid subtraction: previous quarter must be exactly one quarter before current
    valid = (tmp["quarter"] - tmp["prev_q"]) == 1
    single_q = np.where(
        tmp["quarter"] == 1,
        tmp[ni_col],
        np.where(valid, tmp[ni_col] - tmp["prev_cum"], np.nan),
    )
    return pd.Series(single_q, index=df.index, dtype=float)


class DataEngine:
    """
    Central data I/O class for the risk model project.

    New tables compared to original:
        index_daily          : CSI 300 index daily close
        quarterly_financials : (code, ann_date, end_date) × [roe, ocf_to_revenue, n_income_single_q]

    Point-in-time (PIT) principle for quarterly_financials:
        ann_date is the public announcement date.  The FactorEngine performs
        merge_asof(ann_date <= trade_date) to ensure no look-ahead bias.
        When fina_indicator and income have different ann_dates for the same
        end_date, the LATER of the two is stored (more conservative).
    """

    def __init__(self):
        if config.TUSHARE_TOKEN in ("", "your_tushare_token"):
            raise ValueError(
                "TUSHARE_TOKEN is not set. Edit src/config.py and replace "
                "'your_tushare_token' with your actual token."
            )
        self.pro = ts.pro_api(config.TUSHARE_TOKEN)
        # config.DB_PATH = '../data/stock_data.db' relative to src/
        # This file is in src/data_preparation/, so parent.parent = src/
        self.db_path: str = str(
            (Path(__file__).parent.parent / config.DB_PATH).resolve()
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables if they do not exist (idempotent)."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                code    TEXT  NOT NULL,
                date    TEXT  NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                vol     REAL,
                amount  REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS daily_basic (
                code      TEXT  NOT NULL,
                date      TEXT  NOT NULL,
                pe        REAL,
                pb        REAL,
                total_mv  REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS stock_info (
                code      TEXT  PRIMARY KEY,
                name      TEXT,
                industry  TEXT
            );

            CREATE TABLE IF NOT EXISTS adj_factor (
                code        TEXT  NOT NULL,
                date        TEXT  NOT NULL,
                adj_factor  REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS stock_st (
                code        TEXT  NOT NULL,
                start_date  TEXT  NOT NULL,
                end_date    TEXT
            );

            CREATE TABLE IF NOT EXISTS index_daily (
                date   TEXT PRIMARY KEY,
                close  REAL
            );

            CREATE TABLE IF NOT EXISTS quarterly_financials (
                code              TEXT  NOT NULL,
                ann_date          TEXT  NOT NULL,
                end_date          TEXT  NOT NULL,
                roe               REAL,
                ocf_to_revenue    REAL,
                n_income_single_q REAL,
                PRIMARY KEY (code, ann_date, end_date)
            );
            """
        )
        conn.commit()

        # Schema migrations for older databases
        migrations = [
            ("daily_price",  "ALTER TABLE daily_price ADD COLUMN amount REAL"),
            ("stock_info",   "ALTER TABLE stock_info ADD COLUMN list_date TEXT"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN turnover_rate REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN turnover_rate_f REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN volume_ratio REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN pe_ttm REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN ps REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN ps_ttm REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN dv_ratio REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN dv_ttm REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN total_share REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN float_share REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN free_share REAL"),
            ("daily_basic",  "ALTER TABLE daily_basic ADD COLUMN circ_mv REAL"),
        ]
        for _, sql in migrations:
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass

        conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_constituents(self) -> List[str]:
        """Return CSI 300 constituents as of config.START_DATE (avoids look-ahead bias)."""
        start_dt = datetime.strptime(config.START_DATE, "%Y%m%d")
        end_dt = start_dt + timedelta(days=31)
        df = self.pro.index_weight(
            index_code=config.UNIVERSE_INDEX,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No constituents returned for '{config.UNIVERSE_INDEX}' "
                f"around {config.START_DATE}."
            )
        return df["con_code"].drop_duplicates().tolist()

    @staticmethod
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        """Map Tushare column names to project conventions."""
        return df.rename(columns={"ts_code": "code", "trade_date": "date"})

    def _fetch_sw_l1_industry_map(self, codes: List[str]):
        """Build code → Shenwan L1 industry name via index_classify + index_member_all."""
        try:
            time.sleep(config.SLEEP_PER_CALL)
            cls_df = self.pro.index_classify(level="L1", src="SW2021")
            if cls_df is None or cls_df.empty:
                return None
            if "index_code" not in cls_df.columns or "industry_name" not in cls_df.columns:
                return None
            code_to_industry = {}
            for _, row in cls_df.iterrows():
                time.sleep(config.SLEEP_PER_CALL)
                members = self.pro.index_member_all(l1_code=row["index_code"])
                if members is not None and not members.empty and "ts_code" in members.columns:
                    for _, m in members.iterrows():
                        code_to_industry[m["ts_code"]] = row["industry_name"]
            return pd.Series(code_to_industry) if code_to_industry else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Download & persist
    # ------------------------------------------------------------------

    def download_data(self) -> None:
        """
        Download all data for the CSI 300 universe and persist to SQLite.

        Steps:
            1. Stock metadata (name, list_date, Shenwan L1 industry)
            2. Daily OHLCV + extended fundamentals per stock
            3. Adjustment factors per stock
            4. ST status history per stock
            5. CSI 300 index daily close  (NEW)
            6. Quarterly financials per stock: roe, ocf_to_revenue, NI  (NEW)
        """
        codes = self._get_constituents()
        print(f"Universe: {len(codes)} stocks  [{config.UNIVERSE_INDEX}]")
        print(f"Date range: {config.START_DATE} -> {config.END_DATE}")

        conn = sqlite3.connect(self.db_path)

        # ---- Step 1a: Stock metadata ----
        print("Fetching stock metadata...")
        info_df = self.pro.stock_basic(fields="ts_code,name,list_date", list_status="L")
        if info_df is None or info_df.empty:
            conn.close()
            return
        info_df = info_df[info_df["ts_code"].isin(codes)].copy()
        info_df = self._rename(info_df)

        # ---- Step 1b: Shenwan L1 industry ----
        industry_map = self._fetch_sw_l1_industry_map(codes)
        if industry_map is not None:
            info_df["industry"] = info_df["code"].map(industry_map).fillna("其他")
            print("  Industry: Shenwan L1 (SW2021)")
        else:
            print("  [WARN] SW L1 unavailable; using stock_basic industry.")
            full_df = self.pro.stock_basic(
                fields="ts_code,name,industry,list_date", list_status="L"
            )
            if full_df is not None and not full_df.empty:
                full_df = self._rename(full_df[full_df["ts_code"].isin(codes)])
                info_df["industry"] = info_df["code"].map(
                    full_df.set_index("code")["industry"]
                ).fillna("其他")
            else:
                info_df["industry"] = "其他"

        conn.execute("DELETE FROM stock_info")
        info_df.to_sql("stock_info", conn, if_exists="append", index=False)
        conn.commit()

        # ---- Step 2: Daily price + fundamentals ----
        _basic_fields = (
            "ts_code,trade_date,"
            "turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,"
            "dv_ratio,dv_ttm,"
            "total_share,float_share,free_share,total_mv,circ_mv"
        )
        price_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM daily_price", conn)["code"].tolist()
        )
        n = len(codes)
        skipped_price = 0

        print("Downloading daily price + fundamentals...")
        for i, ts_code in enumerate(codes):
            if ts_code in price_cached:
                skipped_price += 1
                continue
            try:
                time.sleep(config.SLEEP_PER_CALL)
                price_df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
                time.sleep(config.SLEEP_PER_CALL)
                basic_df = self.pro.daily_basic(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields=_basic_fields,
                )
                if price_df is not None and not price_df.empty:
                    self._rename(price_df).to_sql(
                        "daily_price", conn, if_exists="append", index=False
                    )
                if basic_df is not None and not basic_df.empty:
                    self._rename(basic_df).to_sql(
                        "daily_basic", conn, if_exists="append", index=False
                    )
            except Exception as exc:
                print(f"  [WARN] {ts_code}: {exc}")
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  {i + 1 - skipped_price} / {n}")
        conn.commit()
        print(f"Price done. Skipped: {skipped_price}.")

        # ---- Step 3: Adjustment factors ----
        adj_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM adj_factor", conn)["code"].tolist()
        )
        skipped_adj = 0
        print("Downloading adj_factor...")
        for i, ts_code in enumerate(codes):
            if ts_code in adj_cached:
                skipped_adj += 1
                continue
            try:
                time.sleep(config.SLEEP_PER_CALL)
                adj_df = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,adj_factor",
                )
                if adj_df is not None and not adj_df.empty:
                    self._rename(adj_df).to_sql(
                        "adj_factor", conn, if_exists="append", index=False
                    )
            except Exception as exc:
                print(f"  [WARN] adj {ts_code}: {exc}")
            if (i + 1) % 50 == 0:
                conn.commit()
        conn.commit()
        print(f"Adj done. Skipped: {skipped_adj}.")

        # ---- Step 4: ST status ----
        st_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM stock_st", conn)["code"].tolist()
        )
        skipped_st = 0
        print("Downloading ST status...")
        for i, ts_code in enumerate(codes):
            if ts_code in st_cached:
                skipped_st += 1
                continue
            try:
                time.sleep(config.SLEEP_PER_CALL)
                st_df = self.pro.stock_st(ts_code=ts_code)
                if st_df is not None and not st_df.empty:
                    st_df = st_df.rename(columns={"ts_code": "code"})
                    keep = [c for c in ["code", "start_date", "end_date"] if c in st_df.columns]
                    st_df = st_df.dropna(subset=["start_date"])
                    if not st_df.empty:
                        st_df[keep].to_sql("stock_st", conn, if_exists="append", index=False)
                    else:
                        pd.DataFrame([{"code": ts_code, "start_date": "99991231", "end_date": "99991231"}]).to_sql(
                            "stock_st", conn, if_exists="append", index=False
                        )
                else:
                    pd.DataFrame([{"code": ts_code, "start_date": "99991231", "end_date": "99991231"}]).to_sql(
                        "stock_st", conn, if_exists="append", index=False
                    )
            except Exception as exc:
                print(f"  [WARN] ST {ts_code}: {exc}")
            if (i + 1) % 50 == 0:
                conn.commit()
        conn.commit()
        print(f"ST done. Skipped: {skipped_st}.")

        # ---- Step 5: CSI 300 index daily close ----
        self._download_index_daily(conn)

        # ---- Step 6: Quarterly financials ----
        self._download_quarterly_financials(codes, conn)

        conn.close()
        print("download_data() complete.")

    def _download_index_daily(self, conn: sqlite3.Connection) -> None:
        """Download CSI 300 index daily close prices into index_daily table."""
        existing = pd.read_sql("SELECT COUNT(*) AS n FROM index_daily", conn)["n"].iloc[0]
        if existing > 0:
            print("index_daily already cached; skipping.")
            return

        print("Downloading index daily close (CSI 300)...")
        try:
            time.sleep(config.SLEEP_PER_CALL)
            df = self.pro.index_daily(
                ts_code=config.UNIVERSE_INDEX,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                fields="trade_date,close",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={"trade_date": "date"}).sort_values("date")
                df[["date", "close"]].to_sql(
                    "index_daily", conn, if_exists="append", index=False
                )
                conn.commit()
                print(f"  index_daily: {len(df)} rows saved.")
            else:
                print("  [WARN] No index data returned.")
        except Exception as exc:
            print(f"  [WARN] index_daily download failed: {exc}")

    def _download_quarterly_financials(
        self, codes: List[str], conn: sqlite3.Connection
    ) -> None:
        """
        Download per-stock quarterly financials and persist to quarterly_financials.

        For each stock, makes two API calls:
            1. fina_indicator  → roe_avg, ocf_to_revenue
            2. income          → n_income_attr_p (归属于母公司净利润, cumulative)

        Both are merged on end_date; the later ann_date of the two is used (conservative PIT).
        Cumulative n_income is converted to single-quarter values before storage.
        """
        fin_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM quarterly_financials", conn)[
                "code"
            ].tolist()
        )
        skipped = 0
        n = len(codes)
        print("Downloading quarterly financials (roe, ocf_to_revenue, net income)...")

        for i, ts_code in enumerate(codes):
            if ts_code in fin_cached:
                skipped += 1
                continue
            try:
                # --- fina_indicator ---
                time.sleep(config.SLEEP_PER_CALL)
                fi = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,ann_date,end_date,roe_avg,ocf_to_revenue",
                )

                # --- income statement ---
                time.sleep(config.SLEEP_PER_CALL)
                inc = self.pro.income(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,ann_date,end_date,n_income_attr_p",
                )

                # Sentinel row so stock is marked as cached even if both fail
                has_data = False

                if fi is not None and not fi.empty:
                    fi = fi.rename(columns={"ts_code": "code", "roe_avg": "roe"})
                    # Tushare may omit fields silently; add missing columns as NaN
                    for _col in ["roe", "ocf_to_revenue"]:
                        if _col not in fi.columns:
                            fi[_col] = np.nan
                    # Keep latest ann_date per end_date
                    fi = (
                        fi.sort_values("ann_date")
                        .groupby("end_date")
                        .last()
                        .reset_index()
                    )
                    has_data = True
                else:
                    fi = pd.DataFrame(columns=["code", "ann_date", "end_date", "roe", "ocf_to_revenue"])

                if inc is not None and not inc.empty:
                    inc = inc.rename(columns={"ts_code": "code"})
                    if "n_income_attr_p" not in inc.columns:
                        inc["n_income_attr_p"] = np.nan
                    inc = (
                        inc.sort_values("ann_date")
                        .groupby("end_date")
                        .last()
                        .reset_index()
                    )
                    has_data = True
                else:
                    inc = pd.DataFrame(columns=["code", "ann_date", "end_date", "n_income_attr_p"])

                if not has_data:
                    # No data at all — insert sentinel to mark as cached
                    pd.DataFrame([{
                        "code": ts_code, "ann_date": "99991231", "end_date": "99991231",
                        "roe": None, "ocf_to_revenue": None, "n_income_single_q": None,
                    }]).to_sql("quarterly_financials", conn, if_exists="append", index=False)
                    continue

                # Merge on end_date; outer join so we get rows from either source
                merged = fi.merge(
                    inc[["end_date", "ann_date", "n_income_attr_p"]],
                    on="end_date",
                    how="outer",
                    suffixes=("_fi", "_inc"),
                )
                # code might be NaN on rows that only came from inc
                if "code_fi" in merged.columns:
                    merged["code"] = merged["code_fi"].fillna(ts_code)
                    merged = merged.drop(columns=["code_fi"], errors="ignore")
                merged["code"] = merged.get("code", ts_code).fillna(ts_code)

                # Use later ann_date (more conservative PIT).
                # ann_date columns are YYYYMMDD strings; outer-join NaN values are float.
                # Convert both to datetime before taking max to avoid cross-type comparison.
                ann_fi  = pd.to_datetime(
                    merged.get("ann_date_fi",  pd.Series([pd.NaT] * len(merged))),
                    format="%Y%m%d", errors="coerce",
                )
                ann_inc = pd.to_datetime(
                    merged.get("ann_date_inc", pd.Series([pd.NaT] * len(merged))),
                    format="%Y%m%d", errors="coerce",
                )
                merged["ann_date"] = (
                    pd.DataFrame({"fi": ann_fi, "inc": ann_inc})
                    .max(axis=1)
                    .dt.strftime("%Y%m%d")
                )
                merged = merged.drop(
                    columns=["ann_date_fi", "ann_date_inc"], errors="ignore"
                )

                # Filter to quarters only (end_date month in {03,06,09,12})
                merged["_month"] = pd.to_datetime(
                    merged["end_date"], format="%Y%m%d", errors="coerce"
                ).dt.month
                merged = merged[merged["_month"].isin([3, 6, 9, 12])].drop(columns=["_month"])

                # Compute single-quarter net income
                merged = merged.sort_values("end_date").reset_index(drop=True)
                merged["n_income_single_q"] = _to_single_quarter(
                    merged, ni_col="n_income_attr_p"
                )

                out = merged[
                    ["code", "ann_date", "end_date", "roe", "ocf_to_revenue", "n_income_single_q"]
                ].dropna(subset=["ann_date", "end_date"])
                out["code"] = ts_code

                if not out.empty:
                    out.to_sql(
                        "quarterly_financials", conn, if_exists="append", index=False
                    )

            except Exception as exc:
                print(f"  [WARN] quarterly {ts_code}: {exc}")

            if (i + 1) % 30 == 0:
                conn.commit()
                print(f"  quarterly financials: {i + 1 - skipped} / {n}")

        conn.commit()
        print(f"Quarterly financials done. Skipped: {skipped}.")

    # ------------------------------------------------------------------
    # Public helper: fetch latest adj factor
    # ------------------------------------------------------------------

    def fetch_latest_adj_factor(self, codes: List[str]) -> pd.Series:
        """Fetch the most recent adj_factor for each stock from Tushare API."""
        for offset in range(10):
            query_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                time.sleep(config.SLEEP_PER_CALL)
                df = self.pro.adj_factor(
                    trade_date=query_date,
                    fields="ts_code,adj_factor",
                )
                if df is not None and not df.empty:
                    df = df[df["ts_code"].isin(codes)]
                    if not df.empty:
                        return df.set_index("ts_code")["adj_factor"]
            except Exception:
                continue
        raise RuntimeError("Could not fetch latest adj_factor within the last 10 days.")

    # ------------------------------------------------------------------
    # Load for analysis
    # ------------------------------------------------------------------

    def load_data(self) -> dict:
        """
        Read all data from SQLite into memory.

        Returns
        -------
        dict with keys:
            df_price       : MultiIndex (date, code), cols [open, high, low, close, vol, amount]
            df_mv          : MultiIndex (date, code), col [total_mv]  (backward compat)
            df_basic       : MultiIndex (date, code), 15 fundamental fields
            df_industry    : indexed by code, cols [name, industry, list_date]
            df_adj         : MultiIndex (date, code), col [adj_factor]
            df_st          : flat [code, start_date, end_date]
            df_index       : Series, index=date (str), values=close (CSI 300)
            df_financials  : flat [code, ann_date, end_date, roe, ocf_to_revenue,
                                   n_income_single_q]  (sorted by code, ann_date)
        """
        conn = sqlite3.connect(self.db_path)

        price_df = pd.read_sql("SELECT * FROM daily_price", conn)
        price_df = price_df.set_index(["date", "code"]).sort_index()

        basic_df = pd.read_sql("SELECT * FROM daily_basic", conn)
        basic_df = basic_df.set_index(["date", "code"]).sort_index()

        mv_df = basic_df[["total_mv"]].copy()

        info_df = pd.read_sql("SELECT * FROM stock_info", conn, index_col="code")

        adj_df = pd.read_sql("SELECT * FROM adj_factor", conn)
        adj_df = adj_df.set_index(["date", "code"]).sort_index()

        st_df = pd.read_sql("SELECT code, start_date, end_date FROM stock_st", conn)

        # CSI 300 index close prices
        idx_df = pd.read_sql("SELECT date, close FROM index_daily ORDER BY date", conn)
        index_series = idx_df.set_index("date")["close"]

        # Quarterly financials (exclude sentinel rows)
        fin_df = pd.read_sql("SELECT * FROM quarterly_financials", conn)
        fin_df = fin_df[fin_df["ann_date"] != "99991231"].sort_values(
            ["code", "ann_date"]
        ).reset_index(drop=True)

        conn.close()

        return {
            "df_price":      price_df,
            "df_mv":         mv_df,
            "df_basic":      basic_df,
            "df_industry":   info_df,
            "df_adj":        adj_df,
            "df_st":         st_df,
            "df_index":      index_series,
            "df_financials": fin_df,
        }
