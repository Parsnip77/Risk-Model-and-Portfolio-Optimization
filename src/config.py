"""
Global configuration for the multi-factor stock selection project.

Centralizes all parameters to eliminate hardcoding in business logic.

IMPORTANT: This file contains your Tushare token. It is listed in .gitignore
           and must never be committed to version control.
"""

# ------------------------------------------------------------
# Tushare Pro API
# Register and obtain a token at: https://tushare.pro
# ------------------------------------------------------------
TUSHARE_TOKEN = ""

# ------------------------------------------------------------
# Data download date range (format: YYYYMMDD)
# ------------------------------------------------------------
START_DATE = "20190101"
END_DATE = "20231231"

# ------------------------------------------------------------
# Universe: CSI 300 index
# Tushare accepts '000300.SH' (SSE) or '399300.SZ' (SZSE) for the same index.
# If one fails with index_weight, try the other.
# ------------------------------------------------------------
UNIVERSE_INDEX = "000300.SH"

# ------------------------------------------------------------
# Paths (relative to src/, resolved at runtime in data_loader.py)
# ------------------------------------------------------------
DB_PATH = "../data/stock_data.db"

# ------------------------------------------------------------
# API throttle: seconds to sleep between consecutive Tushare calls
# Basic plan allows ~500 calls/min; 0.2 s is a safe conservative value.
# ------------------------------------------------------------
SLEEP_PER_CALL = 0.35
