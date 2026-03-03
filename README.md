本文档用简明语言说明本仓库中**所有文件与代码的用途、使用方式**，并展示项目架构。

## 一、项目概述

本项目是一个**多因子选股模型**的量化交易示例，使用 Python 实现从数据获取、因子构建到回测的完整流程。

**第一阶段：数据层（ETL）+ 因子计算 + 因子预处理**

第一阶段用 Tushare Pro 拉取沪深300成分股的日线行情与基本面指标，写入本地 SQLite 数据库；在此基础上复现《101 Formulaic Alphas》中的经典因子；再通过预处理模块对原始因子执行去极值、标准化、中性化，最终由 `data_preparation_main.py` 串联全流程，将四张核对齐的宽表导出为 Parquet 文件。

**第二阶段：因子评估层 + 分层回测 + 线性回归合成因子 + 扣费净收益回测**

第二阶段对每个因子进行 IC 评估，计算其信息系数均值和信息比率，绘制 IC 时间序列图和数值累计图，并据此进行有效因子筛选。针对每个单因子进行分层回测，绘制 G1 ～ G5 分组收益曲线及 G5-G1 多空收益曲线，给出累计收益率、年化收益率、年化波动率、夏普比率、最大回撤五个指标。最后，使用滚动窗口线性回归的方式尝试对多因子进行简单线性合成，并进行计算 0.2% 双边交易费用的实盘扣费回测，同时额外给出日均换手率和盈亏平衡换手率指标。总脚本由`analyze_main.py`执行，报告写入`result.txt`。

**第三阶段：ML 合成因子层 + 分层回测和净收益回测**  

第三阶段接入 LightGBM，以 15 个 alpha 为特征，24 个月训练 + 6 个月验证 + 6 个月测试进行滚动时间切分，训练合成一个 ML Alpha，并调用分层回测与净收益回测。最终给出回测指标数据报告、特征重要性参考和 SHAP 分析 beeswarm 图。总脚本由`ml_analyze_main.py`执行，报告写入`result_ml.txt`。

----

项目成果简要展示：

1. result.txt

```

==============================================================
  Step 1 / 6  —  Loading Parquet files
==============================================================
  [OK]  prices.parquet   : 361,333 rows | cols: ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'volume', 'adj_factor', 'tradable']
  [OK]  factors_clean    : 364,200 rows | factors: ['alpha001', 'alpha003', 'alpha006', 'alpha012', 'alpha038', 'alpha040', 'alpha041', 'alpha042', 'alpha054', 'alpha072', 'alpha088', 'alpha094', 'alpha098', 'alpha101', 'alpha_5_day_reversal']

==============================================================
  Step 2 / 6  —  Computing 5-day forward return
==============================================================
  [OK]  target shape     : 364,200 rows  (non-NaN: 359,446)

==============================================================
  Step 3 / 6  —  Single-factor IC analysis
==============================================================
--------------------------------------------------------------
  Factor: alpha001
--------------------------------------------------------------
  [  ]    IC Mean : -0.0112
  [  ]    IC Std  : 0.0954
  [  ]    ICIR    : -0.1177
  [  ]    IC chart saved.
  [  ]    Backtest metrics:
     Cum Return  Ann Return  Ann Vol  Sharpe  Max DD
G1       0.4258      0.0767   0.0901  0.5186 -0.2332
G2       0.6990      0.1168   0.0935  0.9287 -0.2004
G3       0.2937      0.0551   0.0895  0.2810 -0.2358
G4       0.2637      0.0500   0.0873  0.2290 -0.2445
G5       0.0799      0.0162   0.0867 -0.1598 -0.2743
L-S     -0.2471     -0.0574   0.0428 -2.0432 -0.2532
  [  ]    Backtest plot saved.
      not selected  (threshold: |IC mean| > 1.5%  &  |ICIR| > 0.2)
--------------------------------------------------------------
 ... 以下更多单因子测评信息省略，详见 result.txt
==============================================================
  Step 4 / 6  —  Results summary
==============================================================
  [OK]  Effective alphas (2 / 15):
      * alpha040
      * alpha054

==============================================================
  Step 5 / 6  —  Synthetic factor (rolling OLS + symmetric orthogonalization)
==============================================================
  [  ]  Effective alphas (IC/ICIR screen, for reference): ['alpha040', 'alpha054']
  [  ]  Using ALL 15 factors for OLS combination: ['alpha001', 'alpha003', 'alpha006', 'alpha012', 'alpha038', 'alpha040', 'alpha041', 'alpha042', 'alpha054', 'alpha072', 'alpha088', 'alpha094', 'alpha098', 'alpha101', 'alpha_5_day_reversal']
  [  ]  Rolling window = 3 trading days  |  orthogonalize = True
  [  ]    OLS target: cross-sectional pct-rank of forward_return (per trade_date)
  [OK]  Synthetic factor : 356,035 rows  (dates: 1206)
  [  ]    Backtest metrics (synthetic factor):
     Cum Return  Ann Return  Ann Vol   Sharpe  Max DD
G1      -0.5467     -0.1524   0.0883  -2.0652 -0.6428
G2      -0.1418     -0.0315   0.0869  -0.7075 -0.3581
G3       0.1572      0.0310   0.0901   0.0108 -0.2122
G4       0.8565      0.1380   0.0869   1.2426 -0.1905
G5       3.6823      0.3807   0.0857   4.0906 -0.1369
L-S      9.2715      0.6270   0.0346  17.2651 -0.0101
  [  ]    Synthetic backtest plot saved.

==============================================================
  Step 6 / 6  —  Net-return backtest (with transaction costs)
==============================================================
  [  ]  cost_rate = 0.20%  |  forward_days = 5  |  top 20%
  [  ]    Net-return performance summary:
Cum Return            1.976548
Ann Return            0.257178
Ann Vol               0.183588
Sharpe                1.237438
Max DD               -0.186031
Avg Daily Turnover    0.155391
Breakeven Turnover    0.653735
  [  ]    Net-return chart saved.

==============================================================
  Phase 2 factor analysis complete.
==============================================================

```

![synthetic_factor_backtest](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/synthetic_factor_backtest.png)

![synthetic_factor_net](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/synthetic_factor_net.png)

2. Result_ml.txt

```

======================================================================
  Stage 3 · LightGBM Alpha Synthesis
======================================================================
1. Loading factors_clean.parquet and prices.parquet ...
   factors shape : (364200, 17)
   prices  shape : (361333, 9)

2. Computing 1-day forward returns ...

3. Merging factors and target labels ...
   merged shape  : (357978, 19)
   feature cols  : ['alpha001', 'alpha003', 'alpha006', 'alpha012', 'alpha038', 'alpha040', 'alpha041', 'alpha042', 'alpha054', 'alpha072', 'alpha088', 'alpha094', 'alpha098', 'alpha101', 'alpha_5_day_reversal']
   training target : cs_rank_return  (cross-sectional pct-rank of forward_return, per trade_date)

======================================================================
  Walk-Forward Training
======================================================================
   Settings : train=24m  val=6m  test=6m  embargo=1d
   Estimated folds : 4

--- Fold 1 ---
  Train : 20190102 → 20210126  (n=148,726)
  Val   : 20210127 → 20210804  (n=37,069)
  Test  : 20210806 → 20220216  (n=37,143)
  Best iteration : 160

--- Fold 2 ---
  Train : 20190711 → 20210804  (n=148,587)
  Val   : 20210805 → 20220215  (n=37,141)
  Test  : 20220217 → 20220819  (n=37,176)
  Best iteration : 201

--- Fold 3 ---
  Train : 20200114 → 20220215  (n=148,263)
  Val   : 20220216 → 20220818  (n=37,180)
  Test  : 20220822 → 20230301  (n=37,325)
  Best iteration : 257

--- Fold 4 ---
  Train : 20200724 → 20220818  (n=148,506)
  Val   : 20220819 → 20230228  (n=37,323)
  Test  : 20230302 → 20230901  (n=37,206)
  Best iteration : 460

======================================================================
  Assembling Final ML Alpha
======================================================================
   final_alpha_df shape : (148252, 3)  (after 3-day smoothing)
   date range : 20210810 → 20230901

======================================================================
  IC Analysis — ML Synthetic Factor
======================================================================
   IC Mean : +0.0270
   IC Std  : 0.1015
   ICIR    : +0.2657

   IC chart saved: plots/ml_alpha_ic.png

======================================================================
  Layered Backtest (LayeredBacktester)
======================================================================

Layered Backtest Performance:

     Cum Return  Ann Return  Ann Vol  Sharpe  Max DD
G1      -0.2780     -0.1508   0.1639 -1.1030 -0.3640
G2      -0.1184     -0.0613   0.1690 -0.5404 -0.2636
G3      -0.1277     -0.0663   0.1720 -0.5598 -0.3201
G4       0.0401      0.0199   0.1779 -0.0566 -0.2342
G5       0.1402      0.0681   0.1834  0.2075 -0.2535
L-S      0.5745      0.2559   0.0965  2.3412 -0.0620

======================================================================
  Net Return Backtest (NetReturnBacktester)
======================================================================

Net Return Backtest Summary:

Cum Return           -0.238815
Ann Return           -0.128253
Ann Vol               0.182979
Sharpe               -0.864871
Max DD               -0.302129
Avg Daily Turnover    0.383103
Breakeven Turnover    0.054401

======================================================================
  Feature Importance (Average Across Folds)
======================================================================

Average Feature Importance (gain):

             feature   importance
            alpha054 11610.249995
            alpha038  4778.802732
            alpha042  4275.594320
            alpha041  3594.318232
            alpha001  3416.101811
alpha_5_day_reversal  3364.880188
            alpha094  2727.276456
            alpha012  2686.414797
            alpha088  2335.806660
            alpha101  2195.233947
            alpha006  1777.769686
            alpha040  1769.653773
            alpha003  1663.949998
            alpha072  1479.444898
            alpha098  1144.706116

   Saved: plots/feature_importance.png

======================================================================
  SHAP Analysis (Last Fold Test Sample)
======================================================================
   SHAP sample size : 300 rows
   Saved: plots/shap_beeswarm.png

======================================================================
  Report Written
======================================================================
```

![ml_alpha_ic](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/ml_alpha_ic.png)

![ml_alpha_backtest](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/ml_alpha_backtest.png)

![ml_alpha_net](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/ml_alpha_net.png)

![shap_beeswarm](/Users/infinity/Desktop/Career/Quant/Multi-factor Model for Stock Selection/Multi-factor-Model-for-Stock-Selection/plots/shap_beeswarm.png)

---

## 二、项目架构

```
项目根目录/
├── data/
│   ├── .gitkeep                # 保留空目录用于 git 追踪
│   ├── stock_data.db           # SQLite 数据库（运行 download_data() 后生成）
│   ├── prices.parquet          # 原始每日行情 + 复权因子
│   ├── meta.parquet            # 申万一级行业信息 + 15 字段基本面信息
│   ├── factors_raw.parquet     # 原始 Alpha 因子
│   └── factors_clean.parquet   # 清洗后 Alpha 因子
├── src/
│   ├── __init__.py             # 标识 src 为 Python 包
│   ├── config.py               # 全局配置（Token、日期范围、路径等）
│   ├── data_loader.py          # DataEngine 类：数据下载与读取
│   ├── alphas.py               # Alpha101 类：因子计算（101 Formulaic Alphas）
│   ├── preprocessor.py         # FactorCleaner 类：因子预处理（清洗）
│   ├── targets.py              # calc_forward_return：未来收益率标签生成
│   ├── ic_analyzer.py          # calc_ic / calc_ic_metrics / plot_ic：因子 IC 评估
│   ├── backtester.py           # LayeredBacktester：分层回测（分组、绩效指标、累计净值图）
│   ├── factor_combiner.py      # symmetric_orthogonalize + rolling_linear_combine：对称正交化 + 滚动 OLS 
│   ├── net_backtester.py       # NetReturnBacktester：考虑摩擦成本的纯多头重叠组合回测
│   ├── ml_data_prep.py         # WalkForwardSplitter：量化专用时序滚动切分器
│   └── lgbm_model.py           # AlphaLGBM：LightGBM 训练引擎 + 特征重要性 + SHAP
├── notebooks/
│   └── explore.ipynb           # Jupyter Notebook：数据探索与可视化
├── plots/                      # 图表输出目录（各阶段脚本自动创建）
├── data_preparation_main.py    # 第一阶段总脚本
├── analyze_main.py             # 第二阶段总脚本
├── ml_analyze_main.py          # 第三阶段总脚本
├── .gitignore                  # 版本控制忽略规则
├── requirements.txt            # Python 依赖列表
├── result.txt                  # analyze_main.py 自动生成的因子 summary
├── result_ml.txt               # ml_analyze_main.py 自动生成的 ML 合成因子报告
└── README.md              			# 本说明文档
```

| 文件/目录 | 用途 |
|-----------|------|
| `data/` | 存放 SQLite 数据库文件（不上传至 git） |
| `src/config.py` | 全局参数配置，包含 Tushare Token（不上传至 git） |
| `src/data_loader.py` | `DataEngine` 类：数据下载、缓存、读取 |
| `src/alphas.py` | `Alpha101` 类：复现《101 Formulaic Alphas》中的 15 个因子 |
| `src/preprocessor.py` | `FactorCleaner` 类：对原始因子执行去极值、标准化、中性化 |
| `src/targets.py` | `calc_forward_return(prices_df, d)`：计算 d 日未来收益率标签 |
| `src/ic_analyzer.py` | `calc_ic` / `calc_ic_metrics` / `plot_ic`：截面 Spearman IC 评估 |
| `src/backtester.py` | `LayeredBacktester`：分层回测，含绩效指标计算与累计净值绘图 |
| `src/factor_combiner.py` | `symmetric_orthogonalize`（Löwdin 正交化）+ `rolling_linear_combine`（含正交化开关的滚动 OLS 合成因子） |
| `src/net_backtester.py` | `NetReturnBacktester`：纯多头重叠组合回测，含摩擦成本、换手率、盈亏平衡换手率 |
| `src/ml_data_prep.py` | `WalkForwardSplitter`：量化专用时序滚动切分器，防止未来函数数据泄露 |
| `src/lgbm_model.py` | `AlphaLGBM`：LightGBM 训练引擎，集成特征重要性绘图与 SHAP 分析 |
| `data_preparation_main.py` | 第一阶段总脚本：串联 DataEngine → Alpha101 → FactorCleaner，导出四张 Parquet |
| `analyze_main.py` | 第二阶段总脚本：载入 Parquet，循环单因子 IC 检验，筛选有效 alpha，进行单因子分层回测，滚动线性回归合成因子，进行扣费净收益回测，给出回测报告 |
| `ml_analyze_main.py` | 第三阶段总脚本：LightGBM 合成因子 + 双回测（LayeredBacktester + NetReturnBacktester）+ 回测指标报告 + 特征重要性分析 / SHAP 分析 |
| `plots/` | 图表输出目录（各阶段脚本自动创建） |
| `data/*.parquet` | 导出的宽表数据，共享主键 (trade_date, ts_code)；`prices.parquet` 含 `tradable` 列，`factors_clean.parquet` 中不可交易格子为 NaN |
| `notebooks/explore.ipynb` | 数据探索 + Alpha 因子计算 + 因子清洗示例 |
| `.gitignore` | 忽略 Token、数据库、本地文档等敏感/冗余文件 |
| `requirements.txt` | `pip install -r requirements.txt` 所需依赖 |

---

## 三、各文件使用说明

### 1. requirements.txt

- **用途**：定义项目 Python 依赖。
- **使用**：
  
  ```bash
  pip install -r requirements.txt
  ```

---

### 2. src/config.py

- **用途**：集中管理所有全局参数。

- **主要配置项**：

  | 变量 | 说明 | 默认值 |
  |------|------|--------|
  | `TUSHARE_TOKEN` | Tushare Pro Token（必填） | `"your_tushare_token"` |
  | `START_DATE` | 数据开始日期（YYYYMMDD） | `"20190101"` |
  | `END_DATE` | 数据结束日期（YYYYMMDD） | `"20231231"` |
  | `UNIVERSE_INDEX` | 股票池指数代码 | `"000300.SH"` |
  | `DB_PATH` | 数据库路径（相对于 src/） | `"../data/stock_data.db"` |
  | `SLEEP_PER_CALL` | 每次 API 调用间隔（秒） | `0.35` |

---

### 3. src/data_loader.py

- **用途**：项目唯一的数据 I/O 层，封装为 `DataEngine` 类。

- **数据库表结构**：

  | 表名 | 字段 | 说明 |
  |------|------|------|
  | `daily_price` | code, date, open, high, low, close, vol, amount | 日线行情，主键 (code, date) |
  | `daily_basic` | code, date, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv | 每日扩展基本面（17 字段），主键 (code, date) |
  | `stock_info` | code, name, industry, list_date | 股票静态信息，主键 code；`industry` 为**申万一级**（来自 Tushare `index_classify(level='L1', src='SW2021')` + `index_member_all`，不可用时回退为 `stock_basic` 的 industry）；`list_date` 用于准新股过滤 |
  | `adj_factor` | code, date, adj_factor | 复权因子（前/后复权计算用），主键 (code, date) |
  | `stock_st` | code, start_date, end_date | ST/\*ST 状态历史区间；end_date 为 NULL 表示仍处于 ST 状态；无 ST 历史的股票写入哨兵行 (start_date='99991231') |

  > **模式迁移**：调用 `init_db()` 可对旧版数据库自动补全缺失列/表（幂等，使用 `try/except OperationalError` 方式）。新增的 `daily_basic` 扩展列、`stock_info.list_date`、`stock_st` 表均通过此机制自动添加。已缓存记录的新列默认为 NULL，需删除数据库后重新下载才能获得完整数据。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__()` | 初始化 Tushare Pro API，解析数据库路径 |
  | `init_db()` | 建表 + 模式迁移（幂等，可重复调用） |
  | `download_data()` | 下载 price / extended basic / adj_factor / ST 状态；先拉取 stock_basic（name, list_date），再拉取申万一级行业并合并进 stock_info；四套独立缓存，可单独补充缺失数据 |
  | `fetch_latest_adj_factor(codes)` | 即时调用 Tushare API 获取最新复权因子，供前复权使用 |
  | `load_data()` | 从 SQLite 读取，返回结构化字典（见下） |

- **`load_data()` 返回值**：
  ```python
  {
    "df_price"    : DataFrame  # MultiIndex (date, code)，列：open, high, low, close, vol, amount
    "df_mv"       : DataFrame  # MultiIndex (date, code)，列：total_mv
    "df_basic"    : DataFrame  # MultiIndex (date, code)，15 个扩展基本面字段
    "df_industry" : DataFrame  # index = code，列：name, industry, list_date
    "df_adj"      : DataFrame  # MultiIndex (date, code)，列：adj_factor
    "df_st"       : DataFrame  # 列：code, start_date, end_date（ST 状态区间）
  }
  ```

- **缓存机制**：`download_data()` 对 `daily_price` 和 `adj_factor` 维护**各自独立**的缓存集合，允许在 price 已缓存的情况下单独补充 adj_factor（反之亦然）。如需完全重新下载，删除 `data/stock_data.db` 后重新运行即可。

---

### 4. src/alphas.py

- **用途**：因子计算模块，复现《101 Formulaic Alphas》（Kakushadze, 2015）中的 Alpha 因子。纯计算逻辑，不涉及 I/O；输入为 `DataEngine.load_data()` 返回的数据字典，输出为原始因子值（保留 NaN/inf，清洗由后续模块处理）。

- **已实现因子**：

  | 因子 | 公式 |
  |------|------|
  | `alpha001` | `rank(ts_argmax(signedpower((ret<0)?stddev(ret,20):close, 2), 5)) - 0.5` |
  | `alpha003` | `-1 * correlation(rank(open), rank(volume), 10)` |
  | `alpha006` | `-1 * correlation(open, volume, 10)` |
  | `alpha012` | `sign(delta(volume, 1)) * (-1 * delta(close, 1))` |
  | `alpha038` | `(-1 * rank(ts_rank(close, 10))) * rank(close/open)` |
  | `alpha040` | `(-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)` |
  | `alpha041` | `sqrt(high * low) - vwap` |
  | `alpha042` | `rank(vwap - close) / rank(vwap + close)` |
  | `alpha054` | `(-1 * (low-close) * open^5) / ((low-high) * close^5)` |
  | `alpha072` | `rank(decay_linear(corr((H+L)/2, adv40, 8), 10)) / rank(decay_linear(corr(ts_rank(vwap,3), ts_rank(vol,18), 6), 2))` |
  | `alpha088` | `min(rank(decay_linear((rank(O)+rank(L))-(rank(H)+rank(C)),8)), ts_rank(decay_linear(corr(ts_rank(C,8),ts_rank(adv60,20),8),6),2))` |
  | `alpha094` | `signedpower(rank(vwap - ts_min(vwap,11)), ts_rank(corr(ts_rank(vwap,19), ts_rank(adv60,4), 18), 2)) * -1` |
  | `alpha098` | `rank(decay_linear(corr(vwap,sum(adv5,26),4),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(O),rank(adv15),20),8),6),8))` |
  | `alpha101` | `(close - open) / (high - low + 0.001)` |
  | `alpha_5_day_reversal` | `(close - delay(close, 5)) / delay(close, 5)` |

- **VWAP 计算**：`amount` 字段可用时使用精确公式 `amount × 10 / vol`；否则回退到典型价格 `(H+L+C)/3`。

- **复权模式**（`adj_type` 参数，默认 `"forward"`）：

  | 模式 | 公式 | 说明 |
  |------|------|------|
  | `"forward"` | `P × adj_factor / adj_factor_latest` | 前复权，最新价等于原始价 |
  | `"backward"` | `P × adj_factor` | 后复权，完整体现历史涨幅 |
  | `"raw"` | 不调整 | 直接使用数据库原始价格 |

  > `vol` 和 `amount` 在任何复权模式下均**不调整**。  
  > `adj_factor_latest` 默认取 `df_adj` 中的最后一行；也可通过 `latest_adj` 参数传入 `fetch_latest_adj_factor()` 的返回值以获得更精确的当日因子。

- **已实现辅助函数**：`_rank`、`_delay`、`_delta`、`_corr`、`_cov`、`_stddev`、`_sum`、`_product`、`_ts_min`、`_ts_max`、`_ts_argmax`、`_ts_argmin`、`_ts_rank`、`_scale`、`_decay_linear`、`_sign`、`_log`、`_abs`、`_signed_power`

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict, adj_type, latest_adj)` | 接收数据字典；自动计算精确 vwap 并应用复权 |
  | `alpha006()` … `alpha101()` | 各返回 wide-form DataFrame（行 = 日期，列 = 股票代码） |
  | `get_all_alphas()` | 计算所有已实现因子，返回 MultiIndex (date, code) × alpha 列的 DataFrame |

- **使用示例**：
  ```python
  from alphas import Alpha101
  data = DataEngine().load_data()
  
  alpha = Alpha101(data)                         # 前复权（默认）
  alpha = Alpha101(data, adj_type='backward')    # 后复权
  alpha = Alpha101(data, adj_type='raw')         # 不复权
  
  # 使用 API 获取最新复权因子（更精确的前复权）
  latest_adj = engine.fetch_latest_adj_factor(codes)
  alpha = Alpha101(data, adj_type='forward', latest_adj=latest_adj)
  
  df = alpha.get_all_alphas()   # MultiIndex (date, code) × ['alpha006', ...]
  ```

---

### 5. src/preprocessor.py

- **用途**：因子预处理模块，将 `Alpha101.get_all_alphas()` 的**原始因子**清洗为**可直接输入模型的因子**，封装为 `FactorCleaner` 类。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict)` | 从数据字典中提取对数市值（宽表）和行业 dummy 矩阵，作为中性化回归的基准 |
  | `winsorize(factor_df, method, limits)` | 截面去极值；`method='mad'`（默认）使用中位数绝对偏差，`method='sigma'` 使用均值±标准差 |
  | `standardize(factor_df)` | 截面 Z-score 标准化：\(Z = (X - \mu) / \sigma\) |
  | `neutralize(factor_df)` | 截面 OLS 回归：`Factor = β · [log_mv, industry_dummies] + ε`，返回残差 ε |
  | `process_all(raw_alphas_df)` | 完整五步流水线（见下），返回 `df_clean_factors` |

- **`process_all` 流水线**：

  | 步骤 | 操作 | 目的 |
  |------|------|------|
  | 1. 异常值初筛 | ±inf → NaN | 防止均值/方差计算出现数学错误 |
  | 2. 去极值 | MAD-based 截断（Median ± 3×1.4826×MAD） | 压缩尾部风险，保护后续 OLS 回归 |
  | 3. 初步标准化 | Z-score | 统一量纲，提升数值稳定性 |
  | 4. 中性化 | OLS 残差（剥离市值 + 行业偏好） | 保留纯 Alpha，消除系统性偏差 |
  | 5. 二次标准化 | Z-score on residuals | 使残差在不同日期间具有可比性 |
  | 6. 填补 NaN | 将缺失值替换为 0 | 统一维度，方便后续矩阵运算 |

- **输入**：
  ```python
  data_dict     # DataEngine.load_data() 的返回值（需含 df_mv 和 df_industry）
  raw_alphas_df # Alpha101.get_all_alphas() 的返回值
                # MultiIndex (date, code) × alpha 列，保留 NaN/inf
  ```

- **输出**：
  ```python
  df_clean_factors  # MultiIndex (date, code) × alpha 列
                    # 与输入相同结构，NaN 已填补为 0
  ```

- **使用示例**：
  ```python
  from preprocessor import FactorCleaner
  
  cleaner = FactorCleaner(data)            # data = engine.load_data()
  df_clean = cleaner.process_all(df_alphas)  # df_alphas = alpha.get_all_alphas()
  ```

---

### 6. notebooks/explore.ipynb

- **用途**：三部分演示：  
  - **Part 1**：数据库完整性校验（1.1 OHLCV+amount、1.2 adj_factor 历史浏览、1.3 市值、1.4 行业分布、1.5 缺失值检查）。  
  - **Part 2**：Alpha 因子计算（前复权价格 + 精确 vwap；单独宽表、合并长表、描述统计、NaN 覆盖率、最新截面）。  
  - **Part 3**：因子清洗（`FactorCleaner.process_all()` 输出的清洗因子；描述统计、零填充率、最新截面对比）。
- **使用**：
  ```bash
  jupyter notebook notebooks/explore.ipynb
  ```
  或在 VS Code / Cursor 中直接打开并逐单元格运行。
- **前提**：已运行 `download_data()`，`data/stock_data.db` 已生成。

---

### 7. data_preparation_main.py

- **用途**：第一阶段端到端总脚本，串联 `DataEngine → _compute_tradable → Alpha101 → FactorCleaner`，将全流程处理结果导出为四张 Parquet 文件。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 检查 `data/stock_data.db` 是否存在，缺失则报错退出 |
  | 2 | 调用 `DataEngine.load_data()` 载入全量数据 |
  | 2.5 | 调用 `_compute_tradable(df_price, df_st, list_date_series)` 计算可交易性掩码（停牌 / 退市 / 涨跌停 / ST / 准新股） |
  | 3 | 调用 `Alpha101.get_all_alphas()` 计算原始 Alpha 因子（前复权） |
  | 4 | 调用 `FactorCleaner.process_all()` 执行五步清洗流水线；之后对不可交易股票-日期覆写为 NaN |
  | 5 | 将结果导出为四张 Parquet 文件（见下） |
  | 6 | 打印汇总信息 |

- **`_compute_tradable(df_price, df_st, list_date_series)` 判断逻辑**（五个条件，任一成立即标记为不可交易）：

  | 条件 | 标准 | 说明 |
  |------|------|------|
  | 停牌 | `vol == 0` | 当日成交量为零，无法交易 |
  | 退市 | `close` 为 NaN 或 0 | 股票已不存在 |
  | 涨跌停 | `abs(pct_chg) > 9.5%` 且收盘价紧贴最高/最低价 | 实际流动性极低，无法以市价入场 |
  | ST/\*ST | 当日在 `stock_st` 表记录的任一 ST 区间内 | ST 股价格波动异常、涨跌停板缩窄至 ±5%，收益分布失真 |
  | 准新股 | 上市日期到该截面不足 180 个自然日 | IPO 初期价格发现机制不稳定，因子信号可靠性低 |

  返回 `pd.Series[bool]`（MultiIndex date×code），`True` = 可交易。

- **输出文件**（均存放于 `./data/`，主键逻辑对齐：`trade_date × ts_code`）：

  | 文件 | 列 | 说明 |
  |------|----|------|
  | `prices.parquet` | trade_date, ts_code, open, high, low, close, volume, adj_factor, **tradable** | 原始（未复权）行情 + 复权因子 + 可交易性标志（bool） |
  | `meta.parquet` | trade_date, ts_code, industry, turnover_rate, ... | 每日扩展基本面（15 字段）+ 静态**申万一级**行业 |
  | `factors_raw.parquet` | trade_date, ts_code, alpha006, … | 原始因子值（保留 NaN） |
  | `factors_clean.parquet` | trade_date, ts_code, alpha006, … | 清洗后因子值；**不可交易的股票-日期格子为 NaN**（不再填 0），下游回测 `dropna` 自动排除 |

- **使用**：
  ```bash
  python data_preparation_main.py
  ```
  脚本会逐步打印每个阶段的状态和关键统计信息。

---

### 8. src/targets.py

- **用途**：标签生成模块，计算未来 d 日收益率，为 IC 分析提供 target。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_forward_return(prices_df, d)` | 输入平表 prices_df（含 trade_date / ts_code / close）和前向天数 d，计算 `(close_{T+d} - close_T) / close_T`，返回 MultiIndex `(trade_date, ts_code)` × `forward_return` 的长表 |

- **使用示例**：
  ```python
  import pandas as pd
  from targets import calc_forward_return
  
  prices_df = pd.read_parquet("data/prices.parquet")
  target_df = calc_forward_return(prices_df, d=5)  # 5-day forward return
  ```

---

### 9. src/ic_analyzer.py

- **用途**：因子 IC（信息系数）评估模块，衡量因子值与未来收益率的截面相关性。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_ic(factors_df, target_df)` | 按 `(trade_date, ts_code)` merge 合并；`groupby('trade_date')` 计算截面 Spearman 相关；返回 `ic_series`（index = trade_date） |
  | `calc_ic_metrics(ic_series)` | 计算 IC 均值、标准差、ICIR（= 均值 / 标准差），返回字典 |
  | `plot_ic(ic_series, factor_name, show)` | 绘制 IC 时间序列柱状图 + 累计 IC 折线图双子图，返回 `Figure` 对象 |

- **使用示例**：
  ```python
  from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic
  
  ic_series = calc_ic(single_factor_df, target_df)
  metrics   = calc_ic_metrics(ic_series)  # {'ic_mean': ..., 'ic_std': ..., 'icir': ...}
  plot_ic(ic_series, factor_name="alpha006")
  ```

---

### 10. analyze_main.py

- **用途**：第二阶段总脚本，串联 targets → ic_analyzer → 有效因子筛选 → 全因子对称正交化滚动 OLS 合成 → 双回测，输出报告至控制台与 `result.txt`。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 载入 `prices.parquet` 与 `factors_clean.parquet` |
  | 2 | 调用 `calc_forward_return(prices_df, d=5)` 生成 target |
  | 3 | 遍历每个 alpha 列，依次计算 IC 时间序列、IC metrics，输出图表与分层回测 |
  | 4 | 筛选满足 `abs(IC mean) > IC_MEAN_THRESHOLD` 且 `abs(ICIR) > ICIR_THRESHOLD` 的因子，仅用于展示 |
  | 5 | 以全部 15 个因子（`alpha_cols`）为自变量，对横截面 pct-rank 收益率做滚动 OLS；每截面先调用 `symmetric_orthogonalize` 消除共线性（可通过 `orthogonalize=False` 关闭），得到合成因子后接 `LayeredBacktester` 分层回测 |
  | 6 | 以合成因子调用 `NetReturnBacktester` 进行含摩擦成本的净收益回测 |

- **使用**：
  ```bash
  python analyze_main.py
  ```
  
- **可调配置**（脚本顶部常量）：

  | 变量 | 默认值 | 说明 |
  |------|--------|------|
  | `FORWARD_DAYS` | `5` | 未来收益率天数 |
  | `IC_MEAN_THRESHOLD` | `0.015` | IC 均值绝对值阈值（仅用于展示筛选，不影响 OLS 因子集） |
  | `ICIR_THRESHOLD` | `0.20` | ICIR 绝对值阈值（同上） |
  | `SHOW_PLOTS` | `False` | 是否交互展示 IC 图表 |
  | `COMBINE_WINDOW` | `3` | 滚动 OLS 训练窗口天数 |

---

### 11. src/backtester.py

- **用途**：分层回测模块，将因子值按截面分位数切分为 N 组，计算各组等权收益，输出绩效指标表格与累计净值图。封装为 `LayeredBacktester` 类。

- **类接口**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(factor_df, target_df, num_groups=5, rf=0.03, forward_days=5, plots_dir=None)` | 接收单因子平表与 target，merge 并 dropna |
  | `run_backtest()` | 执行完整 5 步回测，返回绩效指标 DataFrame |
  | `plot(show=True)` | 绘制 G1..GN + L-S 累计净值折线图，保存至 `plots_dir` |

  **`forward_days` 参数说明**：持仓天数，与 `calc_forward_return` 中的 `d` 保持一致（默认 5）。当 `forward_days > 1` 时，每行 `group_ret` 为 d 日累积收益；模块内部自动执行 `group_ret / forward_days`，将其近似为日收益后再复利，从而避免将 d 日收益当作 1 日收益连续复利导致的收益虚高。该近似基于滑动窗口的性质：`sum_T [R_d(T)/d] ≈ sum_t r_t`（大样本 N>>d 时误差可忽略）。精确的净收益回测请使用 `NetReturnBacktester`。

- **5 步回测逻辑**：

  | 步骤 | 内容 |
  |------|------|
  | 1. 分箱 | 每日按因子值 `rank` 后 `pd.qcut` 切成 N 组（G1 最小，GN 最大） |
  | 2. 等权 | 组内所有股票等权，无需额外权重计算 |
  | 3. 组收益 | `groupby(['trade_date', 'group'])['forward_return'].mean().unstack()` |
  | 4. 多空对冲 | `L-S = G_N - G_1`，剥离大盘 Beta，反映纯选股能力 |
  | 5. 绩效指标 | 对每列（G1..GN + L-S）计算累积收益、年化收益、年化波动率、夏普比率、最大回撤 |

- **绩效指标公式**：

  | 指标 | 公式 |
  |------|------|
  | 累积收益 | `(1+r).cumprod() - 1` |
  | 年化收益 | `(1 + cum_ret)^(252/N) - 1` |
  | 年化波动率 | `r.std() × √252` |
  | 夏普比率 | `(ann_ret - rf) / ann_vol`，`rf` 默认 0.03 |
  | 最大回撤 | `min(cumval / cumval.cummax() - 1)` |

- **使用示例**：
  ```python
  from backtester import LayeredBacktester
  
  bt   = LayeredBacktester(single_factor_df, target_df, forward_days=5,
                           plots_dir=pathlib.Path("plots"))
  perf = bt.run_backtest()   # DataFrame: rows=G1..G5+L-S, cols=Cum Return/Ann Return/...
  print(perf)
  bt.plot(show=True)         # 保存至 plots/{factor_col}_backtest.png
  ```

---

### 12. src/factor_combiner.py

- **用途**：将多个 alpha 因子通过对称正交化 + 滚动 OLS 回归线性合成为一个综合因子（Synthetic Factor），供 `LayeredBacktester` 进行回测。

- **公开函数**：
  
  - `symmetric_orthogonalize(X, min_eigenvalue=1e-8)`：Löwdin 对称正交化，对 N×K 截面矩阵消除列间共线性
  - `rolling_linear_combine(factors_df, target_df, factor_cols, window=60, orthogonalize=True)`：滚动 OLS 合成，含正交化开关
  
- **`symmetric_orthogonalize` 逻辑**：
  - 计算截面协方差矩阵 `C = X.T @ X / N`
  - 特征分解 → 计算 `C^(-1/2) = V @ diag(λ^-0.5) @ V.T`
  - 输出 `X @ C^(-1/2)`，各列正交且保留原始列空间的对称结构

- **`rolling_linear_combine` 逻辑流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1. 数据准备 | merge factors + target，dropna，按日期排序 |
  | 2. 滚动预测 | 对每个预测日 T，取 [T-window, T-1] 的数据：若 `orthogonalize=True` 则逐截面正交化后堆叠，再拟合 OLS：`Y = β₁X₁ + ... + βₖXₖ` |
  | 3. 打分 | 预测日 T 当天截面同样（视开关）先正交化，再与 β 做矩阵乘积得分 |
  | 4. 裁剪 | 头部 window 个交易日自动跳过 |
  | 5. 标准化 | 对每日截面做 z-score 归一化（均值 0，标准差 1） |

- **输入**：
  - `factors_df`：平表，含 `trade_date` / `ts_code` / 若干 alpha 列
  - `target_df`：forward_return 标签（MultiIndex 或平表均可）
  - `factor_cols`：参与合成的 alpha 列名列表（至少 1 个）
  - `window`：滚动训练窗口天数（默认 3）
  - `orthogonalize`：是否在 OLS 前对每截面做对称正交化（默认 `True`，设为 `False` 退化为原始行为）

- **输出**：平表 DataFrame，列 `[trade_date, ts_code, synthetic_factor]`，z-score 标准化完毕，头部 window 天已裁剪

- **使用示例**：
  ```python
  from factor_combiner import rolling_linear_combine
  
  # 打开正交化（默认）
  synth_df = rolling_linear_combine(
      factors_df, target_df,
      factor_cols=alpha_cols,   # 全部 15 个因子
      window=60,
      orthogonalize=True,
  )
  # 关闭正交化（退化为原始 OLS）
  synth_df = rolling_linear_combine(
      factors_df, target_df,
      factor_cols=alpha_cols,
      window=60,
      orthogonalize=False,
  )
  # 然后送入 LayeredBacktester
  from backtester import LayeredBacktester
  bt = LayeredBacktester(synth_df, target_df, forward_days=5, plots_dir=pathlib.Path("plots"))
  print(bt.run_backtest())
  bt.plot()
  ```

---

### 13. src/net_backtester.py

- **用途**：纯多头、考虑摩擦成本的回测模块，模拟实盘中重叠投资组合（Overlapping Portfolio）策略的实际净收益表现，封装为 `NetReturnBacktester` 类。

- **重叠组合逻辑**：
  - 每天选出当日 top 20% 股票（等权 `daily_w`）
  - 将资金分为 `d` 个 bucket，当日权重 = 过去 d 天 `daily_w` 的滚动均值（`overlap_w`）
  - 毛收益：`GrossRet_T = overlap_w_{T-1} · R_T`
  - 换手率：`Turnover_T = 0.5 × ||overlap_w_T - overlap_w_{T-1}||₁`
  - 净收益：`NetRet_T = GrossRet_T - Turnover_T × cost_rate`
  - 前 d 天滚动窗口不完整，自动裁剪

- **类接口**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(alpha_df, prices_df, forward_days, cost_rate=0.002, rf=0.03, plots_dir=None)` | 接收合成因子和价格表，延迟计算 |
  | `run_backtest()` | 执行回测，返回 `pd.Series` 绩效指标 |
  | `plot(show=False)` | 绘制累计净值曲线，保存至 `plots_dir` |

- **7 个绩效指标**：

  | 指标 | 公式 |
  |------|------|
  | Cum Return | `(1+net_ret).cumprod().iloc[-1] - 1` |
  | Ann Return | `(1+cum_ret)^(252/N) - 1` |
  | Ann Vol | `net_ret.std() × √252` |
  | Sharpe | `(ann_ret - rf) / ann_vol` |
  | Max DD | `min(cumval / cumval.cummax() - 1)` |
  | Avg Daily Turnover | `mean(Turnover_T)` |
  | Breakeven Turnover | `(ann_gross_ret - rf) / (cost_rate × 252)`，即在当前毛收益下可承受的最大日换手率 |

- **使用示例**：
  ```python
  from net_backtester import NetReturnBacktester
  
  nb = NetReturnBacktester(synth_df, prices_df, forward_days=5,
                           cost_rate=0.002, plots_dir=pathlib.Path("plots"))
  print(nb.run_backtest())  # pd.Series: Ann Return, Sharpe, Breakeven Turnover ...
  nb.plot()                 # 保存至 plots/synthetic_factor_net.png
  ```

---

### 14. data/stock_data.db

- **用途**：本地 SQLite 数据库，存储**五张表**：`daily_price`、`daily_basic`、`stock_info`、`adj_factor`、`stock_st`。
- **生成方式**：由 `DataEngine.init_db()` + `download_data()` 自动生成，无需手动创建。
- **模式迁移**：重新调用 `init_db()` 可对旧版数据库自动补全新列与新表（`amount`、`list_date`、扩展 `daily_basic` 列、`stock_st` 表）。
- **git 状态**：已在 `.gitignore` 中排除（文件较大，且可随时重新生成）。

---

### 15. src/ml_data_prep.py

- **用途**：第三阶段 ML 数据切分模块，实现滚动时序交叉验证，防止训练集获取未来数据，封装为 `WalkForwardSplitter` 类。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(train_months=24, val_months=6, test_months=6, embargo_days=1)` | 配置窗口参数。`embargo_days` 为验证集结束到测试集开始的静默期（需 ≥ 预测周期 `d`，防止 Target 数据泄露） |
  | `split(df)` | 生成器：对含 `trade_date` 列的 DataFrame 滚动切分，每次 yield `(train_mask, val_mask, test_mask)`（布尔掩码） |
  | `n_splits(df)` | 返回给定数据集上预计可生成的折数（不实际产生掩码） |

- **数据格式要求**：输入 DataFrame 必须包含 `trade_date` 列（YYYYMMDD 整数或 datetime 均可）。

- **使用示例**：
  ```python
  from ml_data_prep import WalkForwardSplitter
  
  splitter = WalkForwardSplitter(train_months=24, val_months=6, test_months=6, embargo_days=5)
  for train_mask, val_mask, test_mask in splitter.split(df_merged):
      X_train = df_merged[train_mask][feature_cols]
      ...
  ```

---

### 16. src/lgbm_model.py

- **用途**：第三阶段 LightGBM 训练引擎，使用量化实战防过拟合超参数配置，封装为 `AlphaLGBM` 类。同时集成特征重要性可视化与 SHAP 归因分析，无需独立 `feature_importance.py`。

- **超参数设计原则**：

  | 参数 | 值 | 原因 |
  |------|-----|------|
  | `objective` | `regression_l1` (MAE) | 对金融极值（黑天鹅）不敏感 |
  | `max_depth` | 4 | 防止拟合噪音，捕获 2-3 阶交叉逻辑 |
  | `num_leaves` | 15 | 小于 `2^max_depth=16` |
  | `subsample` / `colsample_bytree` | 0.8 | 行/列随机采样，增强泛化 |
  | `learning_rate` | 0.01 | 配合 early stopping 使用 |
  | `n_estimators` | 1000 | 上限大，由 early stopping 决定实际树数 |

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `train(X_train, y_train, X_val, y_val)` | 训练并监控验证集，连续 50 轮无改善则早停 |
  | `predict(X_test)` | 使用 `best_iteration_` 预测，返回 `np.ndarray` |
  | `get_feature_importance(importance_type='gain')` | 返回按重要性排序的 `pd.DataFrame`（列：feature, importance） |
  | `plot_feature_importance(fold, save_path)` | 绘制当前折的特征重要性横向条形图 |
  | `plot_shap(X_sample, save_path, max_display=15)` | 生成 SHAP beeswarm 图（需安装 `shap`） |

- **使用示例**：
  ```python
  from lgbm_model import AlphaLGBM
  
  model = AlphaLGBM()
  model.train(X_train, y_train, X_val, y_val)
  y_pred = model.predict(X_test)
  imp_df = model.get_feature_importance()      # pd.DataFrame: feature, importance
  model.plot_shap(X_test.sample(300), save_path=Path("plots/shap.png"))
  ```

---

### 17. ml_analyze_main.py

- **用途**：第三阶段端到端总脚本，串联数据加载 → 特征合并 → 滚动训练 → 双回测器 → 图表与报告输出。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 载入 `factors_clean.parquet`（15 个清洗 alpha）与 `prices.parquet` |
  | 2 | 调用 `calc_forward_return(prices_df, d=1)` 生成 target（`forward_return`） |
  | 3 | 按 `(trade_date, ts_code)` 内联合并因子与 target，dropna；对 `forward_return` 做截面百分位排名（`cs_rank_return`）作为训练目标 |
  | 4 | 初始化 `WalkForwardSplitter`（默认：train=24m, val=6m, test=6m, embargo=1d） |
  | 5 | 循环各 Fold：以 `cs_rank_return` 为目标训练 `AlphaLGBM`，预测测试集，记录预测结果与特征重要性 |
  | 6 | 拼接所有 Fold 预测；对每只股票按 3 日滚动均值平滑，降低换手率 |
  | 7 | IC 分析（`calc_ic` / `calc_ic_metrics` / `plot_ic`），输出 IC Mean、IC Std、ICIR，保存 IC 图（`plots/ml_alpha_ic.png`） |
  | 8 | 调用 `LayeredBacktester`，生成分层净值图（`plots/ml_alpha_layered.png`） |
  | 9 | 调用 `NetReturnBacktester`，生成净收益图（`plots/ml_alpha_net.png`） |
  | 10 | 跨折平均特征重要性条形图（`plots/feature_importance.png`） |
  | 11 | SHAP beeswarm 图（最后一折测试集采样，`plots/shap_beeswarm.png`） |
  | 12 | 输出文字报告至 `result_ml.txt` |

- **可调配置**（脚本顶部常量）：

  | 变量 | 默认值 | 说明 |
  |------|--------|------|
  | `FORWARD_DAYS` | `1` | 未来收益率天数 |
  | `TRAIN_MONTHS` | `24` | 训练窗口（月） |
  | `VAL_MONTHS` | `6` | 验证窗口（月） |
  | `TEST_MONTHS` | `6` | 测试窗口（月） |
  | `EMBARGO_DAYS` | `1` | 静默期天数（需 ≥ FORWARD_DAYS） |
  | `SHAP_SAMPLE_SIZE` | `300` | SHAP 分析采样行数 |

- **输出产物**：

  | 文件 | 内容 |
  |------|------|
  | `plots/ml_alpha_ic.png` | ML 合成因子 IC 时间序列图（柱状 + 累计 IC 折线） |
  | `plots/ml_alpha_layered.png` | 分层回测累计净值图（G1..G5 + Long-Short） |
  | `plots/ml_alpha_net.png` | 净收益回测累计净值图 |
  | `plots/feature_importance.png` | 15 个因子的跨折平均特征重要性（gain）条形图 |
  | `plots/shap_beeswarm.png` | SHAP 蜂群图（特征贡献方向与大小） |
  | `result_ml.txt` | 完整文字报告：各折信息 + IC 指标 + 分层绩效表 + 净收益绩效指标 + 特征重要性排名 |

- **使用**：
  ```bash
  python ml_analyze_main.py
  # 报告自动保存至 result_ml.txt，图表保存至 plots/
  ```

- **前提**：已运行 `data_preparation_main.py`，`data/factors_clean.parquet` 和 `data/prices.parquet` 已生成。

---

## 四、推荐使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 填写 Token（编辑 src/config.py，将 'your_tushare_token' 替换为真实 Token）

# 3. 建库并下载数据（每股约 3 次接口：daily + daily_basic + adj_factor）
python - <<EOF
import sys; sys.path.insert(0, 'src')
from data_loader import DataEngine
engine = DataEngine()
engine.init_db()        # 建表 + 模式迁移（幂等）
engine.download_data()  # 下载并缓存；中断后重跑可断点续传
EOF

# 4. 运行第一阶段总脚本（因子计算 + 清洗 + 导出 Parquet）
python data_preparation_main.py

# 5. 运行第二阶段总脚本（因子 IC 评估 + 有效因子筛选 + 单因子分层回测 + 滚动线性回归合成 + 净收益回测）
python analyze_main.py

# 6. 运行第三阶段总脚本（LightGBM 合成因子 + 双回测 + 报告）
python ml_analyze_main.py
# 报告输出至 result_ml.txt，图表保存至 plots/

# 7. （可选）打开 Notebook 交互探索数据及因子
jupyter notebook notebooks/explore.ipynb
```

> **耗时估算**：数据下载约 15 ~ 40 分钟（300 只股票 × 3 次接口 + 限频 sleep）。  
> `data_preparation_main.py` 为纯内存运算，通常在 1 分钟内完成。  
> `analyze_main.py` 为纯内存运算，通常在 1 分钟内完成（含绘图）。  
> `ml_analyze_main.py` 含 LightGBM 滚动训练（约 3~4 折），通常在 2 ~ 5 分钟内完成；SHAP 计算额外约 30 秒。
