# 项目说明 (Instruction)

本文档用简明语言说明本仓库中**所有文件与代码的用途、使用方式**，并展示项目架构。随项目推进会持续更新。

---

## 一、项目概述

本项目实现**多因子风险模型与约束组合优化管理**，分为 4 个阶段：

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 数据处理和特征工程 | 已完成 |
| 2 | 机器学习预测 | 进行中 |
| 3 | 风险模型构建 | 待开始 |
| 4 | 组合优化与回测 | 待开始 |

---

## 二、项目架构（文件与目录）

```
项目根目录/
├── data/                                    # 数据存储目录（不上传 git）
│   ├── stock_data.db                        # SQLite 数据库（自动生成）
│   ├── prices.parquet                       # 日线行情 + 复权因子 + 可交易标志
│   ├── meta.parquet                         # 每日基本面 + 申万一级行业
│   ├── factors_raw.parquet                  # 原始因子值（保留 NaN）
│   └── factors_clean.parquet               # 清洗后因子值（百分位排名，不可交易为 NaN）
├── plots/                                   # 图表输出目录
├── src/
│   ├── data_preparation/                    # 第一阶段
│   │   ├── __init__.py
│   │   ├── data_loader.py                   # DataEngine：数据下载与读取
│   │   ├── factors.py                       # FactorEngine：因子计算（A/B/C 三组）
│   │   ├── preprocessor.py                  # FactorCleaner：因子清洗
│   │   └── data_preparation_main.py        # 第一阶段总脚本
│   ├── features/                            # 第二阶段（待开发）
│   ├── risk_model/                          # 第三阶段（待开发）
│   └── portfolio/                           # 第四阶段（待开发）
├── .gitignore
├── requirements.txt
└── Instruction.md
```

---

## 三、各阶段 Pipeline 详解

### 第一阶段：数据处理和特征工程

#### 总体 Pipeline 流程

```
DataEngine.download_data()
  ├── Tushare: daily_price / daily_basic / adj_factor / stock_st
  ├── Tushare: index_daily      (CSI 300 日线行情，用于 IVOL 计算)
  └── Tushare: quarterly_financials  (季度财务：ROE / OCF/Rev / 净利润)
       ↓
DataEngine.load_data()
  → 返回 8 个 DataFrame
       ↓
_compute_tradable()
  → tradable_mask (MultiIndex date×code 的 bool Series)
       ↓
FactorEngine.get_all_factors()
  → MultiIndex (date, code) × 13 个原始因子 (保留 NaN/inf)
       ↓
FactorCleaner.process_all()
  Step 1: ±inf → NaN
  Step 2: NaN 填充（行业截面中位数 → 全局截面中位数 fallback）
  Step 3: 截面百分位排名 → (0, 1]
  ↓ （tradable_mask 覆写非可交易格子为 NaN）
       ↓
导出 4 张 Parquet 宽表
  prices.parquet / meta.parquet / factors_raw.parquet / factors_clean.parquet
```

#### 文件说明

| 文件 | 类 / 函数 | 说明 |
|------|-----------|------|
| `data_loader.py` | `DataEngine` | 数据下载与读取，支持 7 张 SQLite 表 |
| `factors.py` | `FactorEngine` | 13 个因子计算，含数学工具函数库 |
| `preprocessor.py` | `FactorCleaner` | 两步清洗：中位数填充 + 百分位排名 |
| `data_preparation_main.py` | `main()` | 第一阶段端到端脚本 |

#### 数据库表结构（SQLite: stock_data.db）

| 表名 | 主键 | 说明 |
|------|------|------|
| `daily_price` | (code, date) | 日线 OHLCV + amount（千元） |
| `daily_basic` | (code, date) | 每日扩展基本面（PE、PB、换手率等 15 字段） |
| `stock_info` | code | 股票元信息：名称、申万一级行业、上市日期 |
| `adj_factor` | (code, date) | 复权因子 |
| `stock_st` | (code, start_date) | ST/\*ST 状态历史区间 |
| `index_daily` | date | CSI 300 每日收盘价 |
| `quarterly_financials` | (code, ann_date, end_date) | 季度财务：ROE、OCF/Rev、单季净利润 |

#### 因子清单

**A 组 — 微观结构与量价因子**（使用复权价格）

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_amihud_illiq` | \(\frac{1}{20}\sum\frac{|R_t|}{\text{amount}_t}\times 10^6\) | Amihud 非流动性，值越大流动性越差 |
| `factor_ivol` | \(\text{std}(\varepsilon_t)\)，20 日滚动 OLS 残差 | 特异性波动率，对 CSI300 做回归后的残差标准差 |
| `factor_realized_skewness` | \(\text{skew}(R_t, 20\text{天})\) | 日收益率 20 日偏度（bias-corrected） |
| `factor_vol_price_corr` | \(\text{Pearson}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 10)\) | 量价秩相关（等价于 Spearman 相关） |

**B 组 — 基本面与估值因子**（季度数据，PIT 对齐）

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_ep` | \(1/\text{PE}\) | 盈利收益率 |
| `factor_bp` | \(1/\text{PB}\) | 账面市值比 |
| `factor_roe` | 加权平均 ROE | 来自 `fina_indicator`，按 ann_date PIT 对齐 |
| `factor_ocf_to_revenue` | OCF / 营业收入 | 来自 `fina_indicator`，按 ann_date PIT 对齐 |
| `factor_net_profit_yoy` | \(\text{NI}_q / \text{NI}_{q-4} - 1\) | 单季净利润同比增速，PIT 对齐 |

**C 组 — 截面相对特征**

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_industry_rel_turnover` | 换手率 − 行业截面中位数 | 相对行业换手率偏离度 |
| `factor_industry_rel_bp` | BP − 行业截面中位数 | 相对行业估值偏离度 |
| `factor_ts_rel_turnover` | 换手率 / 60 日滚动均值 | 时序相对换手率 |

#### 输出 Parquet 文件

| 文件 | 主键 | 内容 |
|------|------|------|
| `prices.parquet` | (trade_date, ts_code) | 原始 OHLCV + 复权因子 + tradable（bool） |
| `meta.parquet` | (trade_date, ts_code) | 15 个每日基本面字段 + 申万一级行业 |
| `factors_raw.parquet` | (trade_date, ts_code) | 13 列原始因子值，保留 NaN |
| `factors_clean.parquet` | (trade_date, ts_code) | 13 列百分位排名 (0,1]，非可交易格子为 NaN |

#### 重要特殊细节

**1. 复权逻辑（A 组价量因子）**
所有价格类因子（含 Amihud 中的 |R_t|、IVOL 中的个股收益率）使用**前复权价格**（`adj_type='forward'`），即 `P_adj = P_raw × adj_factor / adj_factor_latest`。`vol` 和 `amount` 在任何复权模式下均不调整。

**2. Amihud 的量纲处理**
数据库中 `amount` 单位为**千元**，需乘以 1000 转换为元再计算非流动性比率，结果再乘以 10⁶ 以获得可读数量级（每百万元成交量对应的价格冲击）。

**3. PIT（Point-in-Time）财务数据对齐**
季度财务因子（ROE、OCF/Rev、净利润同比）必须严格遵守公告日期（`ann_date`）对齐原则：对任意交易日 t，只使用满足 `ann_date ≤ t` 的最新数据（`pandas.merge_asof` 实现）。使用报告期（`end_date`）而非公告日会引入未来函数。

**4. 累计财务数据转单季**
Tushare `income` 接口返回**累计值**（Q1=Q1，H1=Q1+Q2，9M=Q1+Q2+Q3，FY=全年）。转换方式：
- Q1 单季 = Q1 累计
- Q2 单季 = H1 累计 − Q1 累计
- Q3 单季 = 9M 累计 − H1 累计
- Q4 单季 = FY 累计 − 9M 累计
若同年前一季度数据缺失，则该季度单季值为 NaN（而非错误地使用累计值）。

**5. 财务数据 ann_date 保守处理**
`fina_indicator` 和 `income` 接口对同一报告期（`end_date`）可能有不同的公告日期。数据库存储时取**两者中较晚的 ann_date**，确保因子值在两张报表均已公开后才被使用。

**6. 可交易性掩码（tradable_mask）**
清洗后的因子表中，满足以下任一条件的股票-日期格子被覆写为 NaN：停牌（vol=0）、退市（close 为 NaN 或 0）、涨跌停、ST/\*ST 状态、上市未满 180 天（准新股）。下游模型 `dropna` 时自动排除。

**7. 清洗顺序：先填充再排名**
NaN 填充在百分位排名之前执行，确保每个交易日所有可交易股票都能获得有效排名值，不因财务数据缺失而丢失横截面信息。

**8. IVOL 向量化计算**
为提升性能，IVOL 的滚动 OLS 在时间维度上循环（T-d 次），但在股票截面方向**向量化**（一次批量矩阵乘法处理所有 300 只股票），避免双重循环。

---

### 第二阶段：机器学习预测

（待开始，预留）

---

### 第三阶段：风险模型构建

（待开始，预留）

---

### 第四阶段：组合优化与回测

（待开始，预留）

---

## 四、推荐使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 填写 Token（编辑 src/config.py）

# 3. 建库并下载数据
python - <<EOF
import sys
sys.path.insert(0, 'src/data_preparation')
from data_loader import DataEngine
engine = DataEngine()
engine.init_db()
engine.download_data()
EOF

# 4. 运行第一阶段总脚本（因子计算 + 清洗 + 导出 Parquet）
python src/data_preparation/data_preparation_main.py
```

---

## 五、后续更新说明

- 新增模块或脚本时，在第二节目录树与第三节各阶段说明中补充对应条目。
- 数据库表结构变动时，更新第三节「数据库表结构」表格。
- 使用流程变化时，更新第四节。
