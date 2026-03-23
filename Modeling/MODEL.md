# 多因子风险模型与组合优化的数学建模说明

## 1. 问题设定与记号

本项目研究沪深 300 股票池上的短周期选股与组合管理问题。建模流程由四个层次构成：特征工程、收益预测、风险建模、组合优化。前两部分回答“哪些股票在下一持有期更值得配置”，后两部分回答“在给定收益信号下应如何控制风险并确定权重”。

设 \(t\) 表示交易日，\(\mathcal U_t\) 表示时点 \(t\) 的可交易股票集合，\(N_t = |\mathcal U_t|\) 表示该日可交易股票数。对任意股票 \(i \in \mathcal U_t\)，记：

- \(O_{i,t}, H_{i,t}, L_{i,t}, C_{i,t}\) 为当日开、高、低、收价格；
- \(V_{i,t}\) 为成交量；
- \(\phi_{i,t} \in \mathbb R^p\) 为机器学习特征向量；
- \(x_{i,t} \in \mathbb R^K\) 为风险因子暴露向量；
- \(w_t \in \mathbb R^{N_t}\) 为组合权重向量。

默认情况下，项目的收益预测使用日频开盘到开盘收益，风险模型使用日频收盘到收盘收益。两者的角色不同：前者服务于交易信号与回测收益，后者服务于协方差估计与风险约束。

---

## 2. 交易时序与收益定义

本项目采用严格的无前视偏差执行假设：

1. 在 \(t\) 日收盘后，基于当日收盘前可以观测到的数据构造特征与信号；
2. 在 \(t+1\) 日开盘建仓；
3. 在 \(t+d+1\) 日开盘平仓或滚动换仓。

因此，持有期为 \(d\) 时，与 \(t\) 日信号对应的前瞻收益定义为：

$$ r_{i,t}^{(d)} = \frac{O_{i,t+d+1}}{O_{i,t+1}} - 1. $$

项目默认 \(d=1\)，故训练标签和回测收益的基本口径为：

$$ r_{i,t} = \frac{O_{i,t+2}}{O_{i,t+1}} - 1. $$

这一收益定义与第四阶段回测器中实际使用的开盘到开盘持有收益完全一致，因此收益预测目标与回测收益口径保持统一。

---

## 3. 特征工程

### 3.1 数据来源与前复权处理

基础数据包括股票日线行情、日频基本面、复权因子、ST 区间、CSI 300 指数行情以及季度财务数据。价量因子统一基于前复权价格计算。若 \(A_{i,t}\) 为复权因子，\(T_i\) 为样本末端日期，则前复权价格定义为：

$$ P_{i,t}^{\mathrm{fwd}} = P_{i,t}^{\mathrm{raw}} \cdot \frac{A_{i,t}}{A_{i,T_i}}. $$

成交量与成交额不做复权调整。

### 3.2 财务数据的 PIT 对齐

季度财务变量只允许使用在时点 \(t\) 之前已经公告的数据。若某财务变量记为 \(q\)，其公告日为 \(\tau\)，则交易日 \(t\) 对应的可用值为：

$$ q_{i,t} = q_{i,\tau^\star}, \qquad \tau^\star = \max\{\tau : \tau \le t\}. $$

这种 Point-in-Time 对齐保证了财务因子不包含未来信息。

### 3.3 可交易性掩码

项目先构造可交易性掩码 \(m_{i,t} \in \{0,1\}\)，只有 \(m_{i,t}=1\) 的股票-日期样本才会进入后续建模。不可交易条件包括：

- 停牌：\(V_{i,t}=0\)；
- 退市或价格异常：\(C_{i,t}\) 缺失或为 0；
- 涨跌停且收盘价贴近高点或低点；
- ST 或 *ST 状态；
- 上市未满 180 天。

记：

$$ m_{i,t} = \mathbf 1\{\text{股票 } i \text{ 在 } t \text{ 日可交易}\}. $$

清洗后的特征在所有 \(m_{i,t}=0\) 的位置上会被统一置为缺失值，从而在下游训练和回测中自动排除。

### 3.4 因子集合

当前实现共使用 66 个预测特征，其中 52 个以 `factor_*` 的形式输出，14 个以 `alpha*` 的形式输出。按照经济含义与构造方式，可分为微观结构与量价因子、基本面与估值因子、截面相对与补充特征以及 Alpha101 子集因子四组。完整因子列表如下。

#### 3.4.1 A 组：微观结构与量价因子（33 个）

| 因子 | 公式 | 说明 |
| --- | --- | --- |
| `factor_amihud_illiq` | $\frac{1}{20}\sum\frac{\|R_t\|}{\text{amount}_t}\times 10^6$ | Amihud 非流动性，值越大流动性越差 |
| `factor_ivol` | $\text{std}(\varepsilon_t)$，20 日滚动 OLS 残差 | 特异性波动率，对 CSI300 做回归后的残差标准差 |
| `factor_beta` | 20 日滚动 OLS 回归斜率 | 对 CSI300 的市场 beta，与 IVOL 同回归 |
| `factor_realized_skewness` | $\text{skew}(R_t, 20\text{ days})$ | 日收益率 20 日偏度（bias-corrected） |
| `factor_vol_price_corr` | $\text{Corr}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 10)$ | 量价秩相关（Spearman），10 日 |
| `factor_vol_price_corr_20d` | $\text{Corr}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 20)$ | 量价秩相关（Spearman），20 日 |
| `factor_vw_return_5d` | $\sum(R_t \cdot V_t, 5) / \sum(V_t, 5)$ | 成交量加权收益率，5 日 |
| `factor_vw_return_10d` | $\sum(R_t \cdot V_t, 10) / \sum(V_t, 10)$ | 成交量加权收益率，10 日 |
| `factor_vol_oscillator` | $\text{MA}(\text{vol}, 5) / \text{MA}(\text{vol}, 20)$ | 量震荡指标：短期/中期成交量比值 |
| `factor_net_buy_proxy_5d` | $\text{mean}\!\left(\frac{C-O}{H-L+\epsilon}\cdot V,\ 5\right)$ | 净买入代理（方向性成交），5 日 |
| `factor_net_buy_proxy_10d` | $\text{mean}\!\left(\frac{C-O}{H-L+\epsilon}\cdot V,\ 10\right)$ | 净买入代理（方向性成交），10 日 |
| `factor_momentum_1d` | $(C_t - C_{t-1}) / C_{t-1}$ | 1 日价格动量 |
| `factor_momentum_3d` | $(C_t - C_{t-3}) / C_{t-3}$ | 3 日价格动量 |
| `factor_momentum_5d` | $(C_t - C_{t-5}) / C_{t-5}$ | 5 日价格动量 |
| `factor_momentum_10d` | $(C_t - C_{t-10}) / C_{t-10}$ | 10 日价格动量 |
| `factor_momentum_20d` | $(C_t - C_{t-20}) / C_{t-20}$ | 20 日价格动量 |
| `factor_trend_strength` | $\text{mom}_{20} / \sum(\|R_t\|, 20)$ | 趋势强度（IR 代理），接近 ±1 表示单边趋势 |
| `factor_drawdown_from_high` | $(C - \text{tsmax}(C, 60)) / \text{tsmax}(C, 60)$ | 距 60 日最高点的跌幅，恒 ≤ 0 |
| `factor_bias_5d` | $C / \text{MA}(C, 5) - 1$ | 5 日均线乖离率 |
| `factor_bias_10d` | $C / \text{MA}(C, 10) - 1$ | 10 日均线乖离率 |
| `factor_bias_20d` | $C / \text{MA}(C, 20) - 1$ | 20 日均线乖离率 |
| `factor_rvol_5d` | $\text{std}(R_t, 5)$ | 已实现波动率，5 日 |
| `factor_rvol_10d` | $\text{std}(R_t, 10)$ | 已实现波动率，10 日 |
| `factor_rvol_20d` | $\text{std}(R_t, 20)$ | 已实现波动率，20 日（总波动，非残差） |
| `factor_realized_kurtosis` | $\text{kurt}(R_t, 20)$ | 20 日已实现超额峰度（bias-corrected） |
| `factor_hl_range` | $\text{mean}((H-L)/C,\ 10)$ | 日内振幅均值，10 日滚动（流动性 / 波动代理） |
| `factor_downside_vol` | $\text{std}(R_t[R_t < 0],\ 20)$ | 下行波动率（仅负收益日），20 日 |
| `factor_upside_vol` | $\text{std}(R_t[R_t > 0],\ 20)$ | 上行波动率（仅正收益日），20 日 |
| `factor_gap_return` | $(O_t - C_{t-1}) / C_{t-1}$ | 隔夜收益（开盘相对前收） |
| `factor_turnover_volatility` | $\text{std}(\text{turnover\_rate},\ 20)$ | 换手率 20 日滚动标准差 |
| `factor_distance_from_low` | $(C - \text{tsmin}(C, 60)) / \text{tsmin}(C, 60)$ | 距 60 日最低点的涨幅，恒 ≥ 0 |
| `factor_momentum_acceleration` | $\text{mom}_{5d} - \text{mom}_{10d}$ | 动量加速度（短期减中期） |
| `factor_return_consistency` | 20 日内正收益日占比 | 收益一致性，值域 [0, 1] |

#### 3.4.2 B 组：基本面与估值因子（7 个）

| 因子 | 公式 | 说明 |
| --- | --- | --- |
| `factor_ep` | $1/\text{PE}$ | 盈利收益率 |
| `factor_bp` | $1/\text{PB}$ | 账面市值比 |
| `factor_roe` | 加权平均 ROE | 来自 `fina_indicator`，按 ann_date PIT 对齐 |
| `factor_log_mv` | $\log(\text{total\_mv})$ | 对数市值，规模因子 |
| `factor_ps` | $1/\text{ps ttm}$ | 市销率倒数（daily_basic） |
| `factor_dv_ratio` | dv_ratio | 股息率（daily_basic） |
| `factor_circ_mv_ratio` | circ_mv / total_mv | 流通市值占比 |

#### 3.4.3 C 组：截面相对与补充特征（12 个）

| 因子 | 公式 | 说明 |
| --- | --- | --- |
| `factor_industry_rel_turnover` | 换手率 − 行业截面中位数 | 相对行业换手率偏离（差值） |
| `factor_industry_rel_bp` | BP − 行业截面中位数 | 相对行业估值偏离（差值） |
| `factor_industry_rel_mv` | $\log(\text{mv}) - \text{行业中位数}(\log(\text{mv}))$ | 相对行业市值偏离（差值） |
| `factor_industry_rel_ep` | EP − 行业截面中位数 | 相对行业盈利收益率偏离（差值） |
| `factor_industry_rel_roe` | ROE − 行业截面中位数 | 相对行业 ROE 偏离（差值） |
| `factor_ts_rel_turnover` | 换手率 / 60 日滚动均值 | 时序相对换手率（比值） |
| `factor_ind_rel_momentum_5d` | $\text{mom}_{5d} - \text{行业中位数}(\text{mom}_{5d})$ | 行业相对 5 日动量（差值） |
| `factor_ind_rel_momentum_10d` | $\text{mom}_{10d} - \text{行业中位数}(\text{mom}_{10d})$ | 行业相对 10 日动量（差值） |
| `factor_ind_rel_momentum_20d` | $\text{mom}_{20d} - \text{行业中位数}(\text{mom}_{20d})$ | 行业相对 20 日动量（差值） |
| `factor_ind_rel_turnover_ratio` | 换手率 / 行业截面均值 | 行业相对换手率（比值，尺度不变） |
| `factor_ind_rel_vol` | $\text{rvol}_{20d} / \text{行业均值}(\text{rvol}_{20d})$ | 行业相对已实现波动率（比值，以 1 为中心） |
| `factor_5_day_reversal` | $(\text{close} - \text{delay}(\text{close}, 5)) / \text{delay}(\text{close}, 5)$ | 5 日收盘价反转因子 |

#### 3.4.4 Alpha101 子集因子（14 个）

| 因子 | 公式 | 说明 |
| --- | --- | --- |
| `alpha001` | `rank(ts_argmax(signedpower((ret<0)?stddev(ret,20):close, 2), 5)) - 0.5` | 负收益时用波动率替代收盘价，对 5 日最高值的时序位置排名 |
| `alpha003` | `-1 * correlation(rank(open), rank(volume), 10)` | 开盘价排名与成交量排名的滚动相关性取反 |
| `alpha006` | `-1 * correlation(open, volume, 10)` | 开盘价与成交量的滚动相关性取反 |
| `alpha012` | `sign(delta(volume, 1)) * (-1 * delta(close, 1))` | 成交量方向乘以收盘价变动反向 |
| `alpha038` | `(-1 * rank(ts_rank(close, 10))) * rank(close/open)` | 近期高位且高涨幅的股票做空 |
| `alpha040` | `(-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)` | 高价波动率乘以高价-成交量相关性 |
| `alpha041` | `sqrt(high * low) - vwap` | 高低价几何均值与成交均价之差（精确 vwap） |
| `alpha042` | `rank(vwap - close) / rank(vwap + close)` | 收盘价相对 vwap 的位置比率（delay-0 均值回归） |
| `alpha054` | `(-1 * (low-close) * open^5) / ((low-high) * close^5)` | 基于开/收/高/低价五次幂的日内动量 |
| `alpha072` | `rank(decay_linear(corr((H+L)/2, adv40, 8), 10)) / rank(decay_linear(corr(ts_rank(vwap,3), ts_rank(vol,18), 6), 2))` | 中价-量相关性与 vwap-量 ts_rank 相关性之比 |
| `alpha088` | `min(rank(decay_linear((rank(O)+rank(L))-(rank(H)+rank(C)),8)), ts_rank(decay_linear(corr(ts_rank(C,8),ts_rank(adv60,20),8),6),2))` | 价格结构排名差与成交量相关性的最小值 |
| `alpha094` | `signedpower(rank(vwap - ts_min(vwap,11)), ts_rank(corr(ts_rank(vwap,19), ts_rank(adv60,4), 18), 2)) * -1` | vwap 距历史低点的幂次排名因子 |
| `alpha098` | `rank(decay_linear(corr(vwap,sum(adv5,26),4),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(O),rank(adv15),20),8),6),8))` | vwap-量相关性与开盘价-量相关性低点时序排名之差 |
| `alpha101` | `(close - open) / (high - low + 0.001)` | 日内动量：价格区间归一化的涨跌幅 |

### 3.5 因子清洗与横截面标准化

对任意原始因子 \(x_{i,t}^{(j)}\)，项目采用如下三步处理：

1. 将 \(\pm \infty\) 替换为缺失值；
2. 用行业横截面中位数填补缺失，若行业中位数仍不存在，则退回到市场横截面中位数；
3. 在每个交易日做横截面百分位排名。

若填补后的值记为 \(\tilde x_{i,t}^{(j)}\)，则最终输入模型的特征定义为：

$$ z_{i,t}^{(j)} = \operatorname{PctRank}_t(\tilde x_{i,t}^{(j)}) \in (0,1]. $$

这种处理同时解决了三个问题：缺失值、量纲不一致和极端值敏感性。由于最终进入机器学习模型的是截面分位数而不是原始值，模型训练更聚焦于相对排序信息而非绝对数值尺度。

---

## 4. 机器学习收益预测模型

### 4.1 建模目标

第二阶段的核心任务是学习从高维特征到未来收益排序的映射关系。记 \(\phi_{i,t}\) 为股票 \(i\) 在 \(t\) 日的特征向量，则模型输出的合成 Alpha 为：

$$ \hat \alpha_{i,t} = f_\theta(\phi_{i,t}), $$

其中 \(f_\theta\) 由 LightGBM 回归器表示。

从建模角度看，这一步不再使用传统线性横截面多因子收益回归，而是让树模型直接学习非线性关系、交互效应与阈值结构。

### 4.2 行业中性化标签

收益预测标签不是直接使用原始未来收益，而是先做行业中性化。记原始前瞻收益为 \(r_{i,t}\)，则行业中性收益定义为：

$$ \tilde r_{i,t} = r_{i,t} - \frac{1}{|\mathcal G_{g(i,t),t}|} \sum_{k \in \mathcal G_{g(i,t),t}} r_{k,t}. $$

若某行业在某一天只有 1 只有效股票，则该行业均值不具有统计意义，对应样本被剔除。

在行业中性收益基础上，再对每个交易日做横截面分位数排名，得到训练标签：

$$ y_{i,t} = \operatorname{PctRank}_t(\tilde r_{i,t}) \in (0,1]. $$

这一设计的含义是，模型学习的是行业内部的相对优劣，而不是行业轮动本身。评价与回测仍使用原始收益 \(r_{i,t}\)，从而保持策略收益解释的真实性。

### 4.3 Walk-Forward 时序验证

训练集、验证集和测试集采用严格的时间顺序切分。当前实现的默认设置为：

- 训练集 15 个月；
- 验证集 3 个月；
- 测试集 3 个月；
- embargo 间隔为 1 个交易日；
- 使用扩展窗口模式。

如果把 1 个月近似为 21 个交易日，则每一折均满足：

$$ \mathrm{Train} \rightarrow \mathrm{Validation} \rightarrow \mathrm{Embargo} \rightarrow \mathrm{Test}. $$

这种切分方式避免了随机抽样在时序数据上的信息泄漏，并且允许模型在每一折用验证集进行 early stopping。

### 4.4 LightGBM 学习问题

LightGBM 本质上在求解一个监督学习问题：

$$ \hat y_{i,t} = f_\theta(\phi_{i,t}), \qquad \theta^\star = \arg\min_\theta \sum_{(i,t)\in\mathcal D} \ell\!\left(y_{i,t}, f_\theta(\phi_{i,t})\right). $$

项目中的实现使用浅树、样本子采样、特征子采样和 early stopping，以降低金融数据低信噪比环境中的过拟合风险。

### 4.5 Alpha 平滑

各折测试样本的预测值拼接后，项目对每只股票的 Alpha 做时间序列平滑。若启用 EMA，则平滑后的信号为：

$$ \alpha_{i,t}^{\mathrm{EMA}} = \beta \hat \alpha_{i,t} + (1-\beta)\alpha_{i,t-1}^{\mathrm{EMA}}. $$

若启用滚动均值，则先做：

$$ \alpha_{i,t}^{\mathrm{MA}} = \frac{1}{h}\sum_{s=0}^{h-1} \hat \alpha_{i,t-s}. $$

当前默认配置启用了 \(\beta=0.5\) 的 EMA 平滑，最终得到的 \(\alpha_{i,t}\) 即 `ml_alpha.parquet` 中保存的合成 Alpha。

### 4.6 IC 与回测评价

对每个交易日，策略信号与下一期原始收益之间的截面秩相关定义为：

$$ \mathrm{IC}_t = \operatorname{corr}_{\mathrm{Spearman}}(\alpha_{\cdot,t}, r_{\cdot,t}). $$

基于此进一步计算平均 IC 和 ICIR：

$$ \overline{\mathrm{IC}} = \frac{1}{T}\sum_{t=1}^{T}\mathrm{IC}_t, \qquad \mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{\mathrm{Std}(\mathrm{IC}_t)}. $$

除 IC 之外，项目还通过行业中性分层回测与纯多头净收益回测检验信号的单调性、可交易性和扣费后的真实表现。

---

## 5. 多因子风险模型

### 5.1 风险模型的线性结构

风险模型假设个股收益可分解为公共因子驱动部分与特异性部分：

$$ r_t^{\mathrm{risk}} = X_t f_t + \varepsilon_t. $$

其中：

- \(r_t^{\mathrm{risk}} \in \mathbb R^{N_t}\) 为当日股票的 close-to-close 收益向量；
- \(X_t \in \mathbb R^{N_t \times K}\) 为风险因子暴露矩阵；
- \(f_t \in \mathbb R^K\) 为因子收益；
- \(\varepsilon_t \in \mathbb R^{N_t}\) 为特异性收益。

风险模型使用 close-to-close 收益，是因为这类收益更适合描述日度方差结构；收益预测与回测则使用 open-to-open 收益，以匹配实际交易时序。两套收益口径各司其职，并不冲突。

### 5.2 风险因子暴露

当前实现中，风险暴露由 5 个风格因子和若干申万一级行业哑变量构成：

$$ x_{i,t} = [\mathrm{size}_{i,t}, \mathrm{beta}_{i,t}, \mathrm{momentum}_{i,t}, \mathrm{volatility}_{i,t}, \mathrm{value}_{i,t}, d_{i,t,1}, \dots, d_{i,t,J}]^\top. $$

其中 \(d_{i,t,j}\in\{0,1\}\) 为行业哑变量，表示股票 \(i\) 是否属于行业 \(j\)。

五个风格因子的定义如下。

#### 5.2.1 Size

$$ \mathrm{size}_{i,t} = \log(\mathrm{MV}_{i,t}). $$

#### 5.2.2 Beta

对个股收盘收益与 CSI 300 指数收益做 60 日滚动回归：

$$ r_{i,s}^{\mathrm{cc}} = a_{i,t} + \beta_{i,t} r_{m,s}^{\mathrm{cc}} + u_{i,s}, \qquad s=t-59,\dots,t. $$

其中 \(\beta_{i,t}\) 即风格因子 Beta 的暴露值。

#### 5.2.3 Momentum

项目采用跳过近一个月的中长期动量：

$$ \mathrm{momentum}_{i,t} = \frac{C_{i,t-21}}{C_{i,t-252}} - 1. $$

该定义试图保留中期趋势信息，同时减弱短期反转噪声。

#### 5.2.4 Volatility

$$ \mathrm{volatility}_{i,t} = \mathrm{Std}(r_{i,s}^{\mathrm{cc}} : s=t-19,\dots,t). $$

#### 5.2.5 Value

$$ \mathrm{value}_{i,t} = \frac{1}{\mathrm{PE}_{i,t}}, \qquad \mathrm{PE}_{i,t}\le 0 \Rightarrow \mathrm{value}_{i,t}=\mathrm{NaN}. $$

### 5.3 风格因子的横截面标准化

风格因子先做横截面 \(\pm 3\sigma\) 截尾，再做 z-score 标准化。若原始暴露为 \(x_{i,t,k}\)，则截尾值为：

$$ x_{i,t,k}^{\mathrm{clip}} = \min\{\max(x_{i,t,k}, \mu_{t,k}-3\sigma_{t,k}), \mu_{t,k}+3\sigma_{t,k}\}. $$

标准化后的暴露为：

$$ z_{i,t,k} = \frac{x_{i,t,k}^{\mathrm{clip}} - \bar x_{t,k}^{\mathrm{clip}}}{s_{t,k}^{\mathrm{clip}}}. $$

行业因子保持 0/1 哑变量形式，不做 z-score 变换。最终暴露矩阵 \(X_t\) 由标准化风格因子与行业哑变量拼接而成。

### 5.4 因子收益的横截面加权回归

在每个交易日 \(t\)，项目在可交易股票集合 \(\mathcal U_t\) 上估计截面回归：

$$ r_t^{\mathrm{risk}} = X_t f_t + \varepsilon_t. $$

采用市值加权最小二乘估计：

$$ \hat f_t = \arg\min_f \sum_{i\in\mathcal U_t} \mathrm{MV}_{i,t}\big(r_{i,t}^{\mathrm{risk}} - x_{i,t}^\top f\big)^2. $$

数值实现等价于对每个样本乘以 \(\sqrt{\mathrm{MV}_{i,t}}\) 后再进行最小二乘求解。这样做的原因是，大市值股票的价格噪声通常更低，其风险暴露对指数化组合风险也更有代表性。

回归输出两类对象：

1. 因子收益序列 \(\{\hat f_t\}\)；
2. 特异性残差序列 \(\{\hat \varepsilon_{i,t}\}\)。

### 5.5 因子协方差矩阵

设滚动窗口长度为 \(H=60\) 个交易日，则因子协方差矩阵定义为：

$$ F_t = \operatorname{Cov}(\hat f_{t-H+1},\dots,\hat f_t) + \lambda I. $$

其中 \(\lambda > 0\) 是 ridge 正则项，用于保证矩阵正定性并稳定 Cholesky 分解。

进一步令：

$$ F_t = L_t L_t^\top. $$

项目保存的是 \(L_t^\top\) 的长表形式，为后续二阶锥优化提供直接输入。

### 5.6 特异性风险

项目假设在给定公共因子后，股票之间的残差相关性可以忽略，因此只估计对角特异性风险矩阵：

$$ \Delta_t = \operatorname{diag}(\Delta_{11,t},\dots,\Delta_{N_tN_t,t}). $$

单只股票的特异性方差由滚动 60 日残差方差估计：

$$ \Delta_{ii,t} = \max\{\operatorname{Var}(\hat \varepsilon_{i,t-59:t}), \delta_{\min}^2\}. $$

当某股票有效历史不足最小样本长度时，项目用当日横截面中位数回填对应方差。最终保存的是特异性标准差：

$$ \delta_{i,t} = \sqrt{\Delta_{ii,t}}. $$

### 5.7 协方差矩阵与风险分解

任意交易日 \(t\) 的股票协方差矩阵写为：

$$ \Sigma_t = X_t F_t X_t^\top + \Delta_t. $$

对任意权重向量 \(w_t\)，组合方差为：

$$ w_t^\top \Sigma_t w_t = w_t^\top X_t F_t X_t^\top w_t + w_t^\top \Delta_t w_t. $$

利用 \(F_t = L_t L_t^\top\)，上式可改写为：

$$ w_t^\top \Sigma_t w_t = \|L_t^\top (X_t^\top w_t)\|_2^2 + \|\delta_t \odot w_t\|_2^2. $$

这一分解具有两个直接作用：

1. 不需要显式构造 \(N_t \times N_t\) 的大协方差矩阵；
2. 风险项可以直接写成二范数平方之和，便于在优化器中表示为 SOCP 结构。

### 5.8 风险模型验证

项目以可交易股票构成的市值加权组合为检验对象。设组合权重为：

$$ w_{i,t}^{\mathrm{bench}} = \frac{\mathrm{MV}_{i,t}}{\sum_{k\in\mathcal U_t}\mathrm{MV}_{k,t}}. $$

则预测方差为：

$$ \hat v_t = (w_t^{\mathrm{bench}})^\top \Sigma_t w_t^{\mathrm{bench}}. $$

已实现方差由组合收益的滚动样本方差给出：

$$ v_t^{\mathrm{real}} = \operatorname{Var}(r_{p,t-h+1},\dots,r_{p,t}), \qquad h=60. $$

项目报告中使用的验证指标包括：

$$ \mathrm{Bias\ Ratio} = \frac{\mathbb E[\hat v_t]}{\mathbb E[v_t^{\mathrm{real}}]}, \qquad \mathrm{RMSE}_{\mathrm{var}} = \sqrt{\mathbb E[(\hat v_t - v_t^{\mathrm{real}})^2]}, \qquad \mathrm{RMSE}_{\mathrm{vol}} = \sqrt{\mathbb E[(\sqrt{\hat v_t} - \sqrt{v_t^{\mathrm{real}}})^2]}. $$

此外还会计算预测方差与已实现方差之间的 Pearson 相关、Spearman 相关与 \(R^2\)，用于评价模型的解释能力与校准程度。

---

## 6. 组合优化模型

### 6.1 Alpha 去均值

进入优化器前，先对当日 Alpha 做横截面去均值：

$$ \alpha_{i,t}^{c} = \alpha_{i,t} - \frac{1}{N_t}\sum_{k\in\mathcal U_t}\alpha_{k,t}. $$

这一步不会改变股票排序，但会使信号围绕 0 对称分布，从而让换手惩罚参数和风险惩罚参数具有更稳定的数值尺度。

### 6.2 单期优化目标

对每个交易日 \(t\)，优化器求解如下问题：

$$ \max_{w_t}\ \alpha_t^{c\top} w_t - \frac{\lambda_{\mathrm{TO}}}{2}\|w_t - w_{t-1}\|_1 - \frac{\mu_{\mathrm{risk}}}{2} w_t^\top \Sigma_t w_t. $$

其中：

- \(\lambda_{\mathrm{TO}}\) 为换手惩罚系数；
- \(\mu_{\mathrm{risk}}\) 为风险厌恶系数；
- 若不启用风险模型，则 \(\mu_{\mathrm{risk}}=0\)，问题退化为线性规划；
- 若启用风险模型，则目标中包含二次风险项，问题可写为 SOCP。

这里的 \(\lambda_{\mathrm{TO}}\) 是一个无量纲的策略参数，而不是实际交易费率。实际交易费率只在回测收益中扣除，不直接进入优化目标。

### 6.3 约束条件

#### 6.3.1 满仓与纯多头

$$ \mathbf 1^\top w_t = 1, \qquad w_t \ge 0. $$

#### 6.3.2 单股权重上限

$$ w_{i,t} \le w_{\max}. $$

当前实现默认 \(w_{\max}=0.05\)。

#### 6.3.3 换手率硬约束

定义单边换手率为：

$$ \mathrm{TO}_t = \frac{1}{2}\|w_t - w_{t-1}\|_1. $$

则可选硬约束为：

$$ \frac{1}{2}\|w_t - w_{t-1}\|_1 \le \tau_{\max}. $$

#### 6.3.4 行业偏离约束

记 \(X_t^{\mathrm{ind}}\) 为股票-行业哑变量矩阵，\(w_t^{\mathrm{bench,ind}}\) 为基准行业权重向量，则行业偏离约束为：

$$ |(X_t^{\mathrm{ind}})^\top w_t - w_t^{\mathrm{bench,ind}}| \le \delta_{\mathrm{ind}}. $$

行业基准权重按全部基准股票的市值权重计算，而不是只在当日可交易股票子集内归一化。这样可以使组合行业结构与真实基准保持一致。若约束不可行，项目会逐步放宽 \(\delta_{\mathrm{ind}}\) 直到求解成功或达到容差上限。

#### 6.3.5 方差上限约束

若启用风险上限，则要求：

$$ w_t^\top \Sigma_t w_t \le \bar \sigma_t^2. $$

#### 6.3.6 风格中性约束

若启用风格中性，设股票级基准权重为 \(w_t^{\mathrm{bench,stock}}\)，主动权重为：

$$ w_t^{\mathrm{active}} = w_t - w_t^{\mathrm{bench,stock}}. $$

对每一个风格因子 \(k\)，要求：

$$ |(w_t^{\mathrm{active}})^\top x_{t,k}^{\mathrm{style}}| \le \tau_{\mathrm{style}}. $$

这类约束的作用是限制主动组合相对基准在 Size、Beta、Momentum、Volatility、Value 等风格方向上的偏离。

### 6.4 风险项的二阶锥表示

由于第三阶段已经得到风险分解：

$$ w_t^\top \Sigma_t w_t = \|L_t^\top (X_t^\top w_t)\|_2^2 + \|\delta_t \odot w_t\|_2^2, $$

因此目标函数中的风险惩罚和风险约束都可以表示为二范数平方之和。这一结构可以被 cvxpy 直接转换为二阶锥问题并由 CLARABEL 等求解器高效求解。

### 6.5 重叠组合

当预测持有期 \(d>1\) 时，项目采用重叠持仓机制。设优化器每天给出目标权重 \(w_t^\star\)，则实际持仓定义为最近 \(d\) 个目标组合的平均：

$$ \bar w_t = \frac{1}{d}\sum_{s=0}^{d-1} w_{t-s}^\star. $$

这种做法使得每天只调整大约 \(1/d\) 的组合，能够显著降低换手，并且保证回测持有期与信号预测周期一致。

### 6.6 回测收益与交易成本

对实际持仓 \(\bar w_t\)，股票的开盘到开盘收益定义为：

$$ r_{i,t}^{\mathrm{oo}} = \frac{O_{i,t+1}}{O_{i,t}} - 1. $$

则组合毛收益为：

$$ r_{p,t}^{\mathrm{gross}} = \bar w_{t-1}^\top r_t^{\mathrm{oo}}. $$

单边换手率为：

$$ \mathrm{TO}_t = \frac{1}{2}\|\bar w_t - \bar w_{t-1}\|_1. $$

净收益则定义为：

$$ r_{p,t}^{\mathrm{net}} = r_{p,t}^{\mathrm{gross}} - c \cdot \mathrm{TO}_t, \qquad c = 0.002. $$

在此基础上进一步计算累计收益、年化收益、年化波动率、Sharpe 比率、最大回撤、跟踪误差和信息比率等指标。

---

## 7. 模型结构的特点

本项目的数学建模结构具有三个鲜明特点。

### 7.1 收益预测与风险预测分离

收益侧使用 66 维高维特征输入 LightGBM，强调非线性与交互效应；风险侧使用低维线性因子结构，强调协方差的可解释分解和可优化性。这种“Alpha 模型负责找收益，Risk 模型负责管风险”的结构是现代量化策略中非常常见的做法。

### 7.2 信号训练强调行业内部排序

训练标签在进入模型前先做行业中性化，因此模型主要学习行业内部相对收益差异。这样可以降低行业轮动对训练目标的污染，使 Alpha 更接近“纯选股信号”。

### 7.3 优化器同时平衡收益、换手与风险

第四阶段不是简单的等权买入高分股，而是在统一的凸优化框架中同时考虑：

- 收益信号的方向性；
- 交易稳定性与换手约束；
- 行业、风格和协方差风险暴露。

因此，最终组合既不是纯粹的打分排序结果，也不是传统意义上的单期均值方差解，而是一个具有交易现实约束的动态最优解。

---

## 8. 模型假设与局限

尽管当前实现已经形成了完整的建模闭环，但仍然依赖若干近似假设。

1. 公共因子结构假设：风险模型假设个股风险可以由少数公共因子和独立特异项近似刻画。
2. 对角特异性风险假设：给定公共因子后，股票间残差相关性被忽略。
3. 历史稳定性假设：无论是机器学习收益模型还是风险模型，都默认过去样本中的统计规律在未来一段时间内具有延续性。
4. 行业中性训练假设：项目更强调行业内部选股能力，因此可能弱化部分行业轮动收益。
5. 短周期交易假设：收益标签和回测收益以开盘到开盘为核心，模型更适合短周期滚动换仓，而不是长期基本面配置。

这些假设并不意味着模型无效，而是说明模型的适用范围主要集中在中短周期、宽度较高、行业约束明确的量化选股与组合优化场景。

---

## 9. 总结

整个项目可以概括为一个两层预测、一层优化的体系：

1. 第一阶段构造多维特征并消除缺失、量纲和极端值问题；
2. 第二阶段学习从特征到未来收益排序的非线性映射，得到合成 Alpha；
3. 第三阶段估计由风格因子、行业因子和特异性风险构成的协方差结构；
4. 第四阶段在收益、换手、行业、风格和风险约束下求解最优组合权重。

若将本项目的核心思想浓缩为一句话，则可以表述为：先利用机器学习回答“买什么”，再利用多因子风险模型和凸优化回答“买多少”。这一结构使得收益预测、风险控制和交易约束能够在同一套日频系统中形成一致的数学闭环。
