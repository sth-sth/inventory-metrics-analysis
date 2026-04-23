# 库存缺口/多余量化归因改造计划（初稿）

## 1. 目标与边界

### 目标

- 把当前“偏差描述型”归因升级为“数量贡献型”归因。
- 对每个 SKU-仓库，输出各因素分别导致了多少库存缺口或多余（单位：件）。
- 结果可直接横向对比（同一量纲、可加总、可排序）。

### 边界

- 保留现有 Demo 模式与上传模式，不移除已有演示能力。
- 首版以可解释和稳健为优先，不引入重模型依赖。
- 当关键字段缺失时，不做拍脑袋填补，明确标注“该因素不可计算”。

## 2. 拟采用的科学方法

核心采用“反事实 + Shapley 分解”框架，避免单因素孤立比较带来的交互偏差。

### 2.1 被解释对象（统一量纲）

对每个 SKU-仓库，在同一快照时点定义：

- 缺口量（shortage_units）
- 多余量（excess_units）

并统一以“件”为口径。

### 2.2 结构方程（库存策略层）

1. 目标库存（再订货点口径）

- lead_time_demand = demand_daily * lead_time
- safety_stock = z(service_level) * sigma_demand * sqrt(lead_time)
- target_stock = lead_time_demand + safety_stock

2. 有效库存

- effective_on_hand = on_hand_qty + adj_data_lag + adj_record_error

3. 缺口/多余

- net_gap = target_stock - effective_on_hand
- shortage_units = max(net_gap, 0)
- excess_units = max(-net_gap, 0)

### 2.3 因素集合（可解释维度）

- 需求偏差因子：forecast_daily_demand 与 avg_daily_demand 的差异
- 交期偏差因子：计划交期与实际交期（lead_time_days + avg_delay_days）的差异
- 波动安全库存因子：sigma_demand 与服务水平参数对安全库存的贡献
- 库存记录偏差因子：cycle_count_accuracy 引入的可用库存修正
- 数据滞后因子：data_lag_days 导致的库存时点偏差修正

### 2.4 归因计算（Shapley）

由于 shortage/excess 带有 max 非线性，因素之间有交互，采用 Shapley 贡献值：

- 对每个因素 i，计算其在所有因素排列下的边际贡献均值
- 分别对 shortage_units 和 excess_units 计算 Shapley
- 保证可加总：各因素贡献和 = 实际值 - 基线值

输出示例（每个 SKU-仓库）：

- shortage_total = 18 件
- 其中：需求偏差 +9，交期偏差 +5，数据滞后 +3，记录误差 +1，其余 0

## 3. 基线与反事实定义

### 基线状态（计划状态）

- 需求使用 forecast_daily_demand（若缺失则该因子不可计算）
- 交期使用 lead_time_days（不含 delay）
- 记录误差与数据滞后修正置 0
- 安全库存按约定服务水平与基线 sigma 计算

### 实际状态（观测状态）

- 需求使用 avg_daily_demand
- 交期使用 lead_time_days + avg_delay_days
- 应用记录误差和数据滞后修正
- 安全库存按实际波动参数计算

## 4. 数据映射与可计算性规则

## 4.1 现有字段映射

- inventory: on_hand_qty, avg_daily_demand, lead_time_days, unit_cost
- transactions: delay_days, qty, event_type
- 衍生字段：avg_delay_days, data_lag_days, cycle_count_accuracy, forecast_daily_demand

## 4.2 缺失字段处理原则

- forecast_daily_demand 缺失：需求偏差因子标记不可计算，不参与 Shapley 因子集合
- cycle_count_accuracy 缺失：记录误差因子不可计算
- 任何不可计算因子必须在结果表中显式展示原因

## 5. 页面输出与对比逻辑（拟改造）

新增“量化归因”视图（保留现有表格作为说明层）：

- 表 1：SKU 粒度因素贡献表（单位：件）
- 表 2：按分类汇总贡献（需求/供应/仓储/流程）
- 图 1：Waterfall（基线缺口 -> 各因子贡献 -> 实际缺口）
- 图 2：Top N 缺口贡献因子排序

并支持：

- 中文/英文字段映射
- CSV 一键导出
- Demo 数据可直接演示该功能

## 6. 实施步骤（待你确认后执行）

1. 核心计算层
- 新增 src/quant_attribution.py
- 实现状态方程、反事实构造、Shapley 计算（支持缺失因子自动降维）

2. 数据准备层
- 在 app 中把现有 enrich 结果喂给 quant_attribution
- 增加必要字段校验与不可计算提示

3. 展示层
- 在归因页新增“量化归因”区块与 waterfall 图
- 保留现有业务核对清单，不删除 demo 入口

4. 验证层
- 用 demo 数据做 3 类校验：
  - 可加总性校验（贡献和一致）
  - 单位一致性校验（全部为件）
  - 缺失字段降级校验（不报错、明确提示）

5. 文档层
- 更新 README 与方法文档，补充“从偏差展示到数量归因”的说明

## 7. 交付验收标准

- 每个 SKU 至少输出 shortage_units / excess_units 与因素贡献值
- 贡献值支持排序与跨 SKU 直接对比
- 贡献和与总变化量误差在数值容忍范围内（默认 1e-6）
- Demo 模式可运行且可展示量化归因结果

## 8. 风险与缓解

- 风险：部分业务字段在用户上传数据中缺失
- 缓解：因子级降级 + 明确不可计算原因 + 不影响其他因子计算

- 风险：用户把“因素影响”误解为“因果证明”
- 缓解：文案明确“反事实贡献（attribution）不是因果识别（causal inference）”

## 9. 外部参考（已检索）

- Graves, S. C. (1994). Safety Stock versus Safety Time in MRP Controlled Production Systems. Management Science. DOI: https://doi.org/10.1287/mnsc.40.12.1678
- Lambrecht, M., et al. (2010). Safety stock or safety lead time: coping with unreliability in demand and supply. DOI: https://doi.org/10.1080/00207540903348346
- MRP performance effects due to forecast bias and demand uncertainty (2002). European Journal of Operational Research. DOI: https://doi.org/10.1016/S0377-2217(01)00134-5
- Safety stock planning under causal demand forecasting (2011). International Journal of Production Economics. DOI: https://doi.org/10.1016/j.ijpe.2011.04.017
- Newsvendor / Safety stock 的标准服务水平-分位数思想（公开教材与百科综述）
- Shapley Value（用于非线性交互下可加总贡献分解）: https://en.wikipedia.org/wiki/Shapley_value

注：正式改造时会在文档中补齐更精确的文献条目与方法适用假设说明。

## 10. 需要你确认的决策

- 服务水平参数 z 是否继续默认 1.65（约 95%）？
- 量化归因默认展示“缺口（shortage）”还是“缺口+多余”双视角？
- 是否允许在无 forecast_daily_demand 时使用滚动均值替代（默认不替代）？
