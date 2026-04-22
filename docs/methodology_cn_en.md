# Methodology (CN/EN)

## 1. Inventory Metric Definitions / 指标定义

- DOH = on_hand_qty / avg_daily_demand
- lead_time_demand = avg_daily_demand * lead_time_days
- safety_stock = z * std(avg_daily_demand) * sqrt(lead_time_days)
- reorder_point = lead_time_demand + safety_stock
- coverage_gap (Gap) = on_hand_qty - reorder_point

Interpretation / 解读:
- Gap < 0 means inventory is below protection line; stockout probability rises.
- Gap < 0 表示库存低于保护线，缺货概率升高。
- Gap > 0 means inventory can still cover expected demand during lead time.
- Gap > 0 表示交期内覆盖能力更高。

## 2. Alert Logic / 报警逻辑

- Stockout Risk when coverage_gap < threshold
- 缺货风险：coverage_gap 低于阈值
- Overstock Risk when DOH > threshold, but only after stockout check passes for the same SKU
- 超储风险：DOH 超过阈值，但同一 SKU 先通过缺货检查后才判定
- Supplier Delay when receipt delay_days > threshold
- 供应商延迟：收货 delay_days 超过阈值

When a SKU appears to satisfy both stockout and overstock signals, the app keeps shortage as the higher-priority operating status and emits a logic-review checklist item.
当同一 SKU 表面上同时满足缺货和超储信号时，系统会优先保留缺货作为经营状态，并额外提示口径复核。

## 3. Attribution Logic / 归因逻辑

Attribution factors:
- Demand Surge / 需求激增
- Supplier Delay / 供应延迟
- Planning Gap / 计划缺口

Evidence is calculated by deviations from benchmark, then normalized to impact_score.
通过相对基准偏离计算证据强度，并归一化为 impact_score。

confidence is based on sample coverage by observed days.
confidence 基于观测天数覆盖度。

## 4. Decision Support Logic / 决策支持逻辑

- base_order_qty = max(reorder_point - on_hand_qty, 0)
- recommended_order_qty = ceil(base_order_qty * class_weight * service_uplift)
- investment = recommended_order_qty * unit_cost
- risk_score = max(-coverage_gap, 0) * unit_cost * class_weight

Within budget cap, SKUs are selected by descending risk_score.
在预算上限内，按 risk_score 从高到低进行优先补货。

## 5. Causality Boundary / 因果边界说明

This app provides evidence-based diagnostic attribution, not strict causal proof.
本系统提供基于证据的诊断归因，不等同于严格因果证明。

For causal proof, controlled experiments, intervention logs, and quasi-experimental designs are needed.
若需因果证明，需结合受控实验、干预日志和准实验方法。
