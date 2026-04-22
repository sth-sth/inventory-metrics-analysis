from __future__ import annotations

import pandas as pd


def _num_or_none(metric_row: pd.Series, col: str) -> float | None:
    if col not in metric_row.index:
        return None
    val = pd.to_numeric(pd.Series([metric_row[col]]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def build_issue_breakdown(metric_row: pd.Series, language: str, stockout_threshold: float, overstock_doh: float, delay_days: float) -> pd.DataFrame:
    """Build issue rows from raw business fields only.

    No synthetic score and no fallback defaults. Missing fields are clearly labeled
    as not computable so users can trust what is shown.
    """

    sku = str(metric_row.get("sku", ""))
    warehouse = str(metric_row.get("warehouse", ""))

    avg_daily_demand = _num_or_none(metric_row, "avg_daily_demand")
    forecast_daily_demand = _num_or_none(metric_row, "forecast_daily_demand")
    avg_delay_days = _num_or_none(metric_row, "avg_delay_days")
    on_hand_qty = _num_or_none(metric_row, "on_hand_qty")
    reorder_point = _num_or_none(metric_row, "reorder_point")
    coverage_gap = _num_or_none(metric_row, "coverage_gap")
    doh = _num_or_none(metric_row, "doh")
    data_lag_days = _num_or_none(metric_row, "data_lag_days")
    cycle_count_accuracy = _num_or_none(metric_row, "cycle_count_accuracy")

    cn_rows: list[dict[str, object]] = []
    en_rows: list[dict[str, object]] = []

    if avg_daily_demand is None or forecast_daily_demand is None:
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "需求预测偏差",
                "来自哪里": "inventory.csv + transactions.csv",
                "当前值": "缺少 avg_daily_demand 或 forecast_daily_demand",
                "阈值/基准": "无",
                "差了多少": "无法计算",
                "是否超阈值": "字段缺失",
                "怎么算": "需要同时有实际日均需求和预测日均需求字段。",
                "建议看什么": "先补齐预测字段，再判断偏差。",
                "偏差绝对值": None,
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Demand forecast deviation",
                "source": "inventory.csv + transactions.csv",
                "current": "missing avg_daily_demand or forecast_daily_demand",
                "baseline": "none",
                "gap": "not computable",
                "out_of_bound": "missing field",
                "how": "Both actual and forecast daily demand fields are required.",
                "what_to_check": "Fill forecast field first, then evaluate deviation.",
                "absolute_gap": None,
            }
        )
    else:
        demand_gap = avg_daily_demand - forecast_daily_demand
        demand_pct = abs(demand_gap) / max(abs(forecast_daily_demand), 1e-9)
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "需求预测偏差",
                "来自哪里": "inventory.csv + transactions.csv",
                "当前值": f"实际 {avg_daily_demand:.2f}，预测 {forecast_daily_demand:.2f}",
                "阈值/基准": "无固定阈值，按业务判断",
                "差了多少": f"差值 {demand_gap:+.2f}，相对误差 {demand_pct:.1%}",
                "是否超阈值": "人工判断",
                "怎么算": "实际日均需求 - 预测日均需求。",
                "建议看什么": "偏差大时核对促销、季节、异常订单。",
                "偏差绝对值": abs(demand_gap),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Demand forecast deviation",
                "source": "inventory.csv + transactions.csv",
                "current": f"actual {avg_daily_demand:.2f}, forecast {forecast_daily_demand:.2f}",
                "baseline": "no fixed threshold; business judgement",
                "gap": f"delta {demand_gap:+.2f}, relative error {demand_pct:.1%}",
                "out_of_bound": "manual review",
                "how": "actual daily demand - forecast daily demand",
                "what_to_check": "Validate promotion, seasonality, abnormal orders when deviation is large.",
                "absolute_gap": abs(demand_gap),
            }
        )

    if avg_delay_days is None:
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "收货延迟",
                "来自哪里": "transactions.csv",
                "当前值": "缺少收货 delay_days",
                "阈值/基准": f"延迟阈值 {delay_days:.2f} 天",
                "差了多少": "无法计算",
                "是否超阈值": "字段缺失",
                "怎么算": "需要 receipt 事件的 delay_days。",
                "建议看什么": "补齐收货延迟字段后再判断。",
                "偏差绝对值": None,
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Receipt delay",
                "source": "transactions.csv",
                "current": "missing receipt delay_days",
                "baseline": f"delay threshold {delay_days:.2f} days",
                "gap": "not computable",
                "out_of_bound": "missing field",
                "how": "requires delay_days on receipt events",
                "what_to_check": "Fill receipt delay fields first.",
                "absolute_gap": None,
            }
        )
    else:
        delay_gap = avg_delay_days - delay_days
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "收货延迟",
                "来自哪里": "transactions.csv",
                "当前值": f"平均延迟 {avg_delay_days:.2f} 天",
                "阈值/基准": f"延迟阈值 {delay_days:.2f} 天",
                "差了多少": f"{delay_gap:+.2f} 天",
                "是否超阈值": "是" if delay_gap > 0 else "否",
                "怎么算": "收货事件 delay_days 的平均值减去阈值。",
                "建议看什么": "超阈值优先检查供应商与运输。",
                "偏差绝对值": abs(delay_gap),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Receipt delay",
                "source": "transactions.csv",
                "current": f"average delay {avg_delay_days:.2f} days",
                "baseline": f"delay threshold {delay_days:.2f} days",
                "gap": f"{delay_gap:+.2f} days",
                "out_of_bound": "yes" if delay_gap > 0 else "no",
                "how": "mean(receipt delay_days) - threshold",
                "what_to_check": "Check suppliers and transport when above threshold.",
                "absolute_gap": abs(delay_gap),
            }
        )

    if on_hand_qty is not None and reorder_point is not None and coverage_gap is not None:
        stockout_delta = stockout_threshold - coverage_gap
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "库存缺口",
                "来自哪里": "inventory.csv",
                "当前值": f"在手 {on_hand_qty:.2f}，ROP {reorder_point:.2f}，Gap {coverage_gap:+.2f}",
                "阈值/基准": f"缺货阈值 {stockout_threshold:.2f}",
                "差了多少": f"离阈值 {stockout_delta:+.2f}",
                "是否超阈值": "是" if coverage_gap < stockout_threshold else "否",
                "怎么算": "coverage_gap 与缺货阈值直接比较。",
                "建议看什么": "低于阈值时优先人工确认补货。",
                "偏差绝对值": abs(stockout_delta),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Inventory gap",
                "source": "inventory.csv",
                "current": f"on-hand {on_hand_qty:.2f}, ROP {reorder_point:.2f}, gap {coverage_gap:+.2f}",
                "baseline": f"stockout threshold {stockout_threshold:.2f}",
                "gap": f"distance to threshold {stockout_delta:+.2f}",
                "out_of_bound": "yes" if coverage_gap < stockout_threshold else "no",
                "how": "directly compare coverage_gap with stockout threshold",
                "what_to_check": "Manually validate replenishment when below threshold.",
                "absolute_gap": abs(stockout_delta),
            }
        )

    if doh is not None:
        overstock_gap = doh - overstock_doh
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "库存覆盖过高",
                "来自哪里": "inventory.csv",
                "当前值": f"DOH {doh:.2f} 天",
                "阈值/基准": f"超储阈值 {overstock_doh:.2f} 天",
                "差了多少": f"{overstock_gap:+.2f} 天",
                "是否超阈值": "是" if overstock_gap > 0 else "否",
                "怎么算": "DOH 减去超储阈值。",
                "建议看什么": "超阈值时考虑减采、促销或清理。",
                "偏差绝对值": abs(overstock_gap),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "High inventory coverage",
                "source": "inventory.csv",
                "current": f"DOH {doh:.2f} days",
                "baseline": f"overstock threshold {overstock_doh:.2f} days",
                "gap": f"{overstock_gap:+.2f} days",
                "out_of_bound": "yes" if overstock_gap > 0 else "no",
                "how": "DOH - overstock threshold",
                "what_to_check": "Consider slower buying, promotions, or clearance when above threshold.",
                "absolute_gap": abs(overstock_gap),
            }
        )

    if data_lag_days is not None:
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "信息滞后",
                "来自哪里": "transactions.csv",
                "当前值": f"滞后 {data_lag_days:.2f} 天",
                "阈值/基准": "0 天",
                "差了多少": f"{data_lag_days:+.2f} 天",
                "是否超阈值": "是" if data_lag_days > 0 else "否",
                "怎么算": "库存快照日期 - 最近交易日期。",
                "建议看什么": "滞后越大越要先查数据同步。",
                "偏差绝对值": abs(data_lag_days),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Data lag",
                "source": "transactions.csv",
                "current": f"lag {data_lag_days:.2f} days",
                "baseline": "0 day",
                "gap": f"{data_lag_days:+.2f} days",
                "out_of_bound": "yes" if data_lag_days > 0 else "no",
                "how": "snapshot date - latest transaction date",
                "what_to_check": "Investigate data sync first when lag is high.",
                "absolute_gap": abs(data_lag_days),
            }
        )

    if cycle_count_accuracy is not None:
        acc_gap = 1 - cycle_count_accuracy
        cn_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "问题": "盘点准确率",
                "来自哪里": "transactions.csv",
                "当前值": f"{cycle_count_accuracy:.2%}",
                "阈值/基准": "100%",
                "差了多少": f"{acc_gap:.2%}",
                "是否超阈值": "是" if acc_gap > 0 else "否",
                "怎么算": "1 - cycle_count_accuracy。",
                "建议看什么": "偏离越大越需要检查盘点流程。",
                "偏差绝对值": abs(acc_gap),
            }
        )
        en_rows.append(
            {
                "sku": sku,
                "warehouse": warehouse,
                "issue": "Cycle count accuracy",
                "source": "transactions.csv",
                "current": f"{cycle_count_accuracy:.2%}",
                "baseline": "100%",
                "gap": f"{acc_gap:.2%}",
                "out_of_bound": "yes" if acc_gap > 0 else "no",
                "how": "1 - cycle_count_accuracy",
                "what_to_check": "Larger gap requires cycle-count process review.",
                "absolute_gap": abs(acc_gap),
            }
        )

    out = pd.DataFrame(cn_rows if language == "中文" else en_rows)
    if language == "中文":
        domain_map = {
            "需求预测偏差": "需求侧",
            "收货延迟": "供应侧",
            "库存缺口": "仓储侧",
            "库存覆盖过高": "仓储侧",
            "信息滞后": "流程侧",
            "盘点准确率": "流程侧",
        }
        if "问题" in out.columns:
            out["分类"] = out["问题"].map(domain_map).fillna("其他")
    else:
        domain_map = {
            "Demand forecast deviation": "Demand",
            "Receipt delay": "Supply",
            "Inventory gap": "Warehouse",
            "High inventory coverage": "Warehouse",
            "Data lag": "Process",
            "Cycle count accuracy": "Process",
        }
        if "issue" in out.columns:
            out["domain"] = out["issue"].map(domain_map).fillna("Other")
    return out
