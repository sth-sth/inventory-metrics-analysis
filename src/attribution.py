from __future__ import annotations

import pandas as pd


def _factor_map(language: str) -> dict[str, str]:
    en_map = {
        "需求侧_预测不准": "demand_forecast_error",
        "需求侧_历史不足": "demand_insufficient_history",
        "需求侧_季节波动": "demand_seasonality",
        "需求侧_促销冲击": "demand_promotion_shock",
        "需求侧_突变风险": "demand_market_shift",
        "供应侧_供应商延迟": "supply_supplier_delay",
        "供应侧_交付不稳": "supply_delivery_instability",
        "供应侧_MOQ约束": "supply_moq_pressure",
        "供应侧_运输阻塞": "supply_transport_block",
        "仓储侧_安全库存设置": "warehouse_safety_stock_setting",
        "仓储侧_ROP设置": "warehouse_rop_setting",
        "仓储侧_阈值静态": "warehouse_static_threshold",
        "仓储侧_交期假设": "warehouse_leadtime_assumption",
        "流程侧_盘点规范": "process_cycle_counting",
        "流程侧_信息同步": "process_info_sync",
        "流程侧_审批速度": "process_approval_speed",
        "流程侧_数据滞后": "process_data_latency",
    }
    if language == "English":
        return en_map
    return {v: k for k, v in en_map.items()}


def _factor_meta(language: str) -> dict[str, dict[str, str]]:
    if language == "中文":
        return {
            "需求侧_预测不准": {
                "domain": "需求",
                "raw": "avg_daily_demand, forecast_daily_demand",
                "formula": "abs(avg_daily_demand - forecast_daily_demand) / forecast_daily_demand, then clip(score / 0.5, 0, 1)",
                "note": "预测偏差越大，需求侧风险越高。",
            },
            "需求侧_历史不足": {
                "domain": "需求",
                "raw": "history_days",
                "formula": "history_days < 7 => 1.0; < 14 => 0.6; otherwise 0.2",
                "note": "历史越短，判断越不稳定。",
            },
            "需求侧_季节波动": {
                "domain": "需求",
                "raw": "seasonality_index",
                "formula": "clip(abs(seasonality_index - 1.0) / 0.8, 0, 1)",
                "note": "季节性越强，波动越大。",
            },
            "需求侧_促销冲击": {
                "domain": "需求",
                "raw": "promo_lift_index",
                "formula": "clip((promo_lift_index - 1.0) / 1.0, 0, 1)",
                "note": "促销越强，短期需求越容易突增。",
            },
            "需求侧_突变风险": {
                "domain": "需求",
                "raw": "market_shock_index",
                "formula": "clip((market_shock_index - 1.0) / 0.7, 0, 1)",
                "note": "外部冲击越大，需求越不稳定。",
            },
            "供应侧_供应商延迟": {
                "domain": "供应",
                "raw": "avg_delay_days",
                "formula": "clip(avg_delay_days / 7.0, 0, 1)",
                "note": "到货延迟越长，供应风险越高。",
            },
            "供应侧_交付不稳": {
                "domain": "供应",
                "raw": "delay_std",
                "formula": "clip(delay_std / 3.5, 0, 1)",
                "note": "延迟波动越大，交付越不稳。",
            },
            "供应侧_MOQ约束": {
                "domain": "供应",
                "raw": "moq_qty, reorder_need",
                "formula": "clip(max(moq_qty - reorder_need, 0) / moq_qty, 0, 1)",
                "note": "MOQ 越高，补货越受限制。",
            },
            "供应侧_运输阻塞": {
                "domain": "供应",
                "raw": "in_transit_blocked_flag",
                "formula": "clip(in_transit_blocked_flag, 0, 1)",
                "note": "在途阻塞直接增加供应风险。",
            },
            "仓储侧_安全库存设置": {
                "domain": "仓储",
                "raw": "safety_stock, lead_time_demand",
                "formula": "clip(abs(safety_stock / lead_time_demand - 0.25) / 0.25, 0, 1)",
                "note": "安全库存设置偏离基准越多，风险越高。",
            },
            "仓储侧_ROP设置": {
                "domain": "仓储",
                "raw": "coverage_gap, reorder_point",
                "formula": "clip(abs(coverage_gap) / reorder_point, 0, 1)",
                "note": "ROP 越偏离实际库存，风险越高。",
            },
            "仓储侧_阈值静态": {
                "domain": "仓储",
                "raw": "avg_daily_demand volatility",
                "formula": "clip((demand_cv - 0.25) / 0.6, 0, 1)",
                "note": "需求越波动，静态阈值越不适用。",
            },
            "仓储侧_交期假设": {
                "domain": "仓储",
                "raw": "avg_delay_days, lead_time_days",
                "formula": "clip(abs(avg_delay_days - lead_time_days) / lead_time_days, 0, 1)",
                "note": "交期假设偏差越大，仓储计划越容易失真。",
            },
            "流程侧_盘点规范": {
                "domain": "流程",
                "raw": "cycle_count_accuracy",
                "formula": "clip(1 - cycle_count_accuracy, 0, 1)",
                "note": "盘点准确率越低，流程风险越高。",
            },
            "流程侧_信息同步": {
                "domain": "流程",
                "raw": "info_sync_delay_days",
                "formula": "clip(info_sync_delay_days / 5.0, 0, 1)",
                "note": "信息同步越慢，结果越滞后。",
            },
            "流程侧_审批速度": {
                "domain": "流程",
                "raw": "approval_delay_days",
                "formula": "clip(approval_delay_days / 7.0, 0, 1)",
                "note": "审批越慢，补货闭环越慢。",
            },
            "流程侧_数据滞后": {
                "domain": "流程",
                "raw": "data_lag_days",
                "formula": "clip(data_lag_days / 5.0, 0, 1)",
                "note": "数据滞后越大，结论越晚到。",
            },
        }

    return {
        "demand_forecast_error": {
            "domain": "demand",
            "raw": "avg_daily_demand, forecast_daily_demand",
            "formula": "abs(avg_daily_demand - forecast_daily_demand) / forecast_daily_demand, then clip(score / 0.5, 0, 1)",
            "note": "Higher forecast deviation means higher demand risk.",
        },
        "demand_insufficient_history": {
            "domain": "demand",
            "raw": "history_days",
            "formula": "history_days < 7 => 1.0; < 14 => 0.6; otherwise 0.2",
            "note": "Shorter history means less stable judgment.",
        },
        "demand_seasonality": {
            "domain": "demand",
            "raw": "seasonality_index",
            "formula": "clip(abs(seasonality_index - 1.0) / 0.8, 0, 1)",
            "note": "More seasonality means higher volatility.",
        },
        "demand_promotion_shock": {
            "domain": "demand",
            "raw": "promo_lift_index",
            "formula": "clip((promo_lift_index - 1.0) / 1.0, 0, 1)",
            "note": "Stronger promotions can spike short-term demand.",
        },
        "demand_market_shift": {
            "domain": "demand",
            "raw": "market_shock_index",
            "formula": "clip((market_shock_index - 1.0) / 0.7, 0, 1)",
            "note": "External shocks increase instability.",
        },
        "supply_supplier_delay": {
            "domain": "supply",
            "raw": "avg_delay_days",
            "formula": "clip(avg_delay_days / 7.0, 0, 1)",
            "note": "Longer receipt delay means higher supply risk.",
        },
        "supply_delivery_instability": {
            "domain": "supply",
            "raw": "delay_std",
            "formula": "clip(delay_std / 3.5, 0, 1)",
            "note": "More delay variance means less stable delivery.",
        },
        "supply_moq_pressure": {
            "domain": "supply",
            "raw": "moq_qty, reorder_need",
            "formula": "clip(max(moq_qty - reorder_need, 0) / moq_qty, 0, 1)",
            "note": "Higher MOQ constrains replenishment.",
        },
        "supply_transport_block": {
            "domain": "supply",
            "raw": "in_transit_blocked_flag",
            "formula": "clip(in_transit_blocked_flag, 0, 1)",
            "note": "Blocked transit directly increases supply risk.",
        },
        "warehouse_safety_stock_setting": {
            "domain": "warehouse",
            "raw": "safety_stock, lead_time_demand",
            "formula": "clip(abs(safety_stock / lead_time_demand - 0.25) / 0.25, 0, 1)",
            "note": "Safety stock far from baseline raises risk.",
        },
        "warehouse_rop_setting": {
            "domain": "warehouse",
            "raw": "coverage_gap, reorder_point",
            "formula": "clip(abs(coverage_gap) / reorder_point, 0, 1)",
            "note": "ROP mismatch makes planning less reliable.",
        },
        "warehouse_static_threshold": {
            "domain": "warehouse",
            "raw": "avg_daily_demand volatility",
            "formula": "clip((demand_cv - 0.25) / 0.6, 0, 1)",
            "note": "More demand volatility makes static thresholds weaker.",
        },
        "warehouse_leadtime_assumption": {
            "domain": "warehouse",
            "raw": "avg_delay_days, lead_time_days",
            "formula": "clip(abs(avg_delay_days - lead_time_days) / lead_time_days, 0, 1)",
            "note": "Lead-time mismatch distorts warehouse planning.",
        },
        "process_cycle_counting": {
            "domain": "process",
            "raw": "cycle_count_accuracy",
            "formula": "clip(1 - cycle_count_accuracy, 0, 1)",
            "note": "Lower cycle count accuracy means higher process risk.",
        },
        "process_info_sync": {
            "domain": "process",
            "raw": "info_sync_delay_days",
            "formula": "clip(info_sync_delay_days / 5.0, 0, 1)",
            "note": "Slower info sync delays the result.",
        },
        "process_approval_speed": {
            "domain": "process",
            "raw": "approval_delay_days",
            "formula": "clip(approval_delay_days / 7.0, 0, 1)",
            "note": "Slower approvals slow the replenishment loop.",
        },
        "process_data_latency": {
            "domain": "process",
            "raw": "data_lag_days",
            "formula": "clip(data_lag_days / 5.0, 0, 1)",
            "note": "More data lag means later conclusions.",
        },
    }


def build_factor_detail_table(granular_row: pd.Series, language: str = "中文") -> pd.DataFrame:
    meta = _factor_meta(language)
    factor_map = _factor_map(language)
    factor_cols = [c for c in granular_row.index if c in meta or c in factor_map]
    rows = []
    factor_values = [float(granular_row[c]) for c in factor_cols]
    total = sum(factor_values) or 1.0
    overall_score = float(granular_row.get("overall_score", 0.0))
    overall_n = max(len(factor_cols), 1)
    for factor_col in factor_cols:
        score = float(granular_row[factor_col])
        score_name = factor_map.get(factor_col, factor_col)
        info = meta.get(factor_col if language == "中文" else score_name, {})
        rows.append(
            {
                "factor": factor_col if language == "中文" else score_name,
                "domain": info.get("domain", ""),
                "raw": info.get("raw", ""),
                "formula": info.get("formula", ""),
                "score": score,
                "impact_points": score / overall_n,
                "impact_pct": score / total,
                "note": info.get("note", ""),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out["overall_score"] = overall_score
    return out


def build_domain_detail_table(granular_row: pd.Series, language: str = "中文") -> pd.DataFrame:
    domain_cols = ["demand_domain_score", "supply_domain_score", "warehouse_domain_score", "process_domain_score"]
    labels = {
        "demand_domain_score": "需求" if language == "中文" else "demand",
        "supply_domain_score": "供应" if language == "中文" else "supply",
        "warehouse_domain_score": "仓储" if language == "中文" else "warehouse",
        "process_domain_score": "流程" if language == "中文" else "process",
    }
    rows = []
    for col in domain_cols:
        score = float(granular_row[col])
        rows.append(
            {
                "domain": labels[col],
                "score": score,
                "impact_points": score / 4.0,
                "impact_pct": score / max(float(granular_row.get("overall_score", 0.0)) * 4.0, 1e-9),
                "formula": f"mean({labels[col]} factors)",
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


FACTOR_MAP_EN = {
    "需求侧_预测不准": "demand_forecast_error",
    "需求侧_历史不足": "demand_insufficient_history",
    "需求侧_季节波动": "demand_seasonality",
    "需求侧_促销冲击": "demand_promotion_shock",
    "需求侧_突变风险": "demand_market_shift",
    "供应侧_供应商延迟": "supply_supplier_delay",
    "供应侧_交付不稳": "supply_delivery_instability",
    "供应侧_MOQ约束": "supply_moq_pressure",
    "供应侧_运输阻塞": "supply_transport_block",
    "仓储侧_安全库存设置": "warehouse_safety_stock_setting",
    "仓储侧_ROP设置": "warehouse_rop_setting",
    "仓储侧_阈值静态": "warehouse_static_threshold",
    "仓储侧_交期假设": "warehouse_leadtime_assumption",
    "流程侧_盘点规范": "process_cycle_counting",
    "流程侧_信息同步": "process_info_sync",
    "流程侧_审批速度": "process_approval_speed",
    "流程侧_数据滞后": "process_data_latency",
}


def _factor_columns(df: pd.DataFrame) -> list[str]:
    """Extract granular factor column names from dataframe."""
    return [
        c
        for c in df.columns
        if c.startswith("需求侧_") or c.startswith("供应侧_") or c.startswith("仓储侧_") or c.startswith("流程侧_")
    ]


def root_cause_attribution(granular_df: pd.DataFrame, language: str = "中文") -> pd.DataFrame:
    """Build factor-level attribution from granular factor scores."""
    factor_cols = _factor_columns(granular_df)
    if granular_df.empty or not factor_cols:
        return pd.DataFrame(columns=["sku", "warehouse", "factor", "impact_score"])

    rows = []
    for _, r in granular_df.iterrows():
        total = float(sum(float(r[c]) for c in factor_cols)) or 1.0
        for c in factor_cols:
            factor_name = FACTOR_MAP_EN.get(c, c) if language == "English" else c
            rows.append(
                {
                    "sku": r["sku"],
                    "warehouse": r["warehouse"],
                    "factor": factor_name,
                    "impact_score": float(r[c]) / total,
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["sku", "warehouse", "impact_score"], ascending=[True, True, False])


def attribution_step_breakdown(granular_df: pd.DataFrame, language: str = "中文") -> pd.DataFrame:
    """Domain-level attribution: steps are demand/supply/warehouse/process domains."""
    required = ["sku", "warehouse", "demand_domain_score", "supply_domain_score", "warehouse_domain_score", "process_domain_score"]
    if granular_df.empty or not set(required).issubset(granular_df.columns):
        return pd.DataFrame(columns=["sku", "warehouse", "step", "score"])

    rows = []
    for _, r in granular_df[required].iterrows():
        labels = {
            "demand_domain_score": "需求域" if language == "中文" else "demand",
            "supply_domain_score": "供应域" if language == "中文" else "supply",
            "warehouse_domain_score": "仓储域" if language == "中文" else "warehouse",
            "process_domain_score": "流程域" if language == "中文" else "process",
        }
        for col, step_label in labels.items():
            rows.append({"sku": r["sku"], "warehouse": r["warehouse"], "step": step_label, "score": float(r[col])})

    return pd.DataFrame(rows).sort_values(["sku", "warehouse", "score"], ascending=[True, True, False])
