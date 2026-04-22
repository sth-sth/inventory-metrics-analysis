from __future__ import annotations

import pandas as pd


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
