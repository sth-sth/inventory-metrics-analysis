from __future__ import annotations

import numpy as np
import pandas as pd


def build_inventory_metrics(inventory: pd.DataFrame) -> pd.DataFrame:
    df = inventory.copy()
    df["doh"] = np.where(df["avg_daily_demand"] > 0, df["on_hand_qty"] / df["avg_daily_demand"], np.nan)
    df["inventory_value"] = df["on_hand_qty"] * df["unit_cost"]
    df["lead_time_demand"] = df["avg_daily_demand"] * df["lead_time_days"]

    demand_std = df["avg_daily_demand"].std(ddof=0) if len(df) > 1 else 0.0
    z_score = 1.65
    df["safety_stock"] = z_score * demand_std * np.sqrt(df["lead_time_days"].clip(lower=0))
    df["reorder_point"] = df["lead_time_demand"] + df["safety_stock"]
    df["coverage_gap"] = df["on_hand_qty"] - df["reorder_point"]

    df["inventory_health"] = np.select(
        [
            df["coverage_gap"] < 0,
            (df["doh"] > 120),
            (df["doh"].between(20, 90, inclusive="both")),
        ],
        ["Stockout Risk", "Overstock Risk", "Healthy"],
        default="Watch",
    )
    return df


def build_kpi_summary(metrics_df: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    total_value = float(metrics_df["inventory_value"].sum())
    stockout_risk_items = int((metrics_df["inventory_health"] == "Stockout Risk").sum())
    overstock_risk_items = int((metrics_df["inventory_health"] == "Overstock Risk").sum())

    sales = transactions.loc[transactions["event_type"] == "sale", "qty"].sum()
    avg_inventory_qty = metrics_df["on_hand_qty"].mean()
    inv_turnover = float(sales / avg_inventory_qty) if avg_inventory_qty > 0 else np.nan

    service_level_proxy = float((metrics_df["coverage_gap"] >= 0).mean())

    return {
        "total_inventory_value": total_value,
        "stockout_risk_sku_count": stockout_risk_items,
        "overstock_risk_sku_count": overstock_risk_items,
        "inventory_turnover_proxy": inv_turnover,
        "service_level_proxy": service_level_proxy,
    }


def abc_classification(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df = df.sort_values("inventory_value", ascending=False)
    total = df["inventory_value"].sum()
    if total <= 0:
        df["abc_class"] = "C"
        return df

    df["value_share"] = df["inventory_value"] / total
    df["cum_value_share"] = df["value_share"].cumsum()
    df["abc_class"] = np.select(
        [df["cum_value_share"] <= 0.8, df["cum_value_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    return df
