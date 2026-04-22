from __future__ import annotations

import numpy as np
import pandas as pd


def build_inventory_metrics(
    inventory: pd.DataFrame,
    stockout_gap_threshold: float = 0.0,
    overstock_doh_threshold: float = 120.0,
) -> pd.DataFrame:
    df = inventory.copy()
    df["doh"] = np.where(df["avg_daily_demand"] > 0, df["on_hand_qty"] / df["avg_daily_demand"], np.nan)
    df["inventory_value"] = df["on_hand_qty"] * df["unit_cost"]
    df["lead_time_demand"] = df["avg_daily_demand"] * df["lead_time_days"]

    demand_std = df["avg_daily_demand"].std(ddof=0) if len(df) > 1 else 0.0
    z_score = 1.65
    df["safety_stock"] = z_score * demand_std * np.sqrt(df["lead_time_days"].clip(lower=0))
    df["reorder_point"] = df["lead_time_demand"] + df["safety_stock"]
    df["coverage_gap"] = df["on_hand_qty"] - df["reorder_point"]

    stockout_mask = df["coverage_gap"] < stockout_gap_threshold
    overstock_mask = (df["doh"] > overstock_doh_threshold) & ~stockout_mask

    df["inventory_health"] = np.select(
        [
            stockout_mask,
            overstock_mask,
            (df["doh"].between(20, 90, inclusive="both")) & ~stockout_mask & ~overstock_mask,
        ],
        ["Stockout Risk", "Overstock Risk", "Healthy"],
        default="Watch",
    )
    return df


def build_kpi_summary(
    metrics_df: pd.DataFrame,
    transactions: pd.DataFrame,
    slow_moving_doh_threshold: float = 180.0,
    carrying_cost_rate: float = 0.20,
) -> dict:
    total_value = float(metrics_df["inventory_value"].sum())
    stockout_risk_items = int((metrics_df["inventory_health"] == "Stockout Risk").sum())
    overstock_risk_items = int((metrics_df["inventory_health"] == "Overstock Risk").sum())

    # In-stock rate: SKUs at or above reorder coverage baseline.
    in_stock_rate = float((metrics_df["coverage_gap"] >= 0).mean()) if len(metrics_df) else np.nan

    sales_tx = transactions[transactions["event_type"] == "sale"].copy()
    cost_ref = metrics_df[["sku", "warehouse", "unit_cost"]].drop_duplicates()
    sales_tx = sales_tx.merge(cost_ref, on=["sku", "warehouse"], how="left")
    sales_tx["unit_cost"] = sales_tx["unit_cost"].fillna(0.0)
    sales_tx["cogs"] = sales_tx["qty"] * sales_tx["unit_cost"]
    cogs_value = float(sales_tx["cogs"].sum())

    avg_inventory_value = float(metrics_df["inventory_value"].mean()) if len(metrics_df) else np.nan
    inventory_turnover = float(cogs_value / avg_inventory_value) if avg_inventory_value and avg_inventory_value > 0 else np.nan
    turnover_days = float(365.0 / inventory_turnover) if inventory_turnover and inventory_turnover > 0 else np.nan

    sales_by_sku = sales_tx.groupby(["sku", "warehouse"], as_index=False)["qty"].sum().rename(columns={"qty": "sales_qty"})
    slow_df = metrics_df.merge(sales_by_sku, on=["sku", "warehouse"], how="left")
    slow_df["sales_qty"] = slow_df["sales_qty"].fillna(0.0)
    slow_moving_rate = float(((slow_df["sales_qty"] <= 0) | (slow_df["doh"] >= slow_moving_doh_threshold)).mean()) if len(slow_df) else np.nan

    inventory_cost = float(total_value * carrying_cost_rate)

    return {
        "total_inventory_value": total_value,
        "stockout_risk_sku_count": stockout_risk_items,
        "overstock_risk_sku_count": overstock_risk_items,
        "in_stock_rate": in_stock_rate,
        "inventory_turnover": inventory_turnover,
        "turnover_days": turnover_days,
        "slow_moving_rate": slow_moving_rate,
        "inventory_cost": inventory_cost,
        "cogs_value": cogs_value,
        "avg_inventory_value": avg_inventory_value,
        "carrying_cost_rate": carrying_cost_rate,
        "slow_moving_doh_threshold": slow_moving_doh_threshold,
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
