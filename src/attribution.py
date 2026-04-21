from __future__ import annotations

import numpy as np
import pandas as pd


def root_cause_attribution(metrics_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    risk = metrics_df[metrics_df["inventory_health"] == "Stockout Risk"][["sku", "warehouse", "coverage_gap"]].copy()
    if risk.empty:
        return pd.DataFrame(columns=["sku", "warehouse", "factor", "evidence", "impact_score"])

    tx = transactions.copy()
    tx["is_sale"] = (tx["event_type"] == "sale").astype(int)
    tx["is_receipt"] = (tx["event_type"] == "receipt").astype(int)

    sales = tx[tx["event_type"] == "sale"].groupby(["sku", "warehouse"], as_index=False)["qty"].sum().rename(columns={"qty": "sales_qty"})
    receipts = tx[tx["event_type"] == "receipt"].groupby(["sku", "warehouse"], as_index=False)["qty"].sum().rename(columns={"qty": "receipt_qty"})
    delays = tx[tx["event_type"] == "receipt"].groupby(["sku", "warehouse"], as_index=False)["delay_days"].mean().rename(columns={"delay_days": "avg_delay_days"})

    merged = risk.merge(sales, on=["sku", "warehouse"], how="left").merge(receipts, on=["sku", "warehouse"], how="left").merge(delays, on=["sku", "warehouse"], how="left")
    merged[["sales_qty", "receipt_qty", "avg_delay_days"]] = merged[["sales_qty", "receipt_qty", "avg_delay_days"]].fillna(0)

    merged["sales_minus_receipt"] = merged["sales_qty"] - merged["receipt_qty"]
    bench_sales_gap = float(merged["sales_minus_receipt"].mean()) if not merged.empty else 0.0
    bench_delay = float(merged["avg_delay_days"].mean()) if not merged.empty else 0.0
    bench_gap = float(abs(merged["coverage_gap"]).mean()) if not merged.empty else 1.0

    rows = []
    for _, r in merged.iterrows():
        sales_pressure = max(r["sales_qty"] - r["receipt_qty"], 0)
        delay_pressure = max(r["avg_delay_days"], 0)
        gap_pressure = abs(min(r["coverage_gap"], 0))

        demand_signal = max(sales_pressure - bench_sales_gap, 0)
        delay_signal = max(delay_pressure - bench_delay, 0)
        planning_signal = max(gap_pressure - bench_gap, 0)

        sample_days = transactions[
            (transactions["sku"] == r["sku"]) & (transactions["warehouse"] == r["warehouse"])
        ]["date"].nunique()
        confidence = min(1.0, sample_days / 7.0)

        factors = [
            (
                "Demand Surge",
                demand_signal,
                f"sales-receipt={sales_pressure:.1f}, benchmark={bench_sales_gap:.1f}",
            ),
            (
                "Supplier Delay",
                delay_signal,
                f"avg_delay_days={r['avg_delay_days']:.2f}, benchmark={bench_delay:.2f}",
            ),
            (
                "Planning Gap",
                planning_signal,
                f"abs_gap={gap_pressure:.2f}, benchmark={bench_gap:.2f}",
            ),
        ]

        total = sum(score for _, score, _ in factors) or 1.0
        for factor, score, evidence in factors:
            rows.append(
                {
                    "sku": r["sku"],
                    "warehouse": r["warehouse"],
                    "factor": factor,
                    "evidence": evidence,
                    "impact_score": float(score / total),
                    "confidence": float(np.clip(confidence, 0, 1)),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["sku", "warehouse", "impact_score"], ascending=[True, True, False])
