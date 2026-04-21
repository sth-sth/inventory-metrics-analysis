from __future__ import annotations

import numpy as np
import pandas as pd


def _build_base(metrics_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    risk = metrics_df[metrics_df["inventory_health"] == "Stockout Risk"][["sku", "warehouse", "coverage_gap"]].copy()
    if risk.empty:
        return pd.DataFrame(
            columns=[
                "sku",
                "warehouse",
                "coverage_gap",
                "sales_qty",
                "receipt_qty",
                "avg_delay_days",
                "sales_minus_receipt",
                "bench_sales_gap",
                "bench_delay",
                "bench_gap",
                "demand_signal",
                "delay_signal",
                "planning_signal",
                "confidence",
            ]
        )

    tx = transactions.copy()

    sales = (
        tx[tx["event_type"] == "sale"]
        .groupby(["sku", "warehouse"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "sales_qty"})
    )
    receipts = (
        tx[tx["event_type"] == "receipt"]
        .groupby(["sku", "warehouse"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "receipt_qty"})
    )
    delays = (
        tx[tx["event_type"] == "receipt"]
        .groupby(["sku", "warehouse"], as_index=False)["delay_days"]
        .mean()
        .rename(columns={"delay_days": "avg_delay_days"})
    )

    merged = (
        risk.merge(sales, on=["sku", "warehouse"], how="left")
        .merge(receipts, on=["sku", "warehouse"], how="left")
        .merge(delays, on=["sku", "warehouse"], how="left")
    )
    merged[["sales_qty", "receipt_qty", "avg_delay_days"]] = merged[["sales_qty", "receipt_qty", "avg_delay_days"]].fillna(0)

    merged["sales_minus_receipt"] = merged["sales_qty"] - merged["receipt_qty"]
    bench_sales_gap = float(merged["sales_minus_receipt"].mean()) if not merged.empty else 0.0
    bench_delay = float(merged["avg_delay_days"].mean()) if not merged.empty else 0.0
    bench_gap = float(abs(merged["coverage_gap"]).mean()) if not merged.empty else 1.0

    merged["bench_sales_gap"] = bench_sales_gap
    merged["bench_delay"] = bench_delay
    merged["bench_gap"] = bench_gap
    merged["demand_signal"] = np.maximum(merged["sales_minus_receipt"] - bench_sales_gap, 0)
    merged["delay_signal"] = np.maximum(merged["avg_delay_days"] - bench_delay, 0)
    merged["planning_signal"] = np.maximum(abs(merged["coverage_gap"]) - bench_gap, 0)

    days_covered = (
        tx.groupby(["sku", "warehouse"], as_index=False)["date"]
        .nunique()
        .rename(columns={"date": "sample_days"})
    )
    merged = merged.merge(days_covered, on=["sku", "warehouse"], how="left")
    merged["sample_days"] = merged["sample_days"].fillna(0)
    merged["confidence"] = np.clip(merged["sample_days"] / 7.0, 0, 1)

    return merged


def root_cause_attribution(metrics_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    merged = _build_base(metrics_df, transactions)
    if merged.empty:
        return pd.DataFrame(columns=["sku", "warehouse", "factor", "evidence", "impact_score", "confidence"])

    rows = []
    for _, r in merged.iterrows():
        factors = [
            (
                "Demand Surge",
                r["demand_signal"],
                f"sales-receipt={r['sales_minus_receipt']:.1f}, benchmark={r['bench_sales_gap']:.1f}",
            ),
            (
                "Supplier Delay",
                r["delay_signal"],
                f"avg_delay_days={r['avg_delay_days']:.2f}, benchmark={r['bench_delay']:.2f}",
            ),
            (
                "Planning Gap",
                r["planning_signal"],
                f"abs_gap={abs(r['coverage_gap']):.2f}, benchmark={r['bench_gap']:.2f}",
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
                    "confidence": float(r["confidence"]),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["sku", "warehouse", "impact_score"], ascending=[True, True, False])


def attribution_step_breakdown(metrics_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    merged = _build_base(metrics_df, transactions)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "sku",
                "warehouse",
                "sales_qty",
                "receipt_qty",
                "sales_minus_receipt",
                "bench_sales_gap",
                "demand_delta",
                "avg_delay_days",
                "bench_delay",
                "delay_delta",
                "abs_gap",
                "bench_gap",
                "planning_delta",
                "dominant_factor",
                "confidence",
            ]
        )

    out = merged.copy()
    out["demand_delta"] = out["sales_minus_receipt"] - out["bench_sales_gap"]
    out["delay_delta"] = out["avg_delay_days"] - out["bench_delay"]
    out["abs_gap"] = abs(out["coverage_gap"])
    out["planning_delta"] = out["abs_gap"] - out["bench_gap"]

    def _dominant(row: pd.Series) -> str:
        scores = {
            "Demand Surge": row["demand_signal"],
            "Supplier Delay": row["delay_signal"],
            "Planning Gap": row["planning_signal"],
        }
        return max(scores, key=scores.get)

    out["dominant_factor"] = out.apply(_dominant, axis=1)

    keep_cols = [
        "sku",
        "warehouse",
        "sales_qty",
        "receipt_qty",
        "sales_minus_receipt",
        "bench_sales_gap",
        "demand_delta",
        "avg_delay_days",
        "bench_delay",
        "delay_delta",
        "abs_gap",
        "bench_gap",
        "planning_delta",
        "dominant_factor",
        "confidence",
    ]
    return out[keep_cols].sort_values(["sku", "warehouse"], ascending=[True, True])
