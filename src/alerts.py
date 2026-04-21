from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AlertConfig:
    stockout_gap_threshold: float = 0.0
    overstock_doh_threshold: float = 120.0
    delayed_receipt_days: float = 3.0


def detect_alerts(metrics_df: pd.DataFrame, transactions: pd.DataFrame, cfg: AlertConfig) -> pd.DataFrame:
    alerts = []

    stockout = metrics_df[metrics_df["coverage_gap"] < cfg.stockout_gap_threshold]
    for _, r in stockout.iterrows():
        alerts.append(
            {
                "severity": "High",
                "alert_type": "Stockout Risk",
                "sku": r["sku"],
                "warehouse": r["warehouse"],
                "detail": f"Coverage gap={r['coverage_gap']:.2f} below threshold",
            }
        )

    overstock = metrics_df[metrics_df["doh"] > cfg.overstock_doh_threshold]
    for _, r in overstock.iterrows():
        alerts.append(
            {
                "severity": "Medium",
                "alert_type": "Overstock Risk",
                "sku": r["sku"],
                "warehouse": r["warehouse"],
                "detail": f"DOH={r['doh']:.1f} exceeds threshold",
            }
        )

    late = transactions[
        (transactions["event_type"] == "receipt")
        & (transactions["delay_days"] > cfg.delayed_receipt_days)
    ]
    for _, r in late.iterrows():
        alerts.append(
            {
                "severity": "High",
                "alert_type": "Supplier Delay",
                "sku": r["sku"],
                "warehouse": r["warehouse"],
                "detail": f"Receipt delayed by {r['delay_days']:.1f} days (supplier={r['supplier']})",
            }
        )

    if not alerts:
        return pd.DataFrame(columns=["severity", "alert_type", "sku", "warehouse", "detail"])

    return pd.DataFrame(alerts)
