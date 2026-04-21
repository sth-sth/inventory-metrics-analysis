from __future__ import annotations

import pandas as pd


def generate_recommendations(metrics_df: pd.DataFrame, attribution_df: pd.DataFrame) -> pd.DataFrame:
    recs = []

    for _, r in metrics_df.iterrows():
        if r["inventory_health"] == "Stockout Risk":
            recs.append(
                {
                    "sku": r["sku"],
                    "priority": "P1",
                    "recommendation": "Increase reorder point and expedite inbound for this SKU.",
                    "why": f"coverage_gap={r['coverage_gap']:.2f}, DOH={r['doh']:.1f}",
                }
            )
        elif r["inventory_health"] == "Overstock Risk":
            recs.append(
                {
                    "sku": r["sku"],
                    "priority": "P2",
                    "recommendation": "Reduce replenishment cycle and trigger markdown/bundle strategy.",
                    "why": f"DOH={r['doh']:.1f} above control band",
                }
            )

    if not attribution_df.empty:
        top_factor = (
            attribution_df.groupby("factor", as_index=False)["impact_score"]
            .mean()
            .sort_values("impact_score", ascending=False)
            .head(1)
        )
        if not top_factor.empty:
            f = top_factor.iloc[0]
            recs.append(
                {
                    "sku": "ALL",
                    "priority": "P1",
                    "recommendation": f"Launch cross-functional action on '{f['factor']}' as dominant risk driver.",
                    "why": f"average impact_score={f['impact_score']:.2f}",
                }
            )

    if not recs:
        return pd.DataFrame(columns=["sku", "priority", "recommendation", "why"])
    return pd.DataFrame(recs)
