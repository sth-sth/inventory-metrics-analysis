from __future__ import annotations

import pandas as pd


def generate_recommendations(metrics_df: pd.DataFrame, factor_summary_df: pd.DataFrame, language: str = "中文") -> pd.DataFrame:
    recs = []
    is_cn = language == "中文"

    for _, r in metrics_df.iterrows():
        if r["inventory_health"] == "Stockout Risk":
            recs.append(
                {
                    "sku": r["sku"],
                    "priority": "P1",
                    "recommendation": (
                        "提高该 SKU 再订货点并加急在途补货。"
                        if is_cn
                        else "Increase reorder point and expedite inbound for this SKU."
                    ),
                    "why": f"coverage_gap={r['coverage_gap']:.2f}, DOH={r['doh']:.1f}",
                }
            )
        elif r["inventory_health"] == "Overstock Risk":
            recs.append(
                {
                    "sku": r["sku"],
                    "priority": "P2",
                    "recommendation": (
                        "缩短补货周期，并触发降价/组合销售去库存策略。"
                        if is_cn
                        else "Reduce replenishment cycle and trigger markdown/bundle strategy."
                    ),
                    "why": (
                        f"DOH={r['doh']:.1f} 高于控制带"
                        if is_cn
                        else f"DOH={r['doh']:.1f} above control band"
                    ),
                }
            )

    if not factor_summary_df.empty and {"factor", "impact_score"}.issubset(factor_summary_df.columns):
        top_factor = (
            factor_summary_df.groupby("factor", as_index=False)["impact_score"]
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
                    "recommendation": (
                        f"围绕主导风险因子“{f['factor']}”发起跨部门联动改善。"
                        if is_cn
                        else f"Launch cross-functional action on '{f['factor']}' as dominant risk driver."
                    ),
                    "why": (
                        f"平均影响分={f['impact_score']:.2f}"
                        if is_cn
                        else f"average impact_score={f['impact_score']:.2f}"
                    ),
                }
            )

    if not recs:
        return pd.DataFrame(columns=["sku", "priority", "recommendation", "why"])
    return pd.DataFrame(recs)
