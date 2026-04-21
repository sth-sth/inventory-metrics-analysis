from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from src.alerts import AlertConfig, detect_alerts
from src.attribution import root_cause_attribution
from src.data_io import load_bundle, load_inventory_csv, load_transactions_csv
from src.metrics import abc_classification, build_inventory_metrics, build_kpi_summary
from src.recommendations import generate_recommendations


st.set_page_config(page_title="Inventory Intelligence Cloud", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
:root {
    --brand-ink: #0f2740;
    --brand-accent: #0ea5a4;
    --brand-warn: #f59e0b;
    --brand-danger: #dc2626;
    --bg-soft: #eef5f7;
}
html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(14,165,164,0.12), transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(15,39,64,0.08), transparent 30%),
        linear-gradient(180deg, #f8fbfc 0%, #f3f8fa 100%);
}
.hero {
    border: 1px solid rgba(15,39,64,0.10);
    background: linear-gradient(120deg, rgba(15,39,64,0.94) 0%, rgba(14,165,164,0.90) 100%);
    border-radius: 16px;
    padding: 20px 24px;
    color: #ffffff;
    margin-bottom: 16px;
}
.hero h1 {
    margin: 0;
    font-size: 1.75rem;
}
.hero p {
    margin: 8px 0 0 0;
    opacity: 0.95;
}
.section-card {
    border: 1px solid rgba(15,39,64,0.10);
    border-radius: 14px;
    background: #ffffff;
    padding: 10px 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).parent
    with open(root / "data" / "inventory_demo.csv", "rb") as inv_f, open(root / "data" / "transactions_demo.csv", "rb") as tx_f:
        return load_inventory_csv(inv_f), load_transactions_csv(tx_f)


st.markdown(
    """
<div class="hero">
  <h1>Inventory Intelligence Cloud</h1>
  <p>Upload data, monitor inventory health in real time, detect risk early, and trace root causes with evidence-backed scoring.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Deployment-grade Controls")
    input_mode = st.radio("Data source", ["Use built-in demo data", "Upload custom CSV files"], index=0)

    st.markdown("---")
    st.subheader("Custom Thresholds")
    stockout_threshold = st.number_input("Stockout gap threshold", value=0.0, step=1.0)
    overstock_doh = st.number_input("Overstock DOH threshold", value=120.0, step=5.0)
    delay_days = st.number_input("Delayed receipt threshold (days)", value=3.0, step=1.0)

    st.markdown("---")
    st.caption("Upload schema requirements are documented in README.md")

inventory_file = None
transactions_file = None
if input_mode == "Upload custom CSV files":
    u1, u2 = st.columns(2)
    with u1:
        inventory_file = st.file_uploader("Inventory snapshot CSV", type=["csv"])
    with u2:
        transactions_file = st.file_uploader("Transactions CSV", type=["csv"])

if input_mode == "Use built-in demo data":
    try:
        inventory_df, transactions_df = load_demo_data()
        st.success("Demo data loaded. Switch to upload mode in sidebar for your own data.")
    except Exception as exc:
        st.error(f"Failed to load demo data: {exc}")
        st.stop()
else:
    if not inventory_file or not transactions_file:
        st.info("Upload both CSV files to continue.")
        st.stop()
    try:
        bundle = load_bundle(inventory_file, transactions_file)
        inventory_df, transactions_df = bundle.inventory, bundle.transactions
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")
        st.stop()

metrics_df = build_inventory_metrics(inventory_df)
metrics_df = abc_classification(metrics_df)
kpis = build_kpi_summary(metrics_df, transactions_df)

alert_cfg = AlertConfig(
    stockout_gap_threshold=stockout_threshold,
    overstock_doh_threshold=overstock_doh,
    delayed_receipt_days=delay_days,
)
alerts_df = detect_alerts(metrics_df, transactions_df, alert_cfg)
attribution_df = root_cause_attribution(metrics_df, transactions_df)
recs_df = generate_recommendations(metrics_df, attribution_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Inventory Value", f"${kpis['total_inventory_value']:,.0f}")
c2.metric("Stockout Risk SKU", f"{kpis['stockout_risk_sku_count']}")
c3.metric("Overstock Risk SKU", f"{kpis['overstock_risk_sku_count']}")
c4.metric("Inventory Turnover Proxy", f"{kpis['inventory_turnover_proxy']:.2f}")
c5.metric("Service Level Proxy", f"{kpis['service_level_proxy']:.1%}")

tab_overview, tab_monitoring, tab_diagnosis, tab_actions = st.tabs(
    ["Executive Overview", "Risk Monitoring", "Root-Cause Diagnosis", "Action Center"]
)

with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        fig_health = px.histogram(
            metrics_df,
            x="inventory_health",
            color="inventory_health",
            color_discrete_map={
                "Stockout Risk": "#dc2626",
                "Overstock Risk": "#f59e0b",
                "Healthy": "#0ea5a4",
                "Watch": "#0f2740",
            },
            title="Inventory Health Distribution",
        )
        st.plotly_chart(fig_health, use_container_width=True)

    with right:
        fig_abc = px.pie(
            metrics_df,
            names="abc_class",
            values="inventory_value",
            color="abc_class",
            color_discrete_map={"A": "#0f2740", "B": "#0ea5a4", "C": "#f59e0b"},
            title="Inventory Value by ABC Class",
        )
        st.plotly_chart(fig_abc, use_container_width=True)

with tab_monitoring:
    fig_scatter = px.scatter(
        metrics_df,
        x="doh",
        y="coverage_gap",
        color="inventory_health",
        color_discrete_map={
            "Stockout Risk": "#dc2626",
            "Overstock Risk": "#f59e0b",
            "Healthy": "#0ea5a4",
            "Watch": "#0f2740",
        },
        size="inventory_value",
        hover_data=["sku", "warehouse", "category", "abc_class"],
        title="SKU-level Risk Map: DOH vs Coverage Gap",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.subheader("Alert Center")
    st.dataframe(alerts_df, use_container_width=True)

with tab_diagnosis:
    st.caption("Attribution only uses observed data deviations vs benchmark and reports confidence by sample coverage.")
    if attribution_df.empty:
        st.success("No stockout-risk items detected under current thresholds.")
    else:
        fig_attr = px.bar(
            attribution_df,
            x="sku",
            y="impact_score",
            color="factor",
            color_discrete_map={
                "Demand Surge": "#0f2740",
                "Supplier Delay": "#dc2626",
                "Planning Gap": "#0ea5a4",
            },
            barmode="group",
            hover_data=["warehouse", "evidence", "confidence"],
            title="Attribution by SKU and Factor",
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        st.dataframe(attribution_df, use_container_width=True)

with tab_actions:
    st.subheader("Data-driven Business Recommendations")
    st.dataframe(recs_df, use_container_width=True)

    st.subheader("Operational Drill-down")
    show_cols = [
        "date",
        "sku",
        "category",
        "warehouse",
        "on_hand_qty",
        "avg_daily_demand",
        "lead_time_days",
        "doh",
        "reorder_point",
        "coverage_gap",
        "inventory_health",
        "abc_class",
        "inventory_value",
    ]
    st.dataframe(
        metrics_df[show_cols].sort_values(["inventory_health", "inventory_value"], ascending=[True, False]),
        use_container_width=True,
    )
