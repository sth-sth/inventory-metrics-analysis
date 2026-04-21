from __future__ import annotations

from pathlib import Path

import numpy as np
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


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def bilingual(lang: str, zh: str, en: str) -> str:
    return zh if lang == "中文" else en


with st.sidebar:
    language = st.radio("Language / 语言", ["中文", "English"], index=0)
    st.header(bilingual(language, "生产级控制台", "Deployment-grade Controls"))
    input_mode = st.radio(
        bilingual(language, "数据来源", "Data source"),
        [
            bilingual(language, "使用内置演示数据", "Use built-in demo data"),
            bilingual(language, "上传自定义 CSV 文件", "Upload custom CSV files"),
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader(bilingual(language, "阈值参数", "Custom Thresholds"))
    stockout_threshold = st.number_input(bilingual(language, "缺货 Gap 阈值", "Stockout gap threshold"), value=0.0, step=1.0)
    overstock_doh = st.number_input(bilingual(language, "超储 DOH 阈值", "Overstock DOH threshold"), value=120.0, step=5.0)
    delay_days = st.number_input(
        bilingual(language, "到货延迟阈值（天）", "Delayed receipt threshold (days)"),
        value=3.0,
        step=1.0,
    )

    st.markdown("---")
    st.subheader(bilingual(language, "决策参数", "Decision Parameters"))
    budget_cap = st.number_input(bilingual(language, "补货预算上限", "Replenishment budget cap"), min_value=0.0, value=15000.0, step=1000.0)
    target_service = st.slider(
        bilingual(language, "目标服务水平", "Target service level"),
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    st.markdown("---")
    st.caption(bilingual(language, "上传字段要求见 README.md", "Upload schema requirements are documented in README.md"))

st.markdown(
    f"""
<div class="hero">
  <h1>{bilingual(language, '库存智能决策云', 'Inventory Intelligence Cloud')}</h1>
  <p>{bilingual(language, '上传或使用演示数据，实时监控库存健康、识别风险并基于证据做归因和决策。', 'Upload or use demo data to monitor inventory health, detect risks, and support evidence-based decisions.')}</p>
</div>
""",
    unsafe_allow_html=True,
)

inventory_file = None
transactions_file = None
if input_mode == bilingual(language, "上传自定义 CSV 文件", "Upload custom CSV files"):
    u1, u2 = st.columns(2)
    with u1:
        inventory_file = st.file_uploader(bilingual(language, "库存快照 CSV", "Inventory snapshot CSV"), type=["csv"])
    with u2:
        transactions_file = st.file_uploader(bilingual(language, "流水交易 CSV", "Transactions CSV"), type=["csv"])

if input_mode == bilingual(language, "使用内置演示数据", "Use built-in demo data"):
    try:
        inventory_df, transactions_df = load_demo_data()
        st.success(bilingual(language, "演示数据已加载，可在侧边栏切换到上传模式。", "Demo data loaded. Switch to upload mode in sidebar for your own data."))
    except Exception as exc:
        st.error(f"{bilingual(language, '加载演示数据失败', 'Failed to load demo data')}: {exc}")
        st.stop()
else:
    if not inventory_file or not transactions_file:
        st.info(bilingual(language, "请上传两份 CSV 后继续。", "Upload both CSV files to continue."))
        st.stop()
    try:
        bundle = load_bundle(inventory_file, transactions_file)
        inventory_df, transactions_df = bundle.inventory, bundle.transactions
    except Exception as exc:
        st.error(f"{bilingual(language, '数据加载失败', 'Data loading failed')}: {exc}")
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

flt1, flt2 = st.columns(2)
with flt1:
    selected_warehouses = st.multiselect(
        bilingual(language, "按仓库筛选", "Filter by warehouse"),
        options=sorted(metrics_df["warehouse"].unique().tolist()),
        default=sorted(metrics_df["warehouse"].unique().tolist()),
    )
with flt2:
    selected_categories = st.multiselect(
        bilingual(language, "按品类筛选", "Filter by category"),
        options=sorted(metrics_df["category"].unique().tolist()),
        default=sorted(metrics_df["category"].unique().tolist()),
    )

filtered_metrics = metrics_df[
    metrics_df["warehouse"].isin(selected_warehouses) & metrics_df["category"].isin(selected_categories)
].copy()
if filtered_metrics.empty:
    st.warning(bilingual(language, "当前筛选无数据，请调整筛选条件。", "No data after filtering. Please adjust filters."))
    st.stop()

filtered_kpis = build_kpi_summary(filtered_metrics, transactions_df[transactions_df["warehouse"].isin(selected_warehouses)])
filtered_alerts = alerts_df[alerts_df["warehouse"].isin(selected_warehouses)] if not alerts_df.empty else alerts_df
filtered_attr = attribution_df[attribution_df["warehouse"].isin(selected_warehouses)] if not attribution_df.empty else attribution_df
filtered_recs = recs_df[recs_df["sku"].isin(filtered_metrics["sku"]) | (recs_df["sku"] == "ALL")] if not recs_df.empty else recs_df

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(bilingual(language, "库存总价值", "Total Inventory Value"), f"${filtered_kpis['total_inventory_value']:,.0f}")
c2.metric(bilingual(language, "缺货风险 SKU", "Stockout Risk SKU"), f"{filtered_kpis['stockout_risk_sku_count']}")
c3.metric(bilingual(language, "超储风险 SKU", "Overstock Risk SKU"), f"{filtered_kpis['overstock_risk_sku_count']}")
c4.metric(bilingual(language, "周转率代理", "Inventory Turnover Proxy"), f"{filtered_kpis['inventory_turnover_proxy']:.2f}")
c5.metric(bilingual(language, "服务水平代理", "Service Level Proxy"), f"{filtered_kpis['service_level_proxy']:.1%}")

tab_overview, tab_monitoring, tab_diagnosis, tab_actions = st.tabs(
    [
        bilingual(language, "管理层总览", "Executive Overview"),
        bilingual(language, "风险监控", "Risk Monitoring"),
        bilingual(language, "归因诊断", "Root-Cause Diagnosis"),
        bilingual(language, "决策行动", "Action Center"),
    ]
)

with st.expander(bilingual(language, "方法说明（含 Gap 计算公式）", "Method Notes (including Gap formula)"), expanded=False):
    st.markdown(
        bilingual(
            language,
            """
**核心公式**

- DOH = on_hand_qty / avg_daily_demand
- lead_time_demand = avg_daily_demand * lead_time_days
- safety_stock = z * std(avg_daily_demand) * sqrt(lead_time_days)
- reorder_point = lead_time_demand + safety_stock
- coverage_gap (Gap) = on_hand_qty - reorder_point

**解释**

- Gap < 0: 当前库存低于再订货点，存在缺货风险。
- Gap > 0: 库存覆盖高于再订货点，缓冲较充足。
- 在决策中，建议补货量基础值 = max(reorder_point - on_hand_qty, 0)。
""",
            """
**Core formulas**

- DOH = on_hand_qty / avg_daily_demand
- lead_time_demand = avg_daily_demand * lead_time_days
- safety_stock = z * std(avg_daily_demand) * sqrt(lead_time_days)
- reorder_point = lead_time_demand + safety_stock
- coverage_gap (Gap) = on_hand_qty - reorder_point

**Interpretation**

- Gap < 0: inventory is below reorder point and stockout risk exists.
- Gap > 0: inventory is above reorder point with stronger coverage.
- Base replenishment qty in decisions = max(reorder_point - on_hand_qty, 0).
""",
        )
    )

with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        fig_health = px.histogram(
            filtered_metrics,
            x="inventory_health",
            color="inventory_health",
            color_discrete_map={
                "Stockout Risk": "#dc2626",
                "Overstock Risk": "#f59e0b",
                "Healthy": "#0ea5a4",
                "Watch": "#0f2740",
            },
            title=bilingual(language, "库存健康分布", "Inventory Health Distribution"),
        )
        st.plotly_chart(fig_health, use_container_width=True)

    with right:
        fig_abc = px.pie(
            filtered_metrics,
            names="abc_class",
            values="inventory_value",
            color="abc_class",
            color_discrete_map={"A": "#0f2740", "B": "#0ea5a4", "C": "#f59e0b"},
            title=bilingual(language, "ABC 类别库存价值占比", "Inventory Value by ABC Class"),
        )
        st.plotly_chart(fig_abc, use_container_width=True)

with tab_monitoring:
    fig_scatter = px.scatter(
        filtered_metrics,
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
        title=bilingual(language, "SKU 风险图：DOH vs Gap", "SKU-level Risk Map: DOH vs Coverage Gap"),
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.subheader(bilingual(language, "报警中心", "Alert Center"))
    st.dataframe(filtered_alerts, use_container_width=True)
    st.download_button(
        bilingual(language, "下载报警 CSV", "Download Alerts CSV"),
        data=to_csv_bytes(filtered_alerts),
        file_name="alerts.csv",
        mime="text/csv",
    )

with tab_diagnosis:
    st.caption(
        bilingual(
            language,
            "归因基于观测数据相对基准的偏离，并输出样本覆盖置信度，不做无证据推断。",
            "Attribution uses observed deviation vs benchmark and outputs confidence by sample coverage.",
        )
    )
    if filtered_attr.empty:
        st.success(bilingual(language, "当前阈值下未发现缺货风险项。", "No stockout-risk items detected under current thresholds."))
    else:
        fig_attr = px.bar(
            filtered_attr,
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
            title=bilingual(language, "SKU 归因分解", "Attribution by SKU and Factor"),
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        st.dataframe(filtered_attr, use_container_width=True)

    st.download_button(
        bilingual(language, "下载归因 CSV", "Download Attribution CSV"),
        data=to_csv_bytes(filtered_attr),
        file_name="attribution.csv",
        mime="text/csv",
    )

with tab_actions:
    st.subheader(bilingual(language, "数据驱动业务建议", "Data-driven Business Recommendations"))
    st.dataframe(filtered_recs, use_container_width=True)
    st.download_button(
        bilingual(language, "下载建议 CSV", "Download Recommendations CSV"),
        data=to_csv_bytes(filtered_recs),
        file_name="recommendations.csv",
        mime="text/csv",
    )

    st.subheader(bilingual(language, "智能场景模拟与补货决策", "Smart Scenario Simulation and Replenishment Decision"))
    s1, s2 = st.columns(2)
    with s1:
        demand_shift = st.slider(
            bilingual(language, "需求变化（%）", "Demand shift (%)"),
            min_value=-30,
            max_value=60,
            value=10,
            step=5,
        )
    with s2:
        lead_shift = st.slider(
            bilingual(language, "交期变化（%）", "Lead-time shift (%)"),
            min_value=-30,
            max_value=60,
            value=10,
            step=5,
        )

    sim_input = filtered_metrics[
        ["date", "sku", "category", "warehouse", "on_hand_qty", "avg_daily_demand", "lead_time_days", "unit_cost"]
    ].copy()
    sim_input["avg_daily_demand"] = sim_input["avg_daily_demand"] * (1 + demand_shift / 100)
    sim_input["lead_time_days"] = np.clip(sim_input["lead_time_days"] * (1 + lead_shift / 100), 0, None)

    sim_metrics = abc_classification(build_inventory_metrics(sim_input))
    stockout_now = int((filtered_metrics["coverage_gap"] < 0).sum())
    stockout_sim = int((sim_metrics["coverage_gap"] < 0).sum())
    value_at_risk = float(sim_metrics.loc[sim_metrics["coverage_gap"] < 0, "inventory_value"].sum())

    d1, d2, d3 = st.columns(3)
    d1.metric(bilingual(language, "当前缺货风险数", "Current Stockout Risk"), stockout_now)
    d2.metric(bilingual(language, "模拟缺货风险数", "Simulated Stockout Risk"), stockout_sim)
    d3.metric(bilingual(language, "模拟风险价值", "Simulated Value at Risk"), f"${value_at_risk:,.0f}")

    plan = sim_metrics.copy()
    plan["base_order_qty"] = np.maximum(plan["reorder_point"] - plan["on_hand_qty"], 0)
    class_weight = {"A": 1.2, "B": 1.0, "C": 0.8}
    plan["class_weight"] = plan["abc_class"].map(class_weight).fillna(1.0)
    uplift = 1 + max(target_service - 0.90, 0) * 2
    plan["recommended_order_qty"] = np.ceil(plan["base_order_qty"] * plan["class_weight"] * uplift)
    plan["investment"] = plan["recommended_order_qty"] * plan["unit_cost"]
    plan["risk_score"] = np.maximum(-plan["coverage_gap"], 0) * plan["unit_cost"] * plan["class_weight"]
    plan = plan[plan["recommended_order_qty"] > 0].sort_values(["risk_score", "inventory_value"], ascending=[False, False])

    if plan.empty:
        st.success(bilingual(language, "当前条件下无需补货。", "No replenishment needed under current scenario."))
    else:
        plan["cum_investment"] = plan["investment"].cumsum()
        selected_plan = plan[plan["cum_investment"] <= budget_cap].copy()

        st.caption(
            bilingual(
                language,
                "以下清单按风险评分排序并受预算约束，可直接支持补货优先级决策。",
                "The list is ranked by risk score and constrained by budget for direct replenishment prioritization.",
            )
        )
        show_plan_cols = [
            "sku",
            "warehouse",
            "abc_class",
            "coverage_gap",
            "recommended_order_qty",
            "unit_cost",
            "investment",
            "risk_score",
        ]
        st.dataframe(selected_plan[show_plan_cols], use_container_width=True)
        st.download_button(
            bilingual(language, "下载补货决策 CSV", "Download Replenishment Plan CSV"),
            data=to_csv_bytes(selected_plan[show_plan_cols]),
            file_name="replenishment_plan.csv",
            mime="text/csv",
        )

    st.subheader(bilingual(language, "运营下钻明细", "Operational Drill-down"))
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
        filtered_metrics[show_cols].sort_values(["inventory_health", "inventory_value"], ascending=[True, False]),
        use_container_width=True,
    )
