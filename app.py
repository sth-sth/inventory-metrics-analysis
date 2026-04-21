from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.alerts import AlertConfig, detect_alerts
from src.attribution import attribution_step_breakdown, root_cause_attribution
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


LANG = {
    "中文": {
        "app_title": "库存智能决策云",
        "app_desc": "纯中文界面，支持库存监控、归因诊断、场景模拟与补货决策。",
        "sidebar_title": "控制台",
        "input_mode": "数据来源",
        "input_demo": "使用内置演示数据",
        "input_upload": "上传自定义 CSV",
        "th_title": "阈值参数",
        "stockout": "缺货阈值（Gap < 阈值）",
        "overstock": "超储阈值（DOH > 阈值）",
        "delay": "延迟阈值（天）",
        "decision": "决策参数",
        "budget": "补货预算上限",
        "service": "目标服务水平",
        "up_hint": "上传字段要求请见 README。",
        "up_inv": "库存快照 CSV",
        "up_tx": "交易流水 CSV",
        "kpi_value": "库存总价值",
        "kpi_so": "缺货风险 SKU",
        "kpi_os": "超储风险 SKU",
        "kpi_turn": "周转率代理",
        "kpi_svc": "服务水平代理",
        "tab1": "管理层总览",
        "tab2": "风险监控",
        "tab3": "归因诊断",
        "tab4": "决策行动",
        "method": "计算方法（公式）",
        "overview_h": "库存健康分布",
        "overview_abc": "ABC 库存价值占比",
        "monitor_scatter": "SKU 风险图（DOH 与 Gap）",
        "alert_center": "报警中心",
        "download_alerts": "下载报警数据",
        "diag_caption": "按步骤展示每个 SKU 相对基准多了多少、少了多少。",
        "sku_pick": "选择要诊断的 SKU",
        "step_cards": "SKU 步骤差值卡片",
        "step_table": "SKU 步骤明细",
        "factor_table": "SKU 因子贡献",
        "download_diag": "下载归因明细",
        "recs": "业务建议",
        "download_recs": "下载建议",
        "sim": "场景模拟与补货优先级",
        "demand_shift": "需求变化（%）",
        "lead_shift": "交期变化（%）",
        "risk_now": "当前缺货风险数",
        "risk_sim": "模拟缺货风险数",
        "var_sim": "模拟风险价值",
        "plan_cap": "预算内补货清单",
        "download_plan": "下载补货清单",
        "drill": "运营下钻明细",
        "no_data": "当前筛选无数据，请调整筛选条件。",
        "demo_ok": "演示数据已加载。",
        "demo_fail": "演示数据加载失败",
        "need_upload": "请上传两份 CSV 后继续。",
        "load_fail": "数据加载失败",
        "filter_wh": "按仓库筛选",
        "filter_cat": "按品类筛选",
        "file_alert": "报警.csv",
        "file_diag": "归因明细.csv",
        "file_recs": "建议.csv",
        "file_plan": "补货清单.csv",
    },
    "English": {
        "app_title": "Inventory Intelligence Cloud",
        "app_desc": "Pure English interface for monitoring, attribution diagnosis, simulation, and replenishment decisions.",
        "sidebar_title": "Control Panel",
        "input_mode": "Data Source",
        "input_demo": "Use built-in demo data",
        "input_upload": "Upload custom CSV",
        "th_title": "Thresholds",
        "stockout": "Stockout threshold (Gap < threshold)",
        "overstock": "Overstock threshold (DOH > threshold)",
        "delay": "Delay threshold (days)",
        "decision": "Decision Settings",
        "budget": "Replenishment budget cap",
        "service": "Target service level",
        "up_hint": "See README for upload schema.",
        "up_inv": "Inventory snapshot CSV",
        "up_tx": "Transactions CSV",
        "kpi_value": "Total Inventory Value",
        "kpi_so": "Stockout Risk SKU",
        "kpi_os": "Overstock Risk SKU",
        "kpi_turn": "Turnover Proxy",
        "kpi_svc": "Service Level Proxy",
        "tab1": "Executive Overview",
        "tab2": "Risk Monitoring",
        "tab3": "Attribution Diagnosis",
        "tab4": "Action Center",
        "method": "Calculation Method (Formula)",
        "overview_h": "Inventory Health Distribution",
        "overview_abc": "Inventory Value by ABC Class",
        "monitor_scatter": "SKU Risk Map (DOH vs Gap)",
        "alert_center": "Alert Center",
        "download_alerts": "Download alerts",
        "diag_caption": "Step-by-step deltas show how much each SKU is above or below benchmark.",
        "sku_pick": "Select SKU for diagnosis",
        "step_cards": "SKU Step Delta Cards",
        "step_table": "SKU Step Breakdown",
        "factor_table": "SKU Factor Contribution",
        "download_diag": "Download attribution details",
        "recs": "Business Recommendations",
        "download_recs": "Download recommendations",
        "sim": "Scenario Simulation and Replenishment Priority",
        "demand_shift": "Demand shift (%)",
        "lead_shift": "Lead-time shift (%)",
        "risk_now": "Current stockout risk",
        "risk_sim": "Simulated stockout risk",
        "var_sim": "Simulated value at risk",
        "plan_cap": "Budget-constrained replenishment list",
        "download_plan": "Download replenishment list",
        "drill": "Operational Drill-down",
        "no_data": "No data after filtering. Please adjust filters.",
        "demo_ok": "Demo data loaded.",
        "demo_fail": "Failed to load demo data",
        "need_upload": "Upload both CSV files to continue.",
        "load_fail": "Data loading failed",
        "filter_wh": "Filter by warehouse",
        "filter_cat": "Filter by category",
        "file_alert": "alerts.csv",
        "file_diag": "attribution_details.csv",
        "file_recs": "recommendations.csv",
        "file_plan": "replenishment_plan.csv",
    },
}


HEALTH_MAP = {
    "中文": {
        "Stockout Risk": "缺货风险",
        "Overstock Risk": "超储风险",
        "Healthy": "健康",
        "Watch": "观察",
    },
    "English": {
        "Stockout Risk": "Stockout Risk",
        "Overstock Risk": "Overstock Risk",
        "Healthy": "Healthy",
        "Watch": "Watch",
    },
}


FACTOR_MAP = {
    "中文": {
        "Demand Surge": "需求激增",
        "Supplier Delay": "供应延迟",
        "Planning Gap": "计划缺口",
    },
    "English": {
        "Demand Surge": "Demand Surge",
        "Supplier Delay": "Supplier Delay",
        "Planning Gap": "Planning Gap",
    },
}


def localize_health(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = df.copy()
    out["inventory_health"] = out["inventory_health"].map(HEALTH_MAP[lang]).fillna(out["inventory_health"])
    return out


def localize_factor(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = df.copy()
    if "factor" in out.columns:
        out["factor"] = out["factor"].map(FACTOR_MAP[lang]).fillna(out["factor"])
    if "dominant_factor" in out.columns:
        out["dominant_factor"] = out["dominant_factor"].map(FACTOR_MAP[lang]).fillna(out["dominant_factor"])
    return out


def render_formula(lang: str) -> None:
    if lang == "中文":
        st.markdown(
            r"""
$$
	ext{DOH} = \frac{\text{当前库存}}{\text{日均需求}}
$$

$$
	ext{交期需求} = \text{日均需求} \times \text{交期天数}
$$

$$
	ext{安全库存} = z \times \sigma_{\text{需求}} \times \sqrt{\text{交期天数}}
$$

$$
	ext{再订货点} = \text{交期需求} + \text{安全库存}
$$

$$
	ext{Gap} = \text{当前库存} - \text{再订货点}
$$

Gap 小于 0 代表库存低于保护线，越小代表缺货风险越高。
"""
        )
    else:
        st.markdown(
            r"""
$$
	ext{DOH} = \frac{\text{on\_hand\_qty}}{\text{avg\_daily\_demand}}
$$

$$
	ext{Lead-time Demand} = \text{avg\_daily\_demand} \times \text{lead\_time\_days}
$$

$$
	ext{Safety Stock} = z \times \sigma_{\text{demand}} \times \sqrt{\text{lead\_time\_days}}
$$

$$
	ext{Reorder Point} = \text{Lead-time Demand} + \text{Safety Stock}
$$

$$
	ext{Gap} = \text{on\_hand\_qty} - \text{reorder\_point}
$$

Gap below 0 means inventory is under the protection line; lower value means higher stockout risk.
"""
        )


with st.sidebar:
    lang = st.selectbox("界面语言 / Interface", ["中文", "English"], index=0)
    t = LANG[lang]

    st.header(t["sidebar_title"])
    input_mode = st.radio(t["input_mode"], [t["input_demo"], t["input_upload"]], index=0)

    st.markdown("---")
    st.subheader(t["th_title"])
    stockout_threshold = st.number_input(t["stockout"], value=0.0, step=1.0)
    overstock_doh = st.number_input(t["overstock"], value=120.0, step=5.0)
    delay_days = st.number_input(t["delay"], value=3.0, step=1.0)

    st.markdown("---")
    st.subheader(t["decision"])
    budget_cap = st.number_input(t["budget"], min_value=0.0, value=15000.0, step=1000.0)
    target_service = st.slider(t["service"], min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    st.markdown("---")
    st.caption(t["up_hint"])


st.markdown(
    f"""
<div class="hero">
  <h1>{t['app_title']}</h1>
  <p>{t['app_desc']}</p>
</div>
""",
    unsafe_allow_html=True,
)

inventory_file = None
transactions_file = None
if input_mode == t["input_upload"]:
    c1, c2 = st.columns(2)
    with c1:
        inventory_file = st.file_uploader(t["up_inv"], type=["csv"])
    with c2:
        transactions_file = st.file_uploader(t["up_tx"], type=["csv"])

if input_mode == t["input_demo"]:
    try:
        inventory_df, transactions_df = load_demo_data()
        st.success(t["demo_ok"])
    except Exception as exc:
        st.error(f"{t['demo_fail']}: {exc}")
        st.stop()
else:
    if not inventory_file or not transactions_file:
        st.info(t["need_upload"])
        st.stop()
    try:
        bundle = load_bundle(inventory_file, transactions_file)
        inventory_df, transactions_df = bundle.inventory, bundle.transactions
    except Exception as exc:
        st.error(f"{t['load_fail']}: {exc}")
        st.stop()

metrics_df = abc_classification(build_inventory_metrics(inventory_df))
alerts_df = detect_alerts(
    metrics_df,
    transactions_df,
    AlertConfig(
        stockout_gap_threshold=stockout_threshold,
        overstock_doh_threshold=overstock_doh,
        delayed_receipt_days=delay_days,
    ),
)
attribution_df = root_cause_attribution(metrics_df, transactions_df)
step_df = attribution_step_breakdown(metrics_df, transactions_df)
recs_df = generate_recommendations(metrics_df, attribution_df)

f1, f2 = st.columns(2)
with f1:
    wh_options = sorted(metrics_df["warehouse"].unique().tolist())
    selected_wh = st.multiselect(t["filter_wh"], options=wh_options, default=wh_options)
with f2:
    cat_options = sorted(metrics_df["category"].unique().tolist())
    selected_cat = st.multiselect(t["filter_cat"], options=cat_options, default=cat_options)

filtered_metrics = metrics_df[
    metrics_df["warehouse"].isin(selected_wh) & metrics_df["category"].isin(selected_cat)
].copy()

if filtered_metrics.empty:
    st.warning(t["no_data"])
    st.stop()

filtered_alerts = alerts_df[alerts_df["warehouse"].isin(selected_wh)] if not alerts_df.empty else alerts_df
filtered_attr = attribution_df[attribution_df["warehouse"].isin(selected_wh)] if not attribution_df.empty else attribution_df
filtered_steps = step_df[step_df["warehouse"].isin(selected_wh)] if not step_df.empty else step_df
filtered_recs = recs_df[recs_df["sku"].isin(filtered_metrics["sku"]) | (recs_df["sku"] == "ALL")]

kpi_tx = transactions_df[transactions_df["warehouse"].isin(selected_wh)]
kpis = build_kpi_summary(filtered_metrics, kpi_tx)

local_metrics = localize_health(filtered_metrics, lang)
local_attr = localize_factor(filtered_attr, lang)
local_steps = localize_factor(filtered_steps, lang)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(t["kpi_value"], f"${kpis['total_inventory_value']:,.0f}")
c2.metric(t["kpi_so"], f"{kpis['stockout_risk_sku_count']}")
c3.metric(t["kpi_os"], f"{kpis['overstock_risk_sku_count']}")
c4.metric(t["kpi_turn"], f"{kpis['inventory_turnover_proxy']:.2f}")
c5.metric(t["kpi_svc"], f"{kpis['service_level_proxy']:.1%}")

tabs = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"]])

with st.expander(t["method"], expanded=False):
    render_formula(lang)

with tabs[0]:
    a, b = st.columns(2)
    with a:
        fig1 = px.histogram(local_metrics, x="inventory_health", color="inventory_health", title=t["overview_h"])
        st.plotly_chart(fig1, use_container_width=True)
    with b:
        fig2 = px.pie(local_metrics, names="abc_class", values="inventory_value", title=t["overview_abc"])
        st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    fig3 = px.scatter(
        local_metrics,
        x="doh",
        y="coverage_gap",
        color="inventory_health",
        size="inventory_value",
        hover_data=["sku", "warehouse", "category", "abc_class"],
        title=t["monitor_scatter"],
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader(t["alert_center"])
    st.dataframe(filtered_alerts, use_container_width=True)
    st.download_button(t["download_alerts"], to_csv_bytes(filtered_alerts), t["file_alert"], "text/csv")

with tabs[2]:
    st.caption(t["diag_caption"])
    if local_steps.empty:
        st.info("No stockout-risk SKU under current thresholds." if lang == "English" else "当前阈值下没有缺货风险 SKU。")
    else:
        sku_list = sorted(local_steps["sku"].unique().tolist())
        sku_pick = st.selectbox(t["sku_pick"], options=sku_list)
        sku_step = local_steps[local_steps["sku"] == sku_pick].copy()
        sku_attr = local_attr[local_attr["sku"] == sku_pick].copy()

        step_row = sku_step.iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("需求偏差" if lang == "中文" else "Demand Delta", f"{step_row['demand_delta']:+.2f}")
        d2.metric("延迟偏差" if lang == "中文" else "Delay Delta", f"{step_row['delay_delta']:+.2f}")
        d3.metric("计划偏差" if lang == "中文" else "Planning Delta", f"{step_row['planning_delta']:+.2f}")

        chart_df = pd.DataFrame(
            {
                ("步骤" if lang == "中文" else "Step"): [
                    "需求偏差" if lang == "中文" else "Demand Delta",
                    "延迟偏差" if lang == "中文" else "Delay Delta",
                    "计划偏差" if lang == "中文" else "Planning Delta",
                ],
                ("偏差值" if lang == "中文" else "Delta"): [
                    float(step_row["demand_delta"]),
                    float(step_row["delay_delta"]),
                    float(step_row["planning_delta"]),
                ],
            }
        )
        fig4 = px.bar(chart_df, x=chart_df.columns[0], y=chart_df.columns[1], title=t["step_cards"])
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader(t["step_table"])
        step_show = sku_step[
            [
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
        ].copy()
        if lang == "中文":
            step_show = step_show.rename(
                columns={
                    "sku": "SKU",
                    "warehouse": "仓库",
                    "sales_qty": "销售量",
                    "receipt_qty": "入库量",
                    "sales_minus_receipt": "销售-入库",
                    "bench_sales_gap": "基准(销售-入库)",
                    "demand_delta": "需求偏差",
                    "avg_delay_days": "平均延迟天",
                    "bench_delay": "基准延迟天",
                    "delay_delta": "延迟偏差",
                    "abs_gap": "|Gap|",
                    "bench_gap": "基准|Gap|",
                    "planning_delta": "计划偏差",
                    "dominant_factor": "主导因子",
                    "confidence": "置信度",
                }
            )
        st.dataframe(step_show, use_container_width=True)

        st.subheader(t["factor_table"])
        st.dataframe(sku_attr, use_container_width=True)

    st.download_button(t["download_diag"], to_csv_bytes(local_steps), t["file_diag"], "text/csv")

with tabs[3]:
    st.subheader(t["recs"])
    st.dataframe(filtered_recs, use_container_width=True)
    st.download_button(t["download_recs"], to_csv_bytes(filtered_recs), t["file_recs"], "text/csv")

    st.subheader(t["sim"])
    s1, s2 = st.columns(2)
    with s1:
        demand_shift = st.slider(t["demand_shift"], -30, 60, 10, 5)
    with s2:
        lead_shift = st.slider(t["lead_shift"], -30, 60, 10, 5)

    sim_input = filtered_metrics[
        ["date", "sku", "category", "warehouse", "on_hand_qty", "avg_daily_demand", "lead_time_days", "unit_cost"]
    ].copy()
    sim_input["avg_daily_demand"] = sim_input["avg_daily_demand"] * (1 + demand_shift / 100)
    sim_input["lead_time_days"] = np.clip(sim_input["lead_time_days"] * (1 + lead_shift / 100), 0, None)
    sim_metrics = abc_classification(build_inventory_metrics(sim_input))

    r1, r2, r3 = st.columns(3)
    r1.metric(t["risk_now"], int((filtered_metrics["coverage_gap"] < 0).sum()))
    r2.metric(t["risk_sim"], int((sim_metrics["coverage_gap"] < 0).sum()))
    r3.metric(t["var_sim"], f"${float(sim_metrics.loc[sim_metrics['coverage_gap'] < 0, 'inventory_value'].sum()):,.0f}")

    plan = sim_metrics.copy()
    plan["base_order_qty"] = np.maximum(plan["reorder_point"] - plan["on_hand_qty"], 0)
    class_weight = {"A": 1.2, "B": 1.0, "C": 0.8}
    uplift = 1 + max(target_service - 0.90, 0) * 2
    plan["recommended_order_qty"] = np.ceil(plan["base_order_qty"] * plan["abc_class"].map(class_weight).fillna(1.0) * uplift)
    plan["investment"] = plan["recommended_order_qty"] * plan["unit_cost"]
    plan["risk_score"] = np.maximum(-plan["coverage_gap"], 0) * plan["unit_cost"]
    plan = plan[plan["recommended_order_qty"] > 0].sort_values(["risk_score", "inventory_value"], ascending=[False, False])

    if plan.empty:
        st.success("No replenishment needed." if lang == "English" else "当前无需补货。")
    else:
        plan["cum_investment"] = plan["investment"].cumsum()
        selected_plan = plan[plan["cum_investment"] <= budget_cap].copy()
        st.subheader(t["plan_cap"])
        st.dataframe(
            selected_plan[["sku", "warehouse", "abc_class", "coverage_gap", "recommended_order_qty", "unit_cost", "investment", "risk_score"]],
            use_container_width=True,
        )
        st.download_button(t["download_plan"], to_csv_bytes(selected_plan), t["file_plan"], "text/csv")

    st.subheader(t["drill"])
    st.dataframe(local_metrics, use_container_width=True)


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
