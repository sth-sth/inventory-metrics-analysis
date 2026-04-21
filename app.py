from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
html, body, [class*="css"] {
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
    color: #fff;
    margin-bottom: 16px;
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


TXT = {
    "中文": {
        "title": "库存智能决策云",
        "desc": "用于库存监控、风险归因、场景模拟和补货决策的可视化系统。",
        "panel": "控制面板",
        "source": "数据来源",
        "demo": "使用内置演示数据",
        "upload": "上传自定义 CSV",
        "inv_csv": "库存快照 CSV",
        "tx_csv": "交易流水 CSV",
        "threshold": "阈值参数说明",
        "stockout": "缺货阈值（Gap < 阈值）",
        "overstock": "超储阈值（库存覆盖天数 > 阈值）",
        "delay": "延迟阈值（到货延迟天数 > 阈值）",
        "stockout_help": "当 Gap（当前库存减去再订货点）低于该值时触发缺货风险。",
        "overstock_help": "当库存覆盖天数过高时触发超储风险。",
        "delay_help": "当收货延迟天数超过该值时触发供应延迟报警。",
        "decision": "决策参数",
        "budget": "补货预算上限",
        "service": "目标服务水平",
        "need_upload": "请先上传两份 CSV 文件。",
        "load_fail": "数据加载失败",
        "demo_ok": "演示数据已加载，可直接用于展示。",
        "formula": "公式与解释（易懂版）",
        "demo_block": "演示数据与 CSV 格式参考",
        "preview_inv": "库存演示数据预览（前 10 行）",
        "preview_tx": "交易演示数据预览（前 10 行）",
        "download_inv": "下载库存示例 CSV",
        "download_tx": "下载交易示例 CSV",
        "kpi_value": "库存总价值",
        "kpi_so": "缺货风险 SKU",
        "kpi_os": "超储风险 SKU",
        "kpi_turn": "周转率代理",
        "kpi_svc": "服务水平代理",
        "tab1": "总览",
        "tab2": "监控",
        "tab3": "归因",
        "tab4": "决策",
        "health": "库存健康分布",
        "abc": "ABC 库存价值占比",
        "risk_map": "SKU 风险图（覆盖天数 vs Gap）",
        "alerts": "报警中心",
        "diag_tip": "每个 SKU 都会展示在“需求、延迟、计划”三个步骤上，相对基准多了多少/少了多少。",
        "pick_sku": "选择 SKU 查看详细归因",
        "step_delta": "步骤偏差（正数=高于基准，负数=低于基准）",
        "step_table": "步骤明细",
        "factor_table": "因子贡献",
        "recs": "业务建议",
        "sim": "场景模拟与补货优先级",
        "demand_shift": "需求变化（%）",
        "lead_shift": "交期变化（%）",
        "risk_now": "当前缺货风险数",
        "risk_sim": "模拟缺货风险数",
        "var_sim": "模拟风险价值",
        "plan": "预算内补货清单",
        "drill": "运营下钻明细",
        "no_data": "筛选后无数据，请调整筛选条件。",
        "wh_filter": "按仓库筛选",
        "cat_filter": "按品类筛选",
    },
    "English": {
        "title": "Inventory Intelligence Cloud",
        "desc": "A visual system for inventory monitoring, risk attribution, simulation, and replenishment decisions.",
        "panel": "Control Panel",
        "source": "Data Source",
        "demo": "Use built-in demo data",
        "upload": "Upload custom CSV",
        "inv_csv": "Inventory snapshot CSV",
        "tx_csv": "Transactions CSV",
        "threshold": "Threshold Guidance",
        "stockout": "Stockout threshold (Gap < threshold)",
        "overstock": "Overstock threshold (Days of Inventory > threshold)",
        "delay": "Delay threshold (Delay days > threshold)",
        "stockout_help": "Stockout alert triggers when Gap (on-hand minus reorder point) falls below this value.",
        "overstock_help": "Overstock alert triggers when days of inventory are too high.",
        "delay_help": "Supplier delay alert triggers when receipt delay exceeds this value.",
        "decision": "Decision Settings",
        "budget": "Replenishment budget cap",
        "service": "Target service level",
        "need_upload": "Please upload both CSV files first.",
        "load_fail": "Data loading failed",
        "demo_ok": "Demo data loaded and ready for presentation.",
        "formula": "Formulas and Explanation (Easy Version)",
        "demo_block": "Demo Data and CSV Format Reference",
        "preview_inv": "Inventory demo preview (top 10)",
        "preview_tx": "Transactions demo preview (top 10)",
        "download_inv": "Download inventory sample CSV",
        "download_tx": "Download transactions sample CSV",
        "kpi_value": "Total Inventory Value",
        "kpi_so": "Stockout Risk SKU",
        "kpi_os": "Overstock Risk SKU",
        "kpi_turn": "Turnover Proxy",
        "kpi_svc": "Service Level Proxy",
        "tab1": "Overview",
        "tab2": "Monitoring",
        "tab3": "Attribution",
        "tab4": "Actions",
        "health": "Inventory Health Distribution",
        "abc": "Inventory Value by ABC Class",
        "risk_map": "SKU Risk Map (DOI vs Gap)",
        "alerts": "Alert Center",
        "diag_tip": "Each SKU is decomposed into three steps: demand, delay, and planning. You can see how much each is above/below benchmark.",
        "pick_sku": "Select SKU for detailed attribution",
        "step_delta": "Step Delta (positive=above benchmark, negative=below)",
        "step_table": "Step Breakdown",
        "factor_table": "Factor Contribution",
        "recs": "Business Recommendations",
        "sim": "Scenario Simulation and Replenishment Priority",
        "demand_shift": "Demand shift (%)",
        "lead_shift": "Lead-time shift (%)",
        "risk_now": "Current stockout risk",
        "risk_sim": "Simulated stockout risk",
        "var_sim": "Simulated value at risk",
        "plan": "Budget-constrained replenishment list",
        "drill": "Operational Drill-down",
        "no_data": "No data after filtering, please adjust filters.",
        "wh_filter": "Filter by warehouse",
        "cat_filter": "Filter by category",
    },
}


with st.sidebar:
    language = st.selectbox("界面语言 / Interface", ["中文", "English"], index=0)
    t = TXT[language]

    st.header(t["panel"])
    source_mode = st.radio(t["source"], [t["demo"], t["upload"]], index=0)

    st.markdown("---")
    st.subheader(t["threshold"])
    stockout_threshold = st.number_input(t["stockout"], value=0.0, step=1.0, help=t["stockout_help"])
    overstock_doh = st.number_input(t["overstock"], value=120.0, step=5.0, help=t["overstock_help"])
    delay_days = st.number_input(t["delay"], value=3.0, step=1.0, help=t["delay_help"])

    st.markdown("---")
    st.subheader(t["decision"])
    budget_cap = st.number_input(t["budget"], min_value=0.0, value=15000.0, step=1000.0)
    target_service = st.slider(t["service"], min_value=0.80, max_value=0.99, value=0.95, step=0.01)


st.markdown(
    f"""
<div class="hero">
  <h1>{t['title']}</h1>
  <p>{t['desc']}</p>
</div>
""",
    unsafe_allow_html=True,
)

inventory_file = None
transactions_file = None

if source_mode == t["upload"]:
    up1, up2 = st.columns(2)
    with up1:
        inventory_file = st.file_uploader(t["inv_csv"], type=["csv"])
    with up2:
        transactions_file = st.file_uploader(t["tx_csv"], type=["csv"])

if source_mode == t["demo"]:
    inventory_df, transactions_df = load_demo_data()
    st.success(t["demo_ok"])
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

with st.expander(t["demo_block"], expanded=False):
    demo_inv, demo_tx = load_demo_data()
    c1, c2 = st.columns(2)
    with c1:
        st.write(t["preview_inv"])
        st.dataframe(demo_inv.head(10), use_container_width=True)
        st.download_button(t["download_inv"], to_csv_bytes(demo_inv), "inventory_demo.csv", "text/csv")
    with c2:
        st.write(t["preview_tx"])
        st.dataframe(demo_tx.head(10), use_container_width=True)
        st.download_button(t["download_tx"], to_csv_bytes(demo_tx), "transactions_demo.csv", "text/csv")

with st.expander(t["formula"], expanded=False):
    if language == "中文":
        st.markdown(
            r"""
### 1) 库存覆盖天数（Days of Inventory）

$$
\text{库存覆盖天数} = \frac{\text{当前库存数量}}{\text{日均需求数量}}
$$

含义：当前库存大约还能支持多少天销售。

### 2) 交期需求量（Lead-time Demand）

$$
\text{交期需求量} = \text{日均需求数量} \times \text{补货交期天数}
$$

含义：在补货到达前，预计会消耗多少库存。

### 3) 安全库存（Safety Stock）

$$
\text{安全库存} = z \times \text{需求波动标准差} \times \sqrt{\text{补货交期天数}}
$$

含义：用于防止波动导致断货的缓冲库存。

### 4) 再订货点（Reorder Point）

$$
\text{再订货点} = \text{交期需求量} + \text{安全库存}
$$

### 5) Gap（库存缺口）

$$
\text{Gap} = \text{当前库存数量} - \text{再订货点}
$$

解释：
- Gap < 0：低于补货安全线，缺货风险更高。
- Gap > 0：高于补货安全线，库存更安全。
"""
        )
    else:
        st.markdown(
            r"""
### 1) Days of Inventory

$$
\text{Days of Inventory} = \frac{\text{On-hand Quantity}}{\text{Average Daily Demand}}
$$

Meaning: how many days current stock can support.

### 2) Lead-time Demand

$$
\text{Lead-time Demand} = \text{Average Daily Demand} \times \text{Lead-time Days}
$$

Meaning: expected consumption before replenishment arrives.

### 3) Safety Stock

$$
\text{Safety Stock} = z \times \text{Demand Std Dev} \times \sqrt{\text{Lead-time Days}}
$$

Meaning: buffer stock against demand and lead-time uncertainty.

### 4) Reorder Point

$$
\text{Reorder Point} = \text{Lead-time Demand} + \text{Safety Stock}
$$

### 5) Gap

$$
\text{Gap} = \text{On-hand Quantity} - \text{Reorder Point}
$$

Interpretation:
- Gap < 0: below safety line, higher stockout risk.
- Gap > 0: above safety line, safer inventory position.
"""
        )

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
    selected_wh = st.multiselect(t["wh_filter"], options=wh_options, default=wh_options)
with f2:
    cat_options = sorted(metrics_df["category"].unique().tolist())
    selected_cat = st.multiselect(t["cat_filter"], options=cat_options, default=cat_options)

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

kpis = build_kpi_summary(filtered_metrics, transactions_df[transactions_df["warehouse"].isin(selected_wh)])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(t["kpi_value"], f"${kpis['total_inventory_value']:,.0f}")
k2.metric(t["kpi_so"], f"{kpis['stockout_risk_sku_count']}")
k3.metric(t["kpi_os"], f"{kpis['overstock_risk_sku_count']}")
k4.metric(t["kpi_turn"], f"{kpis['inventory_turnover_proxy']:.2f}")
k5.metric(t["kpi_svc"], f"{kpis['service_level_proxy']:.1%}")

tab1, tab2, tab3, tab4 = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"]])

with tab1:
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(filtered_metrics, x="inventory_health", color="inventory_health", title=t["health"]), use_container_width=True)
    with b:
        st.plotly_chart(px.pie(filtered_metrics, names="abc_class", values="inventory_value", title=t["abc"]), use_container_width=True)

with tab2:
    fig = px.scatter(
        filtered_metrics,
        x="doh",
        y="coverage_gap",
        color="inventory_health",
        size="inventory_value",
        hover_data=["sku", "warehouse", "category", "abc_class"],
        title=t["risk_map"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(t["alerts"])
    st.dataframe(filtered_alerts, use_container_width=True)

with tab3:
    st.caption(t["diag_tip"])
    if filtered_steps.empty:
        st.info("当前阈值下没有缺货风险 SKU。" if language == "中文" else "No stockout-risk SKU under current thresholds.")
    else:
        sku_options = sorted(filtered_steps["sku"].unique().tolist())
        sku = st.selectbox(t["pick_sku"], sku_options)
        sku_step = filtered_steps[filtered_steps["sku"] == sku].copy()
        sku_attr = filtered_attr[filtered_attr["sku"] == sku].copy()
        row = sku_step.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("需求偏差" if language == "中文" else "Demand Delta", f"{row['demand_delta']:+.2f}")
        c2.metric("延迟偏差" if language == "中文" else "Delay Delta", f"{row['delay_delta']:+.2f}")
        c3.metric("计划偏差" if language == "中文" else "Planning Delta", f"{row['planning_delta']:+.2f}")

        chart_df = pd.DataFrame(
            {
                ("步骤" if language == "中文" else "Step"): [
                    "需求偏差" if language == "中文" else "Demand Delta",
                    "延迟偏差" if language == "中文" else "Delay Delta",
                    "计划偏差" if language == "中文" else "Planning Delta",
                ],
                ("偏差值" if language == "中文" else "Delta"): [
                    float(row["demand_delta"]),
                    float(row["delay_delta"]),
                    float(row["planning_delta"]),
                ],
            }
        )
        st.plotly_chart(px.bar(chart_df, x=chart_df.columns[0], y=chart_df.columns[1], title=t["step_delta"]), use_container_width=True)

        st.subheader(t["step_table"])
        st.dataframe(sku_step, use_container_width=True)

        st.subheader(t["factor_table"])
        st.dataframe(sku_attr, use_container_width=True)

with tab4:
    st.subheader(t["recs"])
    st.dataframe(filtered_recs, use_container_width=True)

    st.subheader(t["sim"])
    s1, s2 = st.columns(2)
    with s1:
        demand_shift = st.slider(t["demand_shift"], -30, 60, 10, 5)
    with s2:
        lead_shift = st.slider(t["lead_shift"], -30, 60, 10, 5)

    sim_input = filtered_metrics[["date", "sku", "category", "warehouse", "on_hand_qty", "avg_daily_demand", "lead_time_days", "unit_cost"]].copy()
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
        st.success("当前无需补货。" if language == "中文" else "No replenishment needed.")
    else:
        plan["cum_investment"] = plan["investment"].cumsum()
        selected_plan = plan[plan["cum_investment"] <= budget_cap].copy()
        st.subheader(t["plan"])
        st.dataframe(selected_plan, use_container_width=True)

    st.subheader(t["drill"])
    st.dataframe(filtered_metrics, use_container_width=True)
