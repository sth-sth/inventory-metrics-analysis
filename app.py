from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts import AlertConfig, detect_alerts
from src.attribution import build_issue_breakdown
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
.cell-card {
    border: 1px solid rgba(14,165,164,0.35);
    background: rgba(255,255,255,0.92);
    border-radius: 12px;
    padding: 10px 12px;
    margin-top: 8px;
    box-shadow: 0 6px 20px rgba(15,39,64,0.10);
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


def render_calc_table(df: pd.DataFrame, key: str, language: str, col_help: dict[str, str] | None = None) -> None:
    if df is None or df.empty:
        st.dataframe(df, use_container_width=True)
        return

    event = None
    select_mode = "single-cell"
    try:
        event = st.dataframe(df, use_container_width=True, on_select="rerun", selection_mode="single-cell", key=key)
    except Exception:
        select_mode = "single-row"
        event = st.dataframe(df, use_container_width=True, on_select="rerun", selection_mode="single-row", key=key)
        st.caption(
            "当前环境不支持单元格直接点击，已切换为行选择+列选择详情。"
            if language == "中文"
            else "Cell selection is not supported in this Streamlit version. Switched to row + column detail mode."
        )

    if event is None or not hasattr(event, "selection"):
        return

    if select_mode == "single-cell":
        cells = event.selection.get("cells", [])
        if not cells:
            return
        cell = cells[0]
        row_idx = int(cell.get("row", 0))
        col_ref = cell.get("column")
        if isinstance(col_ref, int):
            if col_ref < 0 or col_ref >= len(df.columns):
                return
            col_name = str(df.columns[col_ref])
        else:
            col_name = str(col_ref)
            if col_name not in df.columns:
                return
    else:
        rows = event.selection.get("rows", [])
        if not rows:
            return
        row_idx = int(rows[0])
        col_name = st.selectbox(
            "选择列查看详情" if language == "中文" else "Pick a column for details",
            options=[str(c) for c in df.columns],
            key=f"{key}_fallback_col",
        )

    value = df.iloc[row_idx][col_name]
    row_label = f"第 {row_idx + 1} 行" if language == "中文" else f"Row {row_idx + 1}"
    col_label = "列" if language == "中文" else "Column"
    val_label = "值" if language == "中文" else "Value"
    explain_label = "说明" if language == "中文" else "Explanation"
    default_note = "该单元格为当前计算结果。" if language == "中文" else "This cell is a current computed result."
    note = (col_help or {}).get(col_name, default_note)

    st.markdown(
        f"""
<div class=\"cell-card\">
  <b>{row_label}</b><br/>
  {col_label}: {col_name}<br/>
  {val_label}: {value}<br/>
  {explain_label}: {note}
</div>
""",
        unsafe_allow_html=True,
    )


def render_formula_button_table(label: str, table_df: pd.DataFrame, key: str, language: str) -> None:
    def _render_chain_card(df: pd.DataFrame) -> None:
        title = "计算说明" if language == "中文" else "Calculation Notes"
        lines = []
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            lines.append(" | ".join(parts))
        body = "<br/>".join(lines) if lines else ("无数据" if language == "中文" else "No data")
        st.markdown(
            f"""
<div class=\"cell-card\">
  <b>{title}</b><br/>
  {body}
</div>
""",
            unsafe_allow_html=True,
        )

    try:
        with st.popover(label):
            _render_chain_card(table_df)
    except Exception:
        if st.button(label, key=f"{key}_btn"):
            st.session_state[f"{key}_show"] = not st.session_state.get(f"{key}_show", False)
        if st.session_state.get(f"{key}_show", False):
            _render_chain_card(table_df)


def render_selected_card(title: str, rows: list[dict[str, object]], language: str) -> None:
    body_lines = []
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items()]
        body_lines.append(" | ".join(parts))
    body = "<br/>".join(body_lines) if body_lines else ("无数据" if language == "中文" else "No data")
    st.markdown(
        f"""
<div class=\"cell-card\">
  <b>{title}</b><br/>
  {body}
</div>
""",
        unsafe_allow_html=True,
    )


def enrich_metrics_with_tx_signals(metrics_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df.copy()

    out = metrics_df.copy()
    tx = transactions_df.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

    receipt_delay = (
        tx[tx["event_type"] == "receipt"]
        .groupby(["sku", "warehouse"], as_index=False)["delay_days"]
        .mean()
        .rename(columns={"delay_days": "avg_delay_days"})
    )
    latest_tx = tx.groupby(["sku", "warehouse"], as_index=False)["date"].max().rename(columns={"date": "latest_tx_date"})

    out = out.merge(receipt_delay, on=["sku", "warehouse"], how="left")
    out = out.merge(latest_tx, on=["sku", "warehouse"], how="left")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["latest_tx_date"] = pd.to_datetime(out["latest_tx_date"], errors="coerce")
    out["data_lag_days"] = (out["date"] - out["latest_tx_date"]).dt.days
    return out


def build_issue_inventory_table(
    metrics_df: pd.DataFrame,
    language: str,
    stockout_threshold: float,
    overstock_doh: float,
    delay_days: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, metric_row in metrics_df.iterrows():
        issue_df = build_issue_breakdown(metric_row, language, stockout_threshold, overstock_doh, delay_days)
        rows.append(issue_df)
    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    abs_col = "偏差绝对值" if language == "中文" else "absolute_gap"
    if abs_col in out.columns:
        out = out.sort_values(abs_col, ascending=False, na_position="last")
    return out


def build_kpi_formula_chains(filtered_metrics: pd.DataFrame, kpis: dict, language: str) -> dict[str, pd.DataFrame]:
    sku_count = int(len(filtered_metrics))
    if language == "中文":
        return {
            "kpi_in_stock": pd.DataFrame(
                [
                    {"步骤": "统计满足覆盖条件的 SKU", "说明": "coverage_gap >= 0 视为现货可覆盖。"},
                    {"步骤": "除以 SKU 总数", "说明": f"现货率 = 满足条件 SKU / {sku_count}，结果 {kpis['in_stock_rate']:.1%}"},
                ]
            ),
            "kpi_turnover": pd.DataFrame(
                [
                    {"步骤": "计算销货成本（COGS）", "说明": f"销售数量 * 单位成本，汇总得到 ${kpis['cogs_value']:,.0f}"},
                    {"步骤": "计算平均存货", "说明": f"当前样本平均存货价值 ${kpis['avg_inventory_value']:,.0f}"},
                    {"步骤": "周转率", "说明": f"周转率 = COGS / 平均存货 = {kpis['inventory_turnover']:.2f}"},
                ]
            ),
            "kpi_turnover_days": pd.DataFrame(
                [
                    {"步骤": "从周转率换算", "说明": "周转天数 = 365 / 周转率。"},
                    {"步骤": "结果", "说明": f"当前周转天数 {kpis['turnover_days']:.1f} 天"},
                ]
            ),
            "kpi_slow_moving": pd.DataFrame(
                [
                    {"步骤": "识别慢销/滞销 SKU", "说明": f"销售为 0 或 DOH >= {kpis['slow_moving_doh_threshold']:.0f} 视为滞销。"},
                    {"步骤": "计算占比", "说明": f"滞销率 = 滞销 SKU / 总 SKU，结果 {kpis['slow_moving_rate']:.1%}"},
                ]
            ),
            "kpi_inventory_cost": pd.DataFrame(
                [
                    {"步骤": "库存总价值", "说明": f"当前库存总价值 ${kpis['total_inventory_value']:,.0f}"},
                    {"步骤": "乘以库存资金占用率", "说明": f"库存成本 = 总价值 * {kpis['carrying_cost_rate']:.0%} = ${kpis['inventory_cost']:,.0f}"},
                ]
            ),
        }

    return {
        "kpi_in_stock": pd.DataFrame(
            [
                {"step": "covered SKUs", "note": "Treat coverage_gap >= 0 as in-stock coverage."},
                {"step": "divide by all SKUs", "note": f"In-stock rate = covered SKUs / {sku_count}, result {kpis['in_stock_rate']:.1%}."},
            ]
        ),
        "kpi_turnover": pd.DataFrame(
            [
                {"step": "compute COGS", "note": f"Sales quantity * unit cost, aggregated COGS = ${kpis['cogs_value']:,.0f}."},
                {"step": "average inventory", "note": f"Average inventory value = ${kpis['avg_inventory_value']:,.0f}."},
                {"step": "turnover", "note": f"Turnover = COGS / avg inventory = {kpis['inventory_turnover']:.2f}."},
            ]
        ),
        "kpi_turnover_days": pd.DataFrame(
            [
                {"step": "convert from turnover", "note": "Turnover days = 365 / turnover."},
                {"step": "result", "note": f"Current turnover days = {kpis['turnover_days']:.1f} days."},
            ]
        ),
        "kpi_slow_moving": pd.DataFrame(
            [
                {"step": "identify slow-moving SKUs", "note": f"No sales or DOH >= {kpis['slow_moving_doh_threshold']:.0f} is treated as slow-moving."},
                {"step": "compute rate", "note": f"Slow-moving rate = slow-moving SKUs / all SKUs = {kpis['slow_moving_rate']:.1%}."},
            ]
        ),
        "kpi_inventory_cost": pd.DataFrame(
            [
                {"step": "inventory value", "note": f"Current inventory value = ${kpis['total_inventory_value']:,.0f}."},
                {"step": "carrying cost", "note": f"Inventory cost = value * {kpis['carrying_cost_rate']:.0%} = ${kpis['inventory_cost']:,.0f}."},
            ]
        ),
    }


def build_kpi_warning_table(kpis: dict, language: str, turnover_min: float, slow_moving_max: float, inventory_cost_max: float) -> pd.DataFrame:
    if language == "中文":
        rows = [
            {
                "指标": "现货率",
                "当前值": f"{kpis['in_stock_rate']:.1%}",
                "阈值": "建议 >= 95%",
                "状态": "预警" if kpis["in_stock_rate"] < 0.95 else "正常",
                "说明": "现货率低说明缺货风险上升。",
            },
            {
                "指标": "周转率",
                "当前值": f"{kpis['inventory_turnover']:.2f}",
                "阈值": f"建议 >= {turnover_min:.2f}",
                "状态": "预警" if kpis["inventory_turnover"] < turnover_min else "正常",
                "说明": "周转率低说明库存周转慢。",
            },
            {
                "指标": "周转天数",
                "当前值": f"{kpis['turnover_days']:.1f}",
                "阈值": "建议越低越好",
                "状态": "预警" if kpis["turnover_days"] > 365 / max(turnover_min, 1e-9) else "正常",
                "说明": "周转天数高说明库存占压时间长。",
            },
            {
                "指标": "滞销率",
                "当前值": f"{kpis['slow_moving_rate']:.1%}",
                "阈值": f"建议 <= {slow_moving_max:.1%}",
                "状态": "预警" if kpis["slow_moving_rate"] > slow_moving_max else "正常",
                "说明": "滞销率高说明无销或慢销 SKU 偏多。",
            },
            {
                "指标": "库存成本",
                "当前值": f"${kpis['inventory_cost']:,.0f}",
                "阈值": f"预算 <= ${inventory_cost_max:,.0f}",
                "状态": "预警" if kpis["inventory_cost"] > inventory_cost_max else "正常",
                "说明": "库存成本超预算会直接影响现金流。",
            },
        ]
        return pd.DataFrame(rows)

    rows = [
        {
            "metric": "In-stock rate",
            "current": f"{kpis['in_stock_rate']:.1%}",
            "threshold": ">= 95%",
            "status": "Warning" if kpis["in_stock_rate"] < 0.95 else "OK",
            "note": "Low in-stock rate indicates rising stockout risk.",
        },
        {
            "metric": "Inventory turnover",
            "current": f"{kpis['inventory_turnover']:.2f}",
            "threshold": f">= {turnover_min:.2f}",
            "status": "Warning" if kpis["inventory_turnover"] < turnover_min else "OK",
            "note": "Low turnover means inventory is moving slowly.",
        },
        {
            "metric": "Turnover days",
            "current": f"{kpis['turnover_days']:.1f}",
            "threshold": "Lower is better",
            "status": "Warning" if kpis["turnover_days"] > 365 / max(turnover_min, 1e-9) else "OK",
            "note": "Higher days indicate longer capital tie-up.",
        },
        {
            "metric": "Slow-moving rate",
            "current": f"{kpis['slow_moving_rate']:.1%}",
            "threshold": f"<= {slow_moving_max:.1%}",
            "status": "Warning" if kpis["slow_moving_rate"] > slow_moving_max else "OK",
            "note": "High rate means many no-sale or slow-sale SKUs.",
        },
        {
            "metric": "Inventory cost",
            "current": f"${kpis['inventory_cost']:,.0f}",
            "threshold": f"<= ${inventory_cost_max:,.0f}",
            "status": "Warning" if kpis["inventory_cost"] > inventory_cost_max else "OK",
            "note": "Over-budget inventory cost impacts cash flow.",
        },
    ]
    return pd.DataFrame(rows)


def build_issue_recommendations(issue_df: pd.DataFrame, language: str) -> pd.DataFrame:
    if issue_df.empty:
        return pd.DataFrame(columns=["分类", "问题", "建议"]) if language == "中文" else pd.DataFrame(columns=["domain", "issue", "action"])

    if language == "中文":
        watch = issue_df[issue_df.get("是否超阈值", "").isin(["是", "人工判断"])].copy()
        if watch.empty:
            watch = issue_df.copy()
        out = watch[["分类", "问题", "差了多少", "建议看什么"]].drop_duplicates().rename(columns={"建议看什么": "建议"})
        return out.reset_index(drop=True)

    watch = issue_df[issue_df.get("out_of_bound", "").isin(["yes", "manual review"])].copy()
    if watch.empty:
        watch = issue_df.copy()
    out = watch[["domain", "issue", "gap", "what_to_check"]].drop_duplicates().rename(columns={"what_to_check": "action"})
    return out.reset_index(drop=True)


TXT = {
    "中文": {
        "lang_picker": "界面语言",
        "title": "库存智能决策云",
        "desc": "用于库存监控、归因识别、场景模拟和补货决策的可视化系统。",
        "panel": "控制面板",
        "source": "数据来源",
        "source_hint": "支持一键运行演示数据或上传真实业务 CSV",
        "demo": "使用内置演示数据（直接运行）",
        "upload": "上传真实业务 CSV",
        "inv_csv": "库存快照 CSV",
        "tx_csv": "交易流水 CSV",
        "threshold": "阈值参数",
        "stockout": "缺货阈值（Gap < 阈值）",
        "overstock": "超储阈值（库存覆盖天数 > 阈值）",
        "delay": "延迟阈值（到货延迟天数 > 阈值）",
        "stockout_help": "Gap 低于阈值时触发缺货风险。",
        "overstock_help": "库存覆盖天数过高时触发超储风险。",
        "delay_help": "收货延迟超过阈值时触发供应延迟报警。",
        "decision": "决策参数",
        "budget": "补货预算上限",
        "service": "目标服务水平",
        "need_upload": "请上传真实业务的两份 CSV 文件后再分析。",
        "load_fail": "数据加载失败",
        "demo_ok": "演示数据已加载并开始分析。",
        "demo_block": "演示数据预览与下载",
        "preview_inv": "库存演示数据预览",
        "preview_tx": "交易演示数据预览",
        "download_inv": "下载库存演示 CSV",
        "download_tx": "下载交易演示 CSV",
        "preview_plan": "演示数据计算后预览",
        "kpi_in_stock": "现货率",
        "kpi_turnover": "周转率（COGS/平均存货）",
        "kpi_turnover_days": "周转天数",
        "kpi_slow_moving": "滞销率",
        "kpi_inventory_cost": "库存成本",
        "tab1": "总览",
        "tab2": "监控",
        "tab3": "归因",
        "tab4": "决策",
        "health": "库存健康分布",
        "abc": "ABC 库存价值占比",
        "risk_map": "SKU 风险图",
        "alerts": "报警中心",
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
        "kpi_formula_btn": "查看计算说明",
        "sim_formula_btn": "模拟说明",
    },
    "English": {
        "lang_picker": "Interface Language",
        "title": "Inventory Intelligence Cloud",
        "desc": "Visual system for inventory monitoring, issue identification, simulation, and replenishment decisions.",
        "panel": "Control Panel",
        "source": "Data Source",
        "source_hint": "Run built-in demo instantly or upload real business CSVs",
        "demo": "Use built-in demo data (run now)",
        "upload": "Upload real business CSVs",
        "inv_csv": "Inventory snapshot CSV",
        "tx_csv": "Transactions CSV",
        "threshold": "Threshold Settings",
        "stockout": "Stockout threshold (Gap < threshold)",
        "overstock": "Overstock threshold (Days of Inventory > threshold)",
        "delay": "Delay threshold (Delay days > threshold)",
        "stockout_help": "Trigger stockout risk when Gap is below threshold.",
        "overstock_help": "Trigger overstock risk when inventory coverage days are too high.",
        "delay_help": "Trigger supplier delay alert when receipt delay exceeds threshold.",
        "decision": "Decision Settings",
        "budget": "Replenishment budget cap",
        "service": "Target service level",
        "need_upload": "Upload both real business CSV files before analysis.",
        "load_fail": "Data loading failed",
        "demo_ok": "Demo data loaded and analysis started.",
        "demo_block": "Demo Data Preview and Download",
        "preview_inv": "Inventory demo preview",
        "preview_tx": "Transactions demo preview",
        "download_inv": "Download inventory demo CSV",
        "download_tx": "Download transactions demo CSV",
        "preview_plan": "Computed preview on demo data",
        "kpi_in_stock": "In-stock Rate",
        "kpi_turnover": "Inventory Turnover (COGS/Avg Inventory)",
        "kpi_turnover_days": "Turnover Days",
        "kpi_slow_moving": "Slow-moving Rate",
        "kpi_inventory_cost": "Inventory Cost",
        "tab1": "Overview",
        "tab2": "Monitoring",
        "tab3": "Attribution",
        "tab4": "Actions",
        "health": "Inventory Health Distribution",
        "abc": "Inventory Value by ABC",
        "risk_map": "SKU Risk Map",
        "alerts": "Alert Center",
        "recs": "Business Recommendations",
        "sim": "Scenario Simulation and Replenishment Priority",
        "demand_shift": "Demand shift (%)",
        "lead_shift": "Lead-time shift (%)",
        "risk_now": "Current stockout risk",
        "risk_sim": "Simulated stockout risk",
        "var_sim": "Simulated value at risk",
        "plan": "Budget-constrained replenishment list",
        "drill": "Operational Drill-down",
        "no_data": "No data after filtering. Adjust your filters.",
        "wh_filter": "Filter by warehouse",
        "cat_filter": "Filter by category",
        "kpi_formula_btn": "View Calculation Notes",
        "sim_formula_btn": "Simulation Notes",
    },
}


with st.sidebar:
    language = st.selectbox(
        "界面语言" if st.session_state.get("lang", "中文") == "中文" else "Interface Language",
        ["中文", "English"],
        index=0,
    )
    st.session_state["lang"] = language
    t = TXT[language]

    st.header(t["panel"])
    st.subheader(t["source"])
    st.caption(t["source_hint"])
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
    carrying_cost_rate = st.slider(
        "库存资金占用率" if language == "中文" else "Inventory carrying cost rate",
        min_value=0.05,
        max_value=0.40,
        value=0.20,
        step=0.01,
    )
    slow_moving_doh_threshold = st.number_input(
        "滞销判定 DOH 阈值" if language == "中文" else "Slow-moving DOH threshold",
        min_value=30.0,
        value=180.0,
        step=10.0,
    )
    turnover_min_threshold = st.number_input(
        "周转率预警下限" if language == "中文" else "Turnover warning lower bound",
        min_value=0.1,
        value=2.0,
        step=0.1,
    )
    slow_moving_max_threshold = st.slider(
        "滞销率预警上限" if language == "中文" else "Slow-moving rate warning upper bound",
        min_value=0.05,
        max_value=0.90,
        value=0.30,
        step=0.01,
    )
    inventory_cost_budget = st.number_input(
        "库存成本预算上限" if language == "中文" else "Inventory cost budget cap",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
    )


st.markdown(
    f"""
<div class=\"hero\">
  <h1>{t['title']}</h1>
  <p>{t['desc']}</p>
</div>
""",
    unsafe_allow_html=True,
)

inventory_file = None
transactions_file = None

if source_mode == t["upload"]:
    c_up_1, c_up_2 = st.columns(2)
    with c_up_1:
        inventory_file = st.file_uploader(t["inv_csv"], type=["csv"])
    with c_up_2:
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
    demo_plan = abc_classification(
        build_inventory_metrics(
            demo_inv,
            stockout_gap_threshold=stockout_threshold,
            overstock_doh_threshold=overstock_doh,
        )
    )
    c1, c2 = st.columns(2)
    with c1:
        st.write(t["preview_inv"])
        render_calc_table(demo_inv.head(10), key="demo_inv_table", language=language)
        st.download_button(t["download_inv"], to_csv_bytes(demo_inv), "inventory_demo.csv", "text/csv")
    with c2:
        st.write(t["preview_tx"])
        render_calc_table(demo_tx.head(10), key="demo_tx_table", language=language)
        st.download_button(t["download_tx"], to_csv_bytes(demo_tx), "transactions_demo.csv", "text/csv")

    st.write(t["preview_plan"])
    render_calc_table(
        demo_plan[[
            "sku",
            "warehouse",
            "on_hand_qty",
            "avg_daily_demand",
            "lead_time_days",
            "lead_time_demand",
            "safety_stock",
            "reorder_point",
            "coverage_gap",
            "abc_class",
        ]].head(10),
        key="demo_plan_table",
        language=language,
    )

metrics_df = abc_classification(
    build_inventory_metrics(
        inventory_df,
        stockout_gap_threshold=stockout_threshold,
        overstock_doh_threshold=overstock_doh,
    )
)
alerts_df = detect_alerts(
    metrics_df,
    transactions_df,
    AlertConfig(
        stockout_gap_threshold=stockout_threshold,
        overstock_doh_threshold=overstock_doh,
        delayed_receipt_days=delay_days,
    ),
)

f1, f2 = st.columns(2)
with f1:
    wh_options = sorted(metrics_df["warehouse"].unique().tolist())
    selected_wh = st.multiselect(t["wh_filter"], options=wh_options, default=wh_options)
with f2:
    cat_options = sorted(metrics_df["category"].unique().tolist())
    selected_cat = st.multiselect(t["cat_filter"], options=cat_options, default=cat_options)

filtered_metrics = metrics_df[metrics_df["warehouse"].isin(selected_wh) & metrics_df["category"].isin(selected_cat)].copy()
if filtered_metrics.empty:
    st.warning(t["no_data"])
    st.stop()

filtered_alerts = alerts_df[alerts_df["warehouse"].isin(selected_wh)] if not alerts_df.empty else alerts_df
filtered_tx = transactions_df[transactions_df["warehouse"].isin(selected_wh)].copy()
filtered_metrics_for_attr = enrich_metrics_with_tx_signals(filtered_metrics, filtered_tx)

kpis = build_kpi_summary(
    filtered_metrics,
    transactions_df[transactions_df["warehouse"].isin(selected_wh)],
    slow_moving_doh_threshold=slow_moving_doh_threshold,
    carrying_cost_rate=carrying_cost_rate,
)
kpi_formula_chains = build_kpi_formula_chains(filtered_metrics=filtered_metrics, kpis=kpis, language=language)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(t["kpi_in_stock"], f"{kpis['in_stock_rate']:.1%}")
    render_formula_button_table(t["kpi_formula_btn"], kpi_formula_chains["kpi_in_stock"], key="kpi_in_stock_formula", language=language)
with k2:
    st.metric(t["kpi_turnover"], f"{kpis['inventory_turnover']:.2f}")
    render_formula_button_table(t["kpi_formula_btn"], kpi_formula_chains["kpi_turnover"], key="kpi_turnover_formula", language=language)
with k3:
    st.metric(t["kpi_turnover_days"], f"{kpis['turnover_days']:.1f}")
    render_formula_button_table(t["kpi_formula_btn"], kpi_formula_chains["kpi_turnover_days"], key="kpi_turnover_days_formula", language=language)
with k4:
    st.metric(t["kpi_slow_moving"], f"{kpis['slow_moving_rate']:.1%}")
    render_formula_button_table(t["kpi_formula_btn"], kpi_formula_chains["kpi_slow_moving"], key="kpi_slow_moving_formula", language=language)
with k5:
    st.metric(t["kpi_inventory_cost"], f"${kpis['inventory_cost']:,.0f}")
    render_formula_button_table(t["kpi_formula_btn"], kpi_formula_chains["kpi_inventory_cost"], key="kpi_inventory_cost_formula", language=language)

kpi_warning_df = build_kpi_warning_table(
    kpis=kpis,
    language=language,
    turnover_min=turnover_min_threshold,
    slow_moving_max=slow_moving_max_threshold,
    inventory_cost_max=inventory_cost_budget,
)
st.subheader("KPI 预警" if language == "中文" else "KPI Warnings")
render_calc_table(kpi_warning_df, key="kpi_warning_table", language=language)

st.markdown("### " + ("点图看计算" if language == "中文" else "Click the chart to see the calculation"))
st.caption("图上的柱子或点，会直接展开底层计算说明。" if language == "中文" else "Click any bar or point on the charts to expand the underlying calculation notes.")

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
    render_calc_table(filtered_alerts, key="alerts_table", language=language)

with tab3:
    st.caption(
        "先选 SKU，再按四个侧筛选；同一 SKU 若同时出现缺货与超储表观信号，会在业务核对里提示口径复核。"
        if language == "中文"
        else "Pick a SKU first, then filter by the four domains; if the same SKU shows both stockout and overstock signals, the checklist will flag a logic review."
    )

    issue_inventory_df = build_issue_inventory_table(
        metrics_df=filtered_metrics_for_attr,
        language=language,
        stockout_threshold=stockout_threshold,
        overstock_doh=overstock_doh,
        delay_days=delay_days,
    )

    if issue_inventory_df.empty:
        st.info("当前筛选下暂无可用归因数据。" if language == "中文" else "No attribution data under current filters.")
    else:
        if language == "中文":
            sku_options = sorted(issue_inventory_df["sku"].dropna().astype(str).unique().tolist()) if "sku" in issue_inventory_df.columns else []
            selected_sku = st.selectbox(t["pick_sku"], options=["全部SKU"] + sku_options, index=0, key="sku_issue_selector")
        else:
            sku_options = sorted(issue_inventory_df["sku"].dropna().astype(str).unique().tolist()) if "sku" in issue_inventory_df.columns else []
            selected_sku = st.selectbox(t["pick_sku"], options=["All SKUs"] + sku_options, index=0, key="sku_issue_selector")

        if selected_sku in {"全部SKU", "All SKUs"}:
            sku_issue_df = issue_inventory_df.copy()
        else:
            sku_issue_df = issue_inventory_df[issue_inventory_df["sku"].astype(str) == selected_sku].copy() if "sku" in issue_inventory_df.columns else issue_inventory_df.copy()

        checklist_df = build_business_checklist(sku_issue_df, language)
        st.subheader("业务核对清单" if language == "中文" else "Business checklist")
        render_calc_table(checklist_df, key="business_checklist_table", language=language)

        if language == "中文":
            domain_options = ["需求侧", "供应侧", "仓储侧", "流程侧"]
            selected_domains = st.multiselect("按分类筛选", options=domain_options, default=domain_options)
            issue_view_df = sku_issue_df[sku_issue_df["分类"].isin(selected_domains)].copy() if "分类" in sku_issue_df.columns else sku_issue_df
        else:
            domain_options = ["Demand", "Supply", "Warehouse", "Process"]
            selected_domains = st.multiselect("Filter by domain", options=domain_options, default=domain_options)
            issue_view_df = sku_issue_df[sku_issue_df["domain"].isin(selected_domains)].copy() if "domain" in sku_issue_df.columns else sku_issue_df

        st.markdown("#### " + ("四侧问题识别清单" if language == "中文" else "Four-domain Issue Inventory"))
        render_calc_table(issue_view_df, key="all_issue_inventory_table", language=language)

        issue_rec_df = build_issue_recommendations(issue_view_df, language)
        st.subheader("自动化建议" if language == "中文" else "Automated Suggestions")
        render_calc_table(issue_rec_df, key="issue_recommendation_table", language=language)

        if not issue_view_df.empty:
            main_issue = issue_view_df.iloc[0]
            if language == "中文":
                card_rows = [
                    {"分类": main_issue.get("分类", ""), "SKU": main_issue.get("sku", ""), "仓库": main_issue.get("warehouse", "")},
                    {"问题": main_issue.get("问题", ""), "当前值": main_issue.get("当前值", ""), "差了多少": main_issue.get("差了多少", "")},
                    {"是否超阈值": main_issue.get("是否超阈值", ""), "怎么算": main_issue.get("怎么算", "")},
                    {"建议": main_issue.get("建议看什么", "")},
                ]
            else:
                card_rows = [
                    {"domain": main_issue.get("domain", ""), "SKU": main_issue.get("sku", ""), "warehouse": main_issue.get("warehouse", "")},
                    {"issue": main_issue.get("issue", ""), "current": main_issue.get("current", ""), "gap": main_issue.get("gap", "")},
                    {"out_of_bound": main_issue.get("out_of_bound", ""), "how": main_issue.get("how", "")},
                    {"action": main_issue.get("what_to_check", "")},
                ]
            render_selected_card("已选问题" if language == "中文" else "Selected issue", card_rows, language)

with tab4:
    recs_df = generate_recommendations(filtered_metrics, language)
    filtered_recs = recs_df[recs_df["sku"].isin(filtered_metrics["sku"]) | (recs_df["sku"] == "ALL")]
    st.subheader(t["recs"])
    render_calc_table(filtered_recs, key="recs_table", language=language)

    st.subheader(t["sim"])
    s1, s2 = st.columns(2)
    with s1:
        demand_shift = st.slider(t["demand_shift"], -30, 60, 10, 5)
    with s2:
        lead_shift = st.slider(t["lead_shift"], -30, 60, 10, 5)

    sim_input = filtered_metrics[["date", "sku", "category", "warehouse", "on_hand_qty", "avg_daily_demand", "lead_time_days", "unit_cost"]].copy()
    sim_input["avg_daily_demand"] = sim_input["avg_daily_demand"] * (1 + demand_shift / 100)
    sim_input["lead_time_days"] = np.clip(sim_input["lead_time_days"] * (1 + lead_shift / 100), 0, None)

    sim_metrics = abc_classification(
        build_inventory_metrics(
            sim_input,
            stockout_gap_threshold=stockout_threshold,
            overstock_doh_threshold=overstock_doh,
        )
    )

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
        render_calc_table(selected_plan, key="selected_plan_table", language=language)

    st.subheader(t["drill"])
    render_calc_table(filtered_metrics, key="drill_table", language=language)
