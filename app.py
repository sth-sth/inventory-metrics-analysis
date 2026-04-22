from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts import AlertConfig, detect_alerts
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


def level_text(score: float, language: str) -> str:
    if score >= 0.75:
        return "高" if language == "中文" else "High"
    if score >= 0.45:
        return "中" if language == "中文" else "Medium"
    return "低" if language == "中文" else "Low"


def _factor_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("需求侧_") or c.startswith("供应侧_") or c.startswith("仓储侧_") or c.startswith("流程侧_")
    ]


def _en_factor_name(name: str) -> str:
    mapping = {
        "需求侧_预测不准": "demand_forecast_error",
        "需求侧_历史不足": "demand_insufficient_history",
        "需求侧_季节波动": "demand_seasonality",
        "需求侧_促销冲击": "demand_promotion_shock",
        "需求侧_突变风险": "demand_market_shift",
        "供应侧_供应商延迟": "supply_supplier_delay",
        "供应侧_交付不稳": "supply_delivery_instability",
        "供应侧_MOQ约束": "supply_moq_pressure",
        "供应侧_运输阻塞": "supply_transport_block",
        "仓储侧_安全库存设置": "warehouse_safety_stock_setting",
        "仓储侧_ROP设置": "warehouse_rop_setting",
        "仓储侧_阈值静态": "warehouse_static_threshold",
        "仓储侧_交期假设": "warehouse_leadtime_assumption",
        "流程侧_盘点规范": "process_cycle_counting",
        "流程侧_信息同步": "process_info_sync",
        "流程侧_审批速度": "process_approval_speed",
        "流程侧_数据滞后": "process_data_latency",
    }
    return mapping.get(name, name)


def render_calc_table(df: pd.DataFrame, key: str, language: str, col_help: dict[str, str] | None = None) -> None:
    if df is None or df.empty:
        st.dataframe(df, use_container_width=True)
        return

    event = None
    select_mode = "single-cell"
    try:
        event = st.dataframe(df, use_container_width=True, on_select="rerun", selection_mode="single-cell", key=key)
    except Exception:
        # Streamlit < compatible versions do not support single-cell selection.
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


def build_granular_diagnosis(metrics_df: pd.DataFrame, transactions_df: pd.DataFrame, language: str) -> pd.DataFrame:
    tx = transactions_df.copy()
    tx["date"] = pd.to_datetime(tx["date"])

    sales = tx[tx["event_type"] == "sale"].groupby(["sku", "warehouse"], as_index=False)["qty"].sum().rename(columns={"qty": "sales_qty"})
    receipts = tx[tx["event_type"] == "receipt"].groupby(["sku", "warehouse"], as_index=False)["qty"].sum().rename(columns={"qty": "receipt_qty"})
    delays = tx[tx["event_type"] == "receipt"].groupby(["sku", "warehouse"], as_index=False)["delay_days"].mean().rename(columns={"delay_days": "avg_delay_days"})
    delay_std = tx[tx["event_type"] == "receipt"].groupby(["sku", "warehouse"], as_index=False)["delay_days"].std(ddof=0).rename(columns={"delay_days": "delay_std"})
    day_count = tx.groupby(["sku", "warehouse"], as_index=False)["date"].nunique().rename(columns={"date": "history_days"})
    latest_tx = tx.groupby(["sku", "warehouse"], as_index=False)["date"].max().rename(columns={"date": "latest_tx_date"})

    base = metrics_df.copy()
    base["date"] = pd.to_datetime(base["date"])
    merged = (
        base.merge(sales, on=["sku", "warehouse"], how="left")
        .merge(receipts, on=["sku", "warehouse"], how="left")
        .merge(delays, on=["sku", "warehouse"], how="left")
        .merge(delay_std, on=["sku", "warehouse"], how="left")
        .merge(day_count, on=["sku", "warehouse"], how="left")
        .merge(latest_tx, on=["sku", "warehouse"], how="left")
    )

    merged[["sales_qty", "receipt_qty", "avg_delay_days", "delay_std", "history_days"]] = merged[[
        "sales_qty",
        "receipt_qty",
        "avg_delay_days",
        "delay_std",
        "history_days",
    ]].fillna(0)

    merged["latest_tx_date"] = pd.to_datetime(merged["latest_tx_date"], errors="coerce")
    merged["data_lag_days"] = (merged["date"] - merged["latest_tx_date"]).dt.days.fillna(0).clip(lower=0)

    defaults = {
        "forecast_daily_demand": merged["avg_daily_demand"],
        "seasonality_index": 1.0,
        "promo_lift_index": 1.0,
        "market_shock_index": 1.0,
        "moq_qty": 0.0,
        "in_transit_blocked_flag": 0.0,
        "cycle_count_accuracy": 0.98,
        "info_sync_delay_days": merged["data_lag_days"],
        "approval_delay_days": merged["avg_delay_days"],
    }
    for col, val in defaults.items():
        if col not in merged.columns:
            merged[col] = val

    for col in [
        "forecast_daily_demand",
        "seasonality_index",
        "promo_lift_index",
        "market_shock_index",
        "moq_qty",
        "in_transit_blocked_flag",
        "cycle_count_accuracy",
        "info_sync_delay_days",
        "approval_delay_days",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["forecast_daily_demand"] = merged["forecast_daily_demand"].fillna(merged["avg_daily_demand"])
    merged["seasonality_index"] = merged["seasonality_index"].fillna(1.0)
    merged["promo_lift_index"] = merged["promo_lift_index"].fillna(1.0)
    merged["market_shock_index"] = merged["market_shock_index"].fillna(1.0)
    merged["moq_qty"] = merged["moq_qty"].fillna(0.0)
    merged["in_transit_blocked_flag"] = merged["in_transit_blocked_flag"].fillna(0.0)
    merged["cycle_count_accuracy"] = merged["cycle_count_accuracy"].fillna(0.98)
    merged["info_sync_delay_days"] = merged["info_sync_delay_days"].fillna(merged["data_lag_days"])
    merged["approval_delay_days"] = merged["approval_delay_days"].fillna(merged["avg_delay_days"])

    forecast_den = merged["forecast_daily_demand"].replace(0, np.nan)
    forecast_error = ((merged["avg_daily_demand"] - merged["forecast_daily_demand"]).abs() / forecast_den).fillna(0.0)
    demand_cv = merged["avg_daily_demand"].std(ddof=0) / max(float(merged["avg_daily_demand"].mean()), 1e-9)

    merged["需求侧_预测不准"] = np.clip(forecast_error / 0.5, 0, 1)
    merged["需求侧_历史不足"] = np.where(merged["history_days"] < 7, 1.0, np.where(merged["history_days"] < 14, 0.6, 0.2))
    merged["需求侧_季节波动"] = np.clip((merged["seasonality_index"] - 1.0).abs() / 0.8, 0, 1)
    merged["需求侧_促销冲击"] = np.clip((merged["promo_lift_index"] - 1.0) / 1.0, 0, 1)
    merged["需求侧_突变风险"] = np.clip((merged["market_shock_index"] - 1.0) / 0.7, 0, 1)

    merged["供应侧_供应商延迟"] = np.clip(merged["avg_delay_days"] / 7.0, 0, 1)
    merged["供应侧_交付不稳"] = np.clip(merged["delay_std"] / 3.5, 0, 1)
    order_need = np.maximum(merged["reorder_point"] - merged["on_hand_qty"], 0)
    moq_den = merged["moq_qty"].replace(0, np.nan)
    merged["供应侧_MOQ约束"] = ((merged["moq_qty"] - order_need).clip(lower=0) / moq_den).fillna(0.0).clip(0, 1)
    merged["供应侧_运输阻塞"] = np.clip(merged["in_transit_blocked_flag"], 0, 1)

    ss_ratio = np.where(merged["lead_time_demand"] > 0, merged["safety_stock"] / merged["lead_time_demand"], 0)
    merged["仓储侧_安全库存设置"] = np.clip(np.abs(ss_ratio - 0.25) / 0.25, 0, 1)
    rop_den = merged["reorder_point"].replace(0, np.nan)
    merged["仓储侧_ROP设置"] = (merged["coverage_gap"].abs() / rop_den).fillna(0.0).clip(0, 1)
    merged["仓储侧_阈值静态"] = np.clip((demand_cv - 0.25) / 0.6, 0, 1)
    lt_den = merged["lead_time_days"].replace(0, np.nan)
    merged["仓储侧_交期假设"] = ((merged["avg_delay_days"] - merged["lead_time_days"]).abs() / lt_den).fillna(0.0).clip(0, 1)

    merged["流程侧_盘点规范"] = np.clip(1 - merged["cycle_count_accuracy"], 0, 1)
    merged["流程侧_信息同步"] = np.clip(merged["info_sync_delay_days"] / 5.0, 0, 1)
    merged["流程侧_审批速度"] = np.clip(merged["approval_delay_days"] / 7.0, 0, 1)
    merged["流程侧_数据滞后"] = np.clip(merged["data_lag_days"] / 5.0, 0, 1)

    demand_cols = [c for c in merged.columns if c.startswith("需求侧_")]
    supply_cols = [c for c in merged.columns if c.startswith("供应侧_")]
    wh_cols = [c for c in merged.columns if c.startswith("仓储侧_")]
    process_cols = [c for c in merged.columns if c.startswith("流程侧_")]
    factor_cols = demand_cols + supply_cols + wh_cols + process_cols

    merged["demand_domain_score"] = merged[demand_cols].mean(axis=1)
    merged["supply_domain_score"] = merged[supply_cols].mean(axis=1)
    merged["warehouse_domain_score"] = merged[wh_cols].mean(axis=1)
    merged["process_domain_score"] = merged[process_cols].mean(axis=1)
    merged["overall_score"] = merged[factor_cols].mean(axis=1)
    merged["overall_level"] = merged["overall_score"].apply(lambda x: level_text(x, language))
    return merged


def build_factor_summary(granular_df: pd.DataFrame, language: str) -> pd.DataFrame:
    factor_cols = _factor_cols(granular_df)
    if granular_df.empty or not factor_cols:
        return pd.DataFrame(columns=["factor", "impact_score"])

    out = pd.DataFrame({"factor": factor_cols, "impact_score": [float(granular_df[c].mean()) for c in factor_cols]})
    out = out.sort_values("impact_score", ascending=False)
    if language == "English":
        out["factor"] = out["factor"].map(_en_factor_name)
    return out


TXT = {
    "中文": {
        "lang_picker": "界面语言",
        "title": "库存智能决策云",
        "desc": "用于库存监控、风险归因、场景模拟和补货决策的可视化系统。",
        "panel": "控制面板",
        "source": "数据来源",
        "demo": "使用内置演示数据",
        "upload": "上传自定义 CSV",
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
        "need_upload": "请先上传两份 CSV 文件。",
        "load_fail": "数据加载失败",
        "demo_ok": "演示数据已加载。",
        "formula": "公式与解释（细粒度版）",
        "demo_block": "演示数据与 CSV 格式参考",
        "preview_inv": "库存演示数据预览",
        "preview_tx": "交易演示数据预览",
        "download_inv": "下载库存示例 CSV",
        "download_tx": "下载交易示例 CSV",
        "preview_plan": "计划值样例",
        "lineage": "数据来源与计算链路",
        "lineage_tip": "先看来源字段，再看计算过程，再看风险分形成方式。",
        "metric_dict": "指标说明",
        "abc_rule": "ABC 规则：按库存价值降序，累计占比 <=80% 为 A，80%-95% 为 B，其余为 C。",
        "kpi_value": "库存总价值",
        "kpi_so": "缺货风险 SKU",
        "kpi_os": "超储风险 SKU",
        "kpi_turn": "周转率代理",
        "kpi_svc": "服务水平代理",
        "tab1": "总览",
        "tab2": "监控",
        "tab3": "归因",
        "tab4": "决策",
        "granular_in_attr": "细粒度评分表",
        "granular_pick": "该 SKU 的细粒度因子评分",
        "granular_lineage": "点击查看细粒度因子来源与计算",
        "domain_scores": "四大域评分",
        "health": "库存健康分布",
        "abc": "ABC 库存价值占比",
        "risk_map": "SKU 风险图",
        "alerts": "报警中心",
        "diag_tip": "每个 SKU 展示四大域评分和细粒度因子分。",
        "pick_sku": "选择 SKU 查看详细归因",
        "trace": "点击查看该 SKU 计算链路",
        "trace_table": "计算链路明细",
        "delta_explain": "已删除旧的基准差值法，统一为细粒度评分法。",
        "domain_overview": "四大域总分",
        "top_factors": "高风险因子（按分值排序）",
        "factor_col": "因子",
        "score_col": "分值",
        "cell_tip": "点击表格任意单元格可查看悬浮卡片详情。",
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
        "lang_picker": "Interface Language",
        "title": "Inventory Intelligence Cloud",
        "desc": "Visual system for inventory monitoring, risk attribution, simulation, and replenishment decisions.",
        "panel": "Control Panel",
        "source": "Data Source",
        "demo": "Use built-in demo data",
        "upload": "Upload custom CSV",
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
        "need_upload": "Please upload both CSV files.",
        "load_fail": "Data loading failed",
        "demo_ok": "Demo data loaded.",
        "formula": "Formulas and Explanation (Granular)",
        "demo_block": "Demo Data and CSV Format Reference",
        "preview_inv": "Inventory demo preview",
        "preview_tx": "Transactions demo preview",
        "download_inv": "Download inventory sample CSV",
        "download_tx": "Download transactions sample CSV",
        "preview_plan": "Planned values sample",
        "lineage": "Data Source and Calculation Lineage",
        "lineage_tip": "Check source fields first, then calculations, then score aggregation.",
        "metric_dict": "Metric Dictionary",
        "abc_rule": "ABC rule: sort by inventory value desc; cumulative share <=80% is A, 80%-95% is B, remaining is C.",
        "kpi_value": "Total Inventory Value",
        "kpi_so": "Stockout Risk SKU",
        "kpi_os": "Overstock Risk SKU",
        "kpi_turn": "Turnover Proxy",
        "kpi_svc": "Service Level Proxy",
        "tab1": "Overview",
        "tab2": "Monitoring",
        "tab3": "Attribution",
        "tab4": "Actions",
        "granular_in_attr": "Granular Score Table",
        "granular_pick": "Granular factor scores for selected SKU",
        "granular_lineage": "Click to view granular factor source and formula",
        "domain_scores": "Domain Scores",
        "health": "Inventory Health Distribution",
        "abc": "Inventory Value by ABC",
        "risk_map": "SKU Risk Map",
        "alerts": "Alert Center",
        "diag_tip": "Each SKU shows domain scores and granular factor scores.",
        "pick_sku": "Select SKU for detailed attribution",
        "trace": "Click to view SKU calculation trace",
        "trace_table": "Calculation trace details",
        "delta_explain": "Legacy benchmark-delta method has been removed. Granular scoring is now the only method.",
        "domain_overview": "Domain Score Overview",
        "top_factors": "Top Risk Factors (sorted by score)",
        "factor_col": "factor",
        "score_col": "score",
        "cell_tip": "Click any table cell to open a floating detail card.",
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
    },
}


with st.sidebar:
    language = st.selectbox("界面语言" if st.session_state.get("lang", "中文") == "中文" else "Interface Language", ["中文", "English"], index=0)
    st.session_state["lang"] = language
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
    demo_plan = abc_classification(build_inventory_metrics(demo_inv))
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

with st.expander(t["formula"], expanded=False):
    if language == "中文":
        st.markdown(
            """
### 1) 基础库存指标

- 库存覆盖天数 = 当前库存数量 / 日均需求数量
- 交期需求量 = 日均需求数量 × 补货交期天数
- 安全库存 = z × 需求标准差 × sqrt(补货交期天数)
- 再订货点 = 交期需求量 + 安全库存

### 2) 细粒度因子评分（0-1）

- 需求侧：预测不准分 = |实际需求 - 预测需求| / 预测需求，再归一化到 0-1
- 供应侧：供应商延迟分 = 平均延迟天 / 7，再归一化到 0-1
- 仓储侧：ROP 设置分 = |Gap| / 再订货点，再归一化到 0-1
- 流程侧：信息同步分 = 信息同步延迟天 / 5，再归一化到 0-1

### 3) 域评分与总评分

- 域评分 = 该域全部因子分的平均值
- 总评分 = 四大域全部因子分的平均值
"""
        )
    else:
        st.markdown(
            """
### 1) Base Inventory Metrics

- Days of Inventory = On-hand Quantity / Average Daily Demand
- Lead-time Demand = Average Daily Demand × Lead-time Days
- Safety Stock = z × demand std × sqrt(lead-time days)
- Reorder Point = Lead-time Demand + Safety Stock

### 2) Granular Factor Scoring (0-1)

- Demand: forecast error = |actual - forecast| / forecast, normalized to 0-1
- Supply: supplier delay = avg_delay_days / 7, normalized to 0-1
- Warehouse: ROP setting = |Gap| / reorder_point, normalized to 0-1
- Process: info sync = info_sync_delay_days / 5, normalized to 0-1

### 3) Domain and Overall Scores

- Domain score = mean of all factors in the domain
- Overall score = mean of all factors across four domains
"""
        )

with st.expander(t["lineage"], expanded=False):
    st.caption(t["lineage_tip"])
    if language == "中文":
        st.markdown(
            """
| 步骤 | 输入来源 | 输出字段 | 说明 |
|---|---|---|---|
| 库存基础数据 | inventory.csv | on_hand_qty, avg_daily_demand, lead_time_days, unit_cost | 原始库存和需求数据 |
| 交易数据 | transactions.csv | event_type, qty, delay_days | 销售/入库/延迟信息 |
| 计划值计算 | 库存基础数据 | lead_time_demand, safety_stock, reorder_point | 形成计划值 |
| 细粒度因子计算 | 实际值 + 计划值 + 交易行为 | 需求/供应/仓储/流程因子分 | 每个因子归一化到 0-1 |
| 风险评分聚合 | 各因子分 | domain_score 与 overall_score | 域均值与总均值 |
"""
        )
    else:
        st.markdown(
            """
| Step | Source Input | Output Fields | Description |
|---|---|---|---|
| Inventory base | inventory.csv | on_hand_qty, avg_daily_demand, lead_time_days, unit_cost | Raw stock and demand data |
| Transactions | transactions.csv | event_type, qty, delay_days | Sales/receipt/delay events |
| Planned value | Inventory base | lead_time_demand, safety_stock, reorder_point | Planned values |
| Granular factor calculation | Actual + planned + transaction behavior | demand/supply/warehouse/process factor scores | Each factor is normalized to 0-1 |
| Risk aggregation | Factor scores | domain_score and overall_score | Domain mean and overall mean |
"""
        )

with st.expander(t["metric_dict"], expanded=False):
    if language == "中文":
        st.markdown(
            """
- 库存总价值：on_hand_qty * unit_cost
- 缺货风险：coverage_gap < 阈值
- 超储风险：库存覆盖天数 > 阈值
- 周转率代理：销售总量 / 平均库存数量
- 服务水平代理：coverage_gap >= 0 的 SKU 占比
- 需求域评分：需求侧因子均值
- 供应域评分：供应侧因子均值
- 仓储域评分：仓储侧因子均值
- 流程域评分：流程侧因子均值
- 总评分：四大域全部因子均值
- ABC：按库存价值降序，累计占比 <=80% 为 A，80%-95% 为 B，其余为 C
"""
        )
    else:
        st.markdown(
            """
- Total inventory value: on_hand_qty * unit_cost
- Stockout risk: coverage_gap < threshold
- Overstock risk: days_of_inventory > threshold
- Turnover proxy: total_sales_qty / average_inventory_qty
- Service-level proxy: share of SKUs where coverage_gap >= 0
- Demand domain score: mean of demand factors
- Supply domain score: mean of supply factors
- Warehouse domain score: mean of warehouse factors
- Process domain score: mean of process factors
- Overall score: mean of all factors across four domains
- ABC: cumulative value <=80% A, 80%-95% B, >95% C
"""
        )
    st.info(t["abc_rule"])

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
granular_df = build_granular_diagnosis(filtered_metrics, filtered_tx, language)
factor_summary_df = build_factor_summary(granular_df, language)
recs_df = generate_recommendations(filtered_metrics, factor_summary_df, language)
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
    render_calc_table(filtered_alerts, key="alerts_table", language=language)

with tab3:
    st.caption(t["diag_tip"])
    st.info(t["delta_explain"])
    st.caption(t["cell_tip"])

    if granular_df.empty:
        st.info("当前筛选下暂无可用归因数据。" if language == "中文" else "No attribution data under current filters.")
    else:
        sku = st.selectbox(t["pick_sku"], sorted(granular_df["sku"].unique().tolist()))
        g = granular_df[granular_df["sku"] == sku].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("需求域" if language == "中文" else "Demand", f"{float(g['demand_domain_score']):.2f}")
        c2.metric("供应域" if language == "中文" else "Supply", f"{float(g['supply_domain_score']):.2f}")
        c3.metric("仓储域" if language == "中文" else "Warehouse", f"{float(g['warehouse_domain_score']):.2f}")
        c4.metric("流程域" if language == "中文" else "Process", f"{float(g['process_domain_score']):.2f}")

        domain_scores = pd.DataFrame(
            [
                {"domain": "需求" if language == "中文" else "demand", "score": float(g["demand_domain_score"])},
                {"domain": "供应" if language == "中文" else "supply", "score": float(g["supply_domain_score"])},
                {"domain": "仓储" if language == "中文" else "warehouse", "score": float(g["warehouse_domain_score"])},
                {"domain": "流程" if language == "中文" else "process", "score": float(g["process_domain_score"])},
            ]
        )
        st.subheader(t["domain_overview"])
        st.plotly_chart(px.bar(domain_scores, x="domain", y="score", title=t["domain_scores"]), use_container_width=True)
        render_calc_table(domain_scores, key="domain_scores_table", language=language, col_help={"score": "域内因子均值" if language == "中文" else "Mean of factors in this domain"})

        factor_rows = []
        for c in _factor_cols(granular_df):
            factor_rows.append({
                t["factor_col"]: c if language == "中文" else _en_factor_name(c),
                t["score_col"]: float(g[c]),
            })
        factor_table = pd.DataFrame(factor_rows).sort_values(t["score_col"], ascending=False)

        st.subheader(t["top_factors"])
        st.plotly_chart(px.bar(factor_table.head(10), x=t["factor_col"], y=t["score_col"], title=t["top_factors"]), use_container_width=True)
        render_calc_table(
            factor_table,
            key="factor_table",
            language=language,
            col_help={
                t["score_col"]: "因子归一化风险分，范围 0-1。" if language == "中文" else "Normalized factor score in range 0-1.",
            },
        )

        with st.expander(t["trace"], expanded=False):
            m = filtered_metrics[filtered_metrics["sku"] == sku].iloc[0]
            trace_rows = pd.DataFrame(
                [
                    {
                        "metric": "交期需求量" if language == "中文" else "lead_time_demand",
                        "input": f"avg_daily_demand={m['avg_daily_demand']:.2f}, lead_time_days={m['lead_time_days']:.2f}",
                        "formula": "avg_daily_demand * lead_time_days",
                        "result": float(m["lead_time_demand"]),
                    },
                    {
                        "metric": "安全库存" if language == "中文" else "safety_stock",
                        "input": f"z=1.65, demand_std(global), lead_time_days={m['lead_time_days']:.2f}",
                        "formula": "z * std(demand) * sqrt(lead_time_days)",
                        "result": float(m["safety_stock"]),
                    },
                    {
                        "metric": "再订货点" if language == "中文" else "reorder_point",
                        "input": f"lead_time_demand={m['lead_time_demand']:.2f}, safety_stock={m['safety_stock']:.2f}",
                        "formula": "lead_time_demand + safety_stock",
                        "result": float(m["reorder_point"]),
                    },
                    {
                        "metric": "Gap",
                        "input": f"on_hand_qty={m['on_hand_qty']:.2f}, reorder_point={m['reorder_point']:.2f}",
                        "formula": "on_hand_qty - reorder_point",
                        "result": float(m["coverage_gap"]),
                    },
                ]
            )
            st.subheader(t["trace_table"])
            render_calc_table(trace_rows, key="trace_table", language=language)

        st.subheader(t["granular_in_attr"])
        granular_cols = [
            "sku",
            "warehouse",
            "overall_score",
            "overall_level",
            "demand_domain_score",
            "supply_domain_score",
            "warehouse_domain_score",
            "process_domain_score",
        ] + _factor_cols(granular_df)
        sku_granular = granular_df[granular_df["sku"] == sku][granular_cols].copy()
        if language == "English":
            sku_granular = sku_granular.rename(columns={c: _en_factor_name(c) for c in _factor_cols(granular_df)})

        st.caption(t["granular_pick"])
        render_calc_table(sku_granular, key="sku_granular_table", language=language)

with tab4:
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
        render_calc_table(selected_plan, key="selected_plan_table", language=language)

    st.subheader(t["drill"])
    render_calc_table(filtered_metrics, key="drill_table", language=language)
