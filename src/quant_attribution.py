from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial, sqrt

import numpy as np
import pandas as pd


@dataclass
class Outcome:
    shortage: float
    excess: float


FACTOR_ORDER = ["demand", "lead_time", "data_lag", "record_accuracy"]
FACTOR_DOMAIN = {
    "demand": "Demand",
    "lead_time": "Supply",
    "data_lag": "Process",
    "record_accuracy": "Process",
}


def _num(row: pd.Series, col: str, default: float = np.nan) -> float:
    if col not in row.index:
        return float(default)
    val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    if pd.isna(val):
        return float(default)
    return float(val)


def _factor_availability(row: pd.Series) -> dict[str, bool]:
    return {
        "demand": not np.isnan(_num(row, "avg_daily_demand")) and not np.isnan(_num(row, "forecast_daily_demand")),
        "lead_time": not np.isnan(_num(row, "lead_time_days")) and not np.isnan(_num(row, "avg_delay_days")),
        "data_lag": not np.isnan(_num(row, "data_lag_days")) and not np.isnan(_num(row, "avg_daily_demand")),
        "record_accuracy": not np.isnan(_num(row, "cycle_count_accuracy")),
    }


def _compute_outcome(row: pd.Series, use_actual: dict[str, bool], z: float, sigma_demand: float) -> Outcome:
    demand_actual = _num(row, "avg_daily_demand", 0.0)
    demand_plan = _num(row, "forecast_daily_demand", demand_actual)

    lead_plan = max(_num(row, "lead_time_days", 0.0), 0.0)
    avg_delay_days = max(_num(row, "avg_delay_days", 0.0), 0.0)
    lead_actual = lead_plan + avg_delay_days

    demand = demand_actual if use_actual.get("demand", False) else demand_plan
    lead_time = lead_actual if use_actual.get("lead_time", False) else lead_plan

    on_hand = _num(row, "on_hand_qty", 0.0)
    data_lag_days = max(_num(row, "data_lag_days", 0.0), 0.0) if use_actual.get("data_lag", False) else 0.0
    cycle_count_accuracy = _num(row, "cycle_count_accuracy", 1.0) if use_actual.get("record_accuracy", False) else 1.0
    cycle_count_accuracy = float(np.clip(cycle_count_accuracy, 0.0, 1.0))

    effective_on_hand = on_hand * cycle_count_accuracy - data_lag_days * max(demand, 0.0)

    lead_time_demand = max(demand, 0.0) * max(lead_time, 0.0)
    safety_stock = float(z) * max(float(sigma_demand), 0.0) * sqrt(max(lead_time, 0.0))
    target_stock = lead_time_demand + safety_stock

    net_gap = target_stock - effective_on_hand
    shortage = max(net_gap, 0.0)
    excess = max(-net_gap, 0.0)
    return Outcome(shortage=shortage, excess=excess)


def build_quantitative_attribution(metrics_df: pd.DataFrame, service_level_z: float = 1.65) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df is None or metrics_df.empty:
        empty_detail = pd.DataFrame(
            columns=[
                "sku",
                "warehouse",
                "view",
                "factor",
                "domain",
                "baseline_units",
                "actual_units",
                "total_change_units",
                "contribution_units",
                "contribution_share",
                "computable",
                "reason",
                "forecast_source",
            ]
        )
        empty_summary = pd.DataFrame(columns=["view", "domain", "contribution_units"])
        return empty_detail, empty_summary

    sigma_demand = float(metrics_df["avg_daily_demand"].std(ddof=0)) if "avg_daily_demand" in metrics_df.columns else 0.0
    if np.isnan(sigma_demand):
        sigma_demand = 0.0

    rows: list[dict[str, object]] = []

    for _, row in metrics_df.iterrows():
        sku = str(row.get("sku", ""))
        warehouse = str(row.get("warehouse", ""))
        availability = _factor_availability(row)
        active_factors = [f for f in FACTOR_ORDER if availability.get(f, False)]

        base_state = {f: False for f in FACTOR_ORDER}
        full_state = {f: (f in active_factors) for f in FACTOR_ORDER}

        baseline = _compute_outcome(row, base_state, service_level_z, sigma_demand)
        actual = _compute_outcome(row, full_state, service_level_z, sigma_demand)

        denom = factorial(len(active_factors)) if active_factors else 1
        shapley_shortage = {f: 0.0 for f in FACTOR_ORDER}
        shapley_excess = {f: 0.0 for f in FACTOR_ORDER}

        if active_factors:
            for perm in permutations(active_factors):
                state = {f: False for f in FACTOR_ORDER}
                prev = _compute_outcome(row, state, service_level_z, sigma_demand)
                for factor in perm:
                    state[factor] = True
                    nxt = _compute_outcome(row, state, service_level_z, sigma_demand)
                    shapley_shortage[factor] += nxt.shortage - prev.shortage
                    shapley_excess[factor] += nxt.excess - prev.excess
                    prev = nxt

            for factor in active_factors:
                shapley_shortage[factor] /= denom
                shapley_excess[factor] /= denom

        total_shortage_change = actual.shortage - baseline.shortage
        total_excess_change = actual.excess - baseline.excess

        reason_map = {
            "demand": "missing avg_daily_demand or forecast_daily_demand",
            "lead_time": "missing lead_time_days or avg_delay_days",
            "data_lag": "missing data_lag_days",
            "record_accuracy": "missing cycle_count_accuracy",
        }

        for factor in FACTOR_ORDER:
            for view, total_change, contrib in (
                ("shortage", total_shortage_change, shapley_shortage[factor]),
                ("excess", total_excess_change, shapley_excess[factor]),
            ):
                share = contrib / total_change if abs(total_change) > 1e-9 else 0.0
                rows.append(
                    {
                        "sku": sku,
                        "warehouse": warehouse,
                        "view": view,
                        "factor": factor,
                        "domain": FACTOR_DOMAIN[factor],
                        "baseline_units": baseline.shortage if view == "shortage" else baseline.excess,
                        "actual_units": actual.shortage if view == "shortage" else actual.excess,
                        "total_change_units": total_change,
                        "contribution_units": contrib,
                        "contribution_share": share,
                        "computable": availability.get(factor, False),
                        "reason": "" if availability.get(factor, False) else reason_map[factor],
                        "forecast_source": str(row.get("forecast_source", "missing")),
                    }
                )

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby(["view", "domain"], as_index=False)["contribution_units"]
        .sum()
        .sort_values(["view", "contribution_units"], ascending=[True, False])
    )
    return detail_df, summary_df
