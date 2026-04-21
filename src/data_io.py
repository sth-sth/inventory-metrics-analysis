from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_INVENTORY_COLUMNS = {
    "date",
    "sku",
    "category",
    "warehouse",
    "on_hand_qty",
    "avg_daily_demand",
    "lead_time_days",
    "unit_cost",
}

REQUIRED_TRANSACTIONS_COLUMNS = {
    "date",
    "sku",
    "warehouse",
    "event_type",
    "qty",
    "delay_days",
    "supplier",
}


@dataclass
class DatasetBundle:
    inventory: pd.DataFrame
    transactions: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def _parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        raise ValueError(f"Column '{col}' contains invalid dates")
    return df


def load_inventory_csv(file_obj) -> pd.DataFrame:
    inv = pd.read_csv(file_obj)
    inv = _normalize_columns(inv)
    _validate_columns(inv, REQUIRED_INVENTORY_COLUMNS, "inventory")
    inv = _parse_dates(inv, "date")

    numeric_cols = ["on_hand_qty", "avg_daily_demand", "lead_time_days", "unit_cost"]
    for col in numeric_cols:
        inv[col] = pd.to_numeric(inv[col], errors="coerce")
    if inv[numeric_cols].isna().any().any():
        raise ValueError("inventory contains non-numeric values in numeric fields")

    if (inv["on_hand_qty"] < 0).any() or (inv["avg_daily_demand"] < 0).any() or (inv["lead_time_days"] < 0).any():
        raise ValueError("inventory has negative values where not allowed")

    return inv


def load_transactions_csv(file_obj) -> pd.DataFrame:
    tx = pd.read_csv(file_obj)
    tx = _normalize_columns(tx)
    _validate_columns(tx, REQUIRED_TRANSACTIONS_COLUMNS, "transactions")
    tx = _parse_dates(tx, "date")

    tx["qty"] = pd.to_numeric(tx["qty"], errors="coerce")
    tx["delay_days"] = pd.to_numeric(tx["delay_days"], errors="coerce")
    if tx[["qty", "delay_days"]].isna().any().any():
        raise ValueError("transactions contains invalid numeric fields")

    allowed = {"sale", "receipt", "adjustment"}
    if not tx["event_type"].isin(allowed).all():
        raise ValueError("transactions.event_type must be one of: sale, receipt, adjustment")

    return tx


def load_bundle(inventory_file_obj, transactions_file_obj) -> DatasetBundle:
    inventory = load_inventory_csv(inventory_file_obj)
    transactions = load_transactions_csv(transactions_file_obj)
    return DatasetBundle(inventory=inventory, transactions=transactions)
