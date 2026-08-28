"""
Order-picking duration analysis — data processing pipeline.

The source workbook ("Отчет сборка мечта.xlsx") is a line-level export from a
warehouse assembly/picking system. Each row is one picked line, scanned on a
hand-held terminal at time `ВремяСборки`, inside an order document.

Two picking regimes are mixed in the data:
  * LIVE picking  — the operator walks to the location, retrieves the item and
                    scans it. Consecutive scans are separated by irregular,
                    travel-dominated gaps.
  * MASS scanning — items were collected manually *off-system* first, then the
                    barcodes are scanned in one rapid burst at a station. Gaps
                    are small and regular and do NOT reflect picking effort.

This module loads the workbook, reconstructs each operator's continuous scan
timeline, classifies every line as LIVE or BULK (mass-scan), and derives a
per-line picking duration from the backward inter-scan gap.

Run `python pipeline.py` to write the processed dataset to
`output/picks_processed.csv`.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

SRC = "Отчет сборка мечта.xlsx"
OUTDIR = Path("output")

# --- Tunable thresholds (justified in ANALYSIS.md) --------------------------
BURST_GAP_S = 30      # consecutive scans <= this are "rapid" (candidate burst)
BURST_MIN_RUN = 5     # a run of this many rapid scans => mass-scan (BULK)
DUP_GAP_S = 3         # gaps below this are duplicate / simultaneous scans
BREAK_CAP_S = 1800    # gaps above this (30 min) are breaks/shift gaps, not a pick

COLS = ["order", "ord_start", "ord_end", "item", "sku",
        "serial", "qty", "scan", "author"]


def load_raw(path: str = SRC) -> pd.DataFrame:
    """Load the workbook and normalise column names / dtypes."""
    df = pd.read_excel(path, sheet_name="TDSheet", dtype=str)
    df.columns = COLS
    for c in ("ord_start", "ord_end", "scan"):
        df[c] = pd.to_datetime(df[c], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["is_serial"] = df["serial"].eq("Да")
    df["category"] = _category(df["item"])
    return df


def _category(names: pd.Series) -> pd.Series:
    """Coarse item category = leading noun of the item name (Russian)."""
    # First 1-2 words capture the product class ("Стиральная машина",
    # "Телефон сотовый", "Монитор", ...). Two-word classes are collapsed.
    two = {"Стиральная машина", "Сушильный аппарат", "Телефон сотовый",
           "Стартовый пакет", "Морозильный ларь", "Робот пылесос",
           "Кухонная плита", "Вытяжка кухонная"}
    def cat(n: str) -> str:
        n = (n or "").strip()
        toks = n.split()
        if len(toks) >= 2 and " ".join(toks[:2]) in two:
            return " ".join(toks[:2])
        return toks[0] if toks else "—"
    return names.map(cat)


def build_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by operator+time and derive the backward inter-scan gap.

    Each operator works a continuous timeline, moving from order to order, so
    the gap is computed within `author` (not within order): the time since the
    operator's previous scan is the time spent reaching and picking this line.
    """
    df = df.sort_values(["author", "scan"]).reset_index(drop=True)
    df["gap_s"] = df.groupby("author")["scan"].diff().dt.total_seconds()
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Flag mass-scan (BULK) lines and compute a valid live picking duration.

    BULK: a maximal run of >= BURST_MIN_RUN consecutive scans (same operator)
          each separated by <= BURST_GAP_S seconds. Such runs are the tell-tale
          of pre-collected items barcoded in one pass.
    A "run boundary" opens whenever the gap is missing (operator's first scan)
    or exceeds BURST_GAP_S.
    """
    df = df.copy()
    boundary = df["gap_s"].isna() | (df["gap_s"] > BURST_GAP_S)
    df["run_id"] = boundary.groupby(df["author"]).cumsum()
    run_len = df.groupby(["author", "run_id"])["scan"].transform("size")
    df["run_len"] = run_len
    df["is_bulk"] = run_len >= BURST_MIN_RUN

    # Picking duration: only meaningful for LIVE lines with a real, bounded gap.
    valid = (~df["is_bulk"]) & df["gap_s"].notna() \
            & (df["gap_s"] >= DUP_GAP_S) & (df["gap_s"] <= BREAK_CAP_S)
    df["duration_s"] = np.where(valid, df["gap_s"], np.nan)
    df["is_live_valid"] = valid

    # Reason a line has no live duration (for transparency / auditing).
    reason = np.full(len(df), "live", dtype=object)
    reason[df["is_bulk"].values] = "bulk_massscan"
    reason[(~df["is_bulk"]) & df["gap_s"].isna()] = "session_start"
    reason[(~df["is_bulk"]) & (df["gap_s"] < DUP_GAP_S)] = "duplicate_scan"
    reason[(~df["is_bulk"]) & (df["gap_s"] > BREAK_CAP_S)] = "break_boundary"
    df["duration_reason"] = reason
    return df


def process(path: str = SRC) -> pd.DataFrame:
    return classify(build_timeline(load_raw(path)))


if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)
    d = process()
    keep = ["order", "author", "scan", "item", "category", "sku", "is_serial",
            "qty", "gap_s", "run_len", "is_bulk", "duration_s",
            "is_live_valid", "duration_reason"]
    d[keep].to_csv(OUTDIR / "picks_processed.csv", index=False)
    print(f"rows={len(d)}  live_valid={int(d['is_live_valid'].sum())}  "
          f"bulk={int(d['is_bulk'].sum())}")
    print(f"wrote {OUTDIR/'picks_processed.csv'}")
