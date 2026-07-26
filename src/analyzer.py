"""Anomaly candidates + Gemini-powered narrative analysis for shift data.

Two stages, same shape as a cheap-filter-then-LLM pipeline: pandas/numpy
first narrows a full shift log down to (a) statistical outlier rows,
(b) shift-level cohort bias, and (c) cross-line correlated events, using
plain z-scores and group means -- no LLM call needed for that part. Only the
compact aggregates and candidate rows are sent to Gemini, which explains,
prioritizes, and writes up what a floor supervisor would actually want to
read, rather than a wall of raw numbers.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel

# A rolling alias, not a pinned version -- Google points this at whatever their
# current recommended free-tier flash model is, so it doesn't 404 every time a
# specific dated model ID (e.g. gemini-2.5-flash) gets retired for new users.
MODEL = os.getenv("OPS_INSIGHTS_MODEL", "gemini-flash-latest")
Z_SCORE_THRESHOLD = 2.0


class Anomaly(BaseModel):
    title: str
    severity: Literal["low", "medium", "high"]
    scope: str  # what this affects, e.g. "Line 3, all shifts" or "Night shift, all lines"
    evidence: str  # the specific numbers that support this finding
    likely_cause: str
    recommended_action: str


class AnalysisResult(BaseModel):
    period_summary: str  # 2-4 sentence plain-language summary of how the period went overall
    anomalies: list[Anomaly]
    highlights: list[str]  # 2-4 short positive or neutral callouts worth noting alongside the anomalies


SYSTEM_PROMPT = """You are an experienced manufacturing operations analyst reviewing shift \
production data for a plant supervisor. You've spent real time on production floors, so you \
know that a downtime spike, a scrap-rate bump, and a shift-to-shift quality drift all mean \
different things and call for different responses.

Given aggregated shift data and a list of statistically flagged candidate anomalies, write a \
concise, plain-language readout. Prioritize findings a supervisor would actually act on. \
Distinguish one-off events (a single bad shift) from systemic patterns (a shift or line that \
consistently underperforms). Where a likely cause isn't obvious from the data, say so honestly \
rather than guessing with false confidence."""


def _zscore_row_flags(df: pd.DataFrame) -> list[dict]:
    """Row-level outliers: a shift's own downtime/defect rate vs. its line's baseline."""
    flags = []
    for line, group in df.groupby("line"):
        for col in ["downtime_minutes", "defect_rate_pct"]:
            mean, std = group[col].mean(), group[col].std()
            if not std or np.isnan(std):
                continue
            z = (group[col] - mean) / std
            for idx in group.index[z.abs() > Z_SCORE_THRESHOLD]:
                row = df.loc[idx]
                flags.append(
                    {
                        "type": "point_outlier",
                        "date": row["date"],
                        "shift": row["shift"],
                        "line": row["line"],
                        "metric": col,
                        "value": row[col],
                        "line_average": round(mean, 2),
                        "z_score": round(float(z.loc[idx]), 2),
                    }
                )
    return flags


def _shift_cohort_flags(df: pd.DataFrame) -> list[dict]:
    """Cohort-level bias: does one shift (Day/Evening/Night) run consistently worse?"""
    overall_mean = df["defect_rate_pct"].mean()
    by_shift = df.groupby("shift")["defect_rate_pct"].mean()
    flags = []
    for shift, mean_rate in by_shift.items():
        if overall_mean > 0 and mean_rate > overall_mean * 1.3:
            flags.append(
                {
                    "type": "shift_cohort_bias",
                    "shift": shift,
                    "shift_avg_defect_rate_pct": round(mean_rate, 2),
                    "overall_avg_defect_rate_pct": round(overall_mean, 2),
                }
            )
    return flags


def _correlated_date_flags(df: pd.DataFrame) -> list[dict]:
    """Cross-line correlation: did every line take a hit on the same date?"""
    by_date = df.groupby("date")["downtime_minutes"].sum()
    mean_total = by_date.mean()
    flags = []
    for date, total in by_date.items():
        if mean_total > 0 and total > mean_total * 1.5:
            reasons = df.loc[df["date"] == date, "downtime_reason"].value_counts().to_dict()
            flags.append(
                {
                    "type": "correlated_multiline_event",
                    "date": date,
                    "total_downtime_minutes": round(total, 1),
                    "average_daily_downtime_minutes": round(mean_total, 1),
                    "downtime_reasons": reasons,
                }
            )
    return flags


def find_candidate_anomalies(df: pd.DataFrame) -> list[dict]:
    """Cheap statistical pre-filter, run before any LLM call."""
    return _zscore_row_flags(df) + _shift_cohort_flags(df) + _correlated_date_flags(df)


def summarize_for_prompt(df: pd.DataFrame) -> dict:
    """Aggregate stats only -- never send the raw per-row log to keep tokens down."""
    return {
        "period": {"start": df["date"].min(), "end": df["date"].max(), "total_shifts_logged": len(df)},
        "overall_averages": {
            "throughput_pct": round(df["throughput_pct"].mean(), 1),
            "defect_rate_pct": round(df["defect_rate_pct"].mean(), 2),
            "oee_pct": round(df["oee_pct"].mean(), 1),
            "downtime_minutes": round(df["downtime_minutes"].mean(), 1),
        },
        "by_line": df.groupby("line")[["throughput_pct", "defect_rate_pct", "oee_pct", "downtime_minutes"]]
        .mean()
        .round(2)
        .to_dict(orient="index"),
        "by_shift": df.groupby("shift")[["throughput_pct", "defect_rate_pct", "oee_pct", "downtime_minutes"]]
        .mean()
        .round(2)
        .to_dict(orient="index"),
        "downtime_reason_totals_minutes": df.groupby("downtime_reason")["downtime_minutes"].sum().round(1).to_dict(),
    }


def analyze(df: pd.DataFrame) -> dict:
    """Run the full pipeline: stats pre-filter -> Gemini synthesis. Returns parsed JSON dict."""
    payload = {
        "aggregate_stats": summarize_for_prompt(df),
        "statistically_flagged_candidates": find_candidate_anomalies(df),
    }

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment
    response = client.models.generate_content(
        model=MODEL,
        contents="Here is the shift production data for the period:\n\n" + json.dumps(payload, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AnalysisResult,
        ),
    )

    if not response.text:
        raise RuntimeError(f"Empty response from Gemini (finish reason: {response.candidates[0].finish_reason}).")

    return json.loads(response.text)
