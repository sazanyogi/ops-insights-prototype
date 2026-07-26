"""Synthetic manufacturing shift/production data for the Ops Insights prototype.

Modeled loosely on real production-floor reporting: shift-level units produced
vs. target, downtime with reason codes, defect/scrap counts, and OEE
(Availability x Performance x Quality). A few realistic anomaly patterns are
seeded in on top of the baseline noise -- a multi-shift breakdown streak on
one line, a systemic night-shift quality drift, and a same-day material
shortage that hits every line at once -- so there's something worth finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHIFT_MINUTES = 480  # 8-hour shifts
SHIFTS = ["Day", "Evening", "Night"]
DOWNTIME_REASONS = [
    "Changeover",
    "Planned Maintenance",
    "Unplanned Breakdown",
    "Material Shortage",
    "Quality Hold",
    "Startup/Warmup",
]
DOWNTIME_WEIGHTS = [0.35, 0.20, 0.15, 0.08, 0.12, 0.10]

# baseline target units per 8-hour shift, per line
LINE_TARGETS = {
    "Line 1": 1200,
    "Line 2": 950,
    "Line 3": 1100,
}


def generate_shift_data(days: int = 14, seed: int = 42) -> pd.DataFrame:
    """Generate one row per (date, shift, line) with seeded anomaly patterns."""
    rng = np.random.default_rng(seed)
    lines = list(LINE_TARGETS.keys())
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    breakdown_line = lines[-1]
    breakdown_dates = set(dates[-3:-1])  # last-but-one and last-but-two days
    shortage_date = dates[days // 2]

    rows = []
    for date in dates:
        for shift in SHIFTS:
            for line in lines:
                target = LINE_TARGETS[line]

                downtime = max(0.0, rng.normal(loc=30, scale=12))
                reason = rng.choice(DOWNTIME_REASONS, p=DOWNTIME_WEIGHTS)

                # Seeded anomaly 1: multi-shift breakdown streak on one line
                if line == breakdown_line and date in breakdown_dates:
                    downtime = rng.uniform(150, 230)
                    reason = "Unplanned Breakdown"

                # Seeded anomaly 2: material shortage hits every line the same day
                if date == shortage_date:
                    downtime += rng.uniform(50, 90)
                    reason = "Material Shortage"

                downtime = min(downtime, SHIFT_MINUTES * 0.9)
                availability = (SHIFT_MINUTES - downtime) / SHIFT_MINUTES

                performance = np.clip(rng.normal(loc=0.95, scale=0.03), 0.6, 1.05)
                units_produced = round(target * availability * performance)

                # Seeded anomaly 3: systemic night-shift quality drift (fatigue/staffing)
                base_defect_rate = rng.normal(loc=0.02, scale=0.006)
                if shift == "Night":
                    base_defect_rate *= 1.7
                defect_rate = float(np.clip(base_defect_rate, 0.001, 0.25))

                defect_count = round(units_produced * defect_rate)
                scrap_count = round(defect_count * rng.uniform(0.3, 0.6))
                operators = rng.integers(4, 9)

                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "shift": shift,
                        "line": line,
                        "units_target": target,
                        "units_produced": int(units_produced),
                        "downtime_minutes": round(downtime, 1),
                        "downtime_reason": reason,
                        "defect_count": int(defect_count),
                        "scrap_count": int(scrap_count),
                        "operators": int(operators),
                        "availability": round(availability, 4),
                        "performance": round(performance, 4),
                    }
                )

    df = pd.DataFrame(rows)
    return add_derived_metrics(df)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach throughput/defect/OEE columns derived from the raw shift log."""
    df = df.copy()
    df["throughput_pct"] = (df["units_produced"] / df["units_target"] * 100).round(1)
    df["defect_rate_pct"] = (df["defect_count"] / df["units_produced"].replace(0, 1) * 100).round(2)
    quality = 1 - (df["defect_count"] / df["units_produced"].replace(0, 1))
    df["oee_pct"] = (df["availability"] * df["performance"] * quality * 100).round(1)
    return df
