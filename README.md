# Ops Insights Prototype

An LLM-powered tool that summarizes production/shift data and flags anomalies -- built on patterns from real, hands-on manufacturing floor experience: OEE (Availability x Performance x Quality), downtime reason codes, and shift handoffs.

Point it at a shift log (or generate a realistic synthetic one) and it surfaces what a floor supervisor actually needs to know: how the period went, what broke the pattern, and why.

## Why this exists

Manufacturing floors generate a shift log every day -- units produced, downtime minutes, defect counts -- but turning that into "here's what actually needs attention" usually falls on whoever's willing to stare at a spreadsheet. This prototype automates the first pass: cheap statistics catch the outliers, Claude explains them in plain language and tells one-off incidents apart from systemic patterns.

## How it works

1. **Data** -- `src/data_generator.py` produces a synthetic shift log (date x shift x line) with realistic seeded patterns: a multi-shift breakdown streak on one line, a systemic night-shift defect-rate drift, and a same-day material shortage that hits every line at once. Or upload your own CSV in the same shape.
2. **Statistical pre-filter** -- `src/analyzer.py` runs three cheap passes with pandas/numpy *before* any LLM call: per-line z-scores on downtime and defect rate (point outliers), shift-level cohort averages (systemic bias, e.g. "Night shift runs consistently worse"), and cross-line correlation by date (an event that hit every line at once). Only the aggregate stats and flagged candidates go to Claude -- never the raw per-row log -- to keep token usage down as the log grows.
3. **Claude synthesis** -- Claude gets the aggregates plus the candidate list and a manufacturing-analyst system prompt, and returns a structured (JSON-schema-constrained) period summary, a severity-ranked anomaly list with likely cause and recommended action, and a few highlights.
4. **Dashboard** -- Streamlit shows the trends (OEE by line, downtime by reason, defect rate by shift) and the Claude readout side by side.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
streamlit run app.py
```

The dashboard works without an API key -- you can generate data and explore the charts. The "Analyze with Claude" button needs `ANTHROPIC_API_KEY` set.

## Bringing your own data

Upload a CSV with these columns: `date, shift, line, units_target, units_produced, downtime_minutes, downtime_reason, defect_count, availability, performance`. `availability` and `performance` are decimals (0-1); everything else derives automatically (throughput %, defect rate %, OEE %).

## Stack

Python, [Streamlit](https://streamlit.io), [Anthropic Claude API](https://platform.claude.com) (structured outputs via `output_config.format`), pandas, numpy.

## Honest limitations

- The anomaly detection is intentionally simple -- z-scores and group means, not a real time-series model. That's a deliberate choice for a prototype: it's auditable and explainable, and Claude's job is reasoning about *why* something looks off, not detecting the outlier itself.
- No persistence -- every session starts from freshly generated (or uploaded) data. A real deployment would want a database and a way to compare period-over-period.
- The synthetic data generator's "realism" is calibrated by eye against general manufacturing reporting conventions, not a specific plant's actual data.
