# Dashboard

An interactive Streamlit companion to the notebook suite. Same numbers, same
data sources, but built for live triage rather than reporting.

## Launch

```bash
poetry run streamlit run src/cx_complaints/dashboard/app.py
```

The home page reports whether the bundled sample and the pipeline outputs are
present, and lists the project paths it resolved. The five analytical pages
live in the sidebar.

## Pages

### Overview

Headline KPIs (total, high-risk, medium-or-higher, untimely), a dual-axis
monthly trend chart (volume vs. high-risk share), and the risk-score
distribution.  A sidebar filter narrows by product and risk level.

### Concentration

Top products, top issues, and two views of companies:

- **Highest volume** &mdash; raw complaint count regardless of company size.
- **Highest high-risk share** &mdash; ranked by risk concentration, with a
  configurable minimum-volume threshold so a single complaint at a small
  company cannot dominate the leaderboard.

### Topics

The topic leaderboard (volume, average risk, high-risk share, label) plus a
volume-vs-risk bubble chart so emerging-and-severe topics are visually
obvious. A topic deep-dive panel surfaces the five highest-scoring complaints
for any topic and shows monthly topic-share trends.

Requires `cx-complaints analyze` to have run first.

### Priority queue

The dashboard's most operational page. Returns complaints that are
simultaneously:

- `risk_level == "high"`,
- not timely (`timely != "yes"`),
- and have a weak response code (`closed`, `closed with explanation`, or `in progress`).

Sorted by `risk_score` descending with sidebar filters by product and
company. Includes a CSV download button for handing the queue off.

### Weekly brief

The live equivalent of notebook 09 + notebook 10. Shows the latest week's
KPIs, week-over-week deltas, two trend charts, topic movement, and the
auto-generated executive brief. Numbers come exclusively from the computed
weekly KPI table.

Requires `cx-complaints analyze` to have run first.

## Architecture

The dashboard sits on top of two layers:

- A **cached loader layer** in `cx_complaints.dashboard.loaders`
  (`load_complaints`, `load_pipeline_outputs`) that wraps the notebook helpers
  with `@st.cache_data` so the same parsed frame is reused across page
  interactions. This module is the only one that imports `streamlit`.
- A **pure helper layer** in `cx_complaints.dashboard.helpers` (`apply_filters`,
  `priority_queue`, `weekly_kpis`, `week_over_week`, `topic_weekly_movement`,
  `build_executive_brief`) that takes DataFrames in and returns DataFrames or
  strings out, with no Streamlit imports. These helpers are unit-tested in
  isolation and remain importable in lightweight installs that omit the
  `dashboard` extra.

The pages stay thin: each is roughly 100–150 lines of Streamlit calls plus a
few Altair chart definitions.
