# Analysis notebooks

This directory contains the analysis suite for the CX Complaint Intelligence project. The notebooks share a common structure so a reader can move between them without re-learning conventions.

## Reading order

The notebooks build on one another. The intended order is:

| # | Notebook | Purpose | Inputs |
|---|----------|---------|--------|
| 01 | `01_complaint_intelligence_analysis.ipynb` | Headline KPIs, concentration analysis, monthly trend, risk distribution. | `data/raw/cfpb_sample.csv` |
| 02 | `02_topic_deep_dive.ipynb` | Topic-level risk concentration with representative complaints. | `reports/sample/*` |
| 03 | `03_supervised_baseline_preparation.ipynb` | Logistic-regression baseline for prioritising high-risk complaints. | `data/raw/cfpb_sample.csv` |
| 04 | `04_model_comparison_and_thresholds.ipynb` | Compare LR vs. linear SVM; sweep operating thresholds. | `data/raw/cfpb_sample.csv` |
| 05 | `05_company_risk_benchmarking.ipynb` | Volume- and risk-share leaderboards with a minimum-volume guardrail. | `data/raw/cfpb_sample.csv` |
| 06 | `06_topic_trend_monitoring.ipynb` | Month-over-month topic deltas with an emerging-topic flag. | `reports/sample/*` |
| 07 | `07_data_quality_and_schema_audit.ipynb` | Schema conformance, ID integrity, missingness, narrative coverage. | `data/raw/cfpb_sample.csv` |
| 08 | `08_response_quality_and_operational_playbook.ipynb` | Priority queue: high-risk + untimely + weak response. | `data/raw/cfpb_sample.csv` |
| 09 | `09_weekly_monitoring_scorecard.ipynb` | Weekly KPIs, WoW deltas, topic movement, escalation checklist. | `reports/sample/*` |
| 10 | `10_executive_brief_generator.ipynb` | Stakeholder-ready Markdown brief built from computed numbers only. | `reports/sample/*` |

## How to reproduce

1. Install dependencies and the package once:

   ```bash
   poetry install
   ```

2. Fetch (or commit) a sample dataset under `data/raw/cfpb_sample.csv`. A small sample ships with the repository.

3. Run the analysis pipeline so notebooks 02, 06, 09, 10 have inputs:

   ```bash
   poetry run cx-complaints analyze --input data/raw/cfpb_sample.csv --output-dir reports/sample
   ```

4. Launch a kernel (Jupyter, VS Code, or `nbconvert --execute`) and open any notebook. Every notebook resolves the project root from its own location, so the working directory does not need to be the repo root.

## Conventions

Every notebook follows the same skeleton:

1. **Header** — title, audience, owner, last-reviewed date, runtime, the source files it reads from.
2. **Setup** — calls `setup_notebook("<id>")` from `cx_complaints.reports.notebook_utils` and prints the environment snapshot.
3. **Analysis sections** — numbered, with a short markdown introduction before each computation.
4. **Insights** — Markdown rendered from the **computed tables**, never hand-written numbers. This keeps the analysis honest and reproducible.
5. **Limitations** — a standard list of caveats reviewers should know before acting.
6. **Exports** — `save_table` / `save_figure` calls that persist every artefact under `reports/notebook/<id>/`, plus an `environment.json` snapshot for reproducibility audits.

## Shared utilities

The boilerplate lives in `src/cx_complaints/reports/notebook_utils.py`. It provides:

- `setup_notebook(notebook_id)` — applies the matplotlib house style, resolves paths, returns a `NotebookContext`.
- `load_complaints(ctx)` — loads, normalises, cleans, and scores the bundled sample in one call.
- `load_pipeline_outputs(ctx)` — loads `complaint_scores.csv` + `topic_terms.csv` and parses dates.
- `kpi_table`, `concentration_table`, `monthly_summary`, `topic_labels` — the standard aggregations used across notebooks.
- `horizontal_bar`, `save_figure`, `save_table` — uniform charting and persistence helpers.
- `render_environment`, `write_environment_snapshot` — reproducibility metadata.

Reach for these helpers first when extending the suite &mdash; it keeps the notebooks short and consistent.

## Where artefacts go

Each notebook writes to its own directory:

```
reports/
└── notebook/
    ├── 01/
    │   ├── concentration_products_issues.png
    │   ├── kpis.csv
    │   ├── environment.json
    │   └── …
    ├── 02/…
    └── …
```

Re-running a notebook overwrites that notebook's directory only, so a partial re-run does not corrupt other notebooks' outputs.
