# Customer Experience Complaint Intelligence

A portfolio-grade customer experience analytics project built around the CFPB Consumer Complaint Database.

The goal is to transform raw complaint records into operational intelligence:

- Which products, issues, and companies generate the most complaints?
- Which complaint themes are emerging?
- Which complaints look operationally risky?
- Which response patterns are associated with weak customer outcomes?
- How can a CX, compliance, or data science team prioritise action?

This repository is intentionally designed as a strong applied data science project, not only as a sentiment-analysis notebook.

## Project scope

The first implementation supports:

1. Fetching complaint data from the public CFPB Consumer Complaint Database API.
2. Normalising the raw API or CSV fields into a stable internal schema.
3. Cleaning customer complaint narratives.
4. Extracting text and operational features.
5. Discovering topics with TF-IDF + NMF.
6. Creating a rule-based severity/risk score.
7. Generating Markdown and CSV report outputs.
8. Running the workflow from a CLI.
9. Testing the critical data-cleaning, schema, topic, and scoring logic.

## Repository layout

```text
.
├── ROADMAP.md
├── README.md
├── pyproject.toml
├── scripts/
│   └── run_sample_pipeline.sh
├── src/
│   └── cx_complaints/
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── modelling/
│       ├── nlp/
│       ├── preprocessing/
│       └── reports/
└── tests/
```

## Installation

This project uses Poetry.

```bash
poetry install --with dev
```

Run quality checks:

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
```

## Fetch a sample from CFPB

```bash
poetry run cx-complaints fetch \
  --output data/raw/cfpb_sample.csv \
  --limit 1000 \
  --has-narrative true \
  --date-min 2024-01-01
```

Useful filters:

```bash
poetry run cx-complaints fetch \
  --output data/raw/credit_reporting.csv \
  --limit 2000 \
  --product "Credit reporting or other personal consumer reports" \
  --has-narrative true
```

## Run the analysis pipeline

```bash
poetry run cx-complaints analyze \
  --input data/raw/cfpb_sample.csv \
  --output-dir reports/sample \
  --topics 8
```

The command creates:

```text
reports/sample/
├── executive_summary.md
├── complaint_scores.csv
├── topic_terms.csv
├── topic_distribution.csv
├── company_summary.csv
├── issue_summary.csv
└── product_summary.csv
```

## Generate EDA charts

```bash
poetry run cx-complaints eda \
  --input data/raw/cfpb_sample.csv \
  --output-dir reports/sample/eda \
  --top-n 10
```

The command creates:

```text
reports/sample/eda/
├── product_share.png
├── issue_share.png
├── monthly_volume_trend.png
├── risk_distribution.png
└── timely_response_share.png
```

## Validate data quality

```bash
poetry run cx-complaints validate \
  --input data/raw/cfpb_sample.csv \
  --min-narrative-share 0.2 \
  --max-missing-date-share 0.05
```

This command checks schema conformance, duplicate complaint IDs, missing
`date_received` share, and narrative coverage. It fails with non-zero exit when
failing checks are detected unless `--no-fail-on-error` is passed.

## Train supervised baselines

```bash
poetry run cx-complaints train-baselines \
  --input data/raw/cfpb_sample.csv \
  --output-dir reports/sample/baselines \
  --train-fraction 0.8 \
  --min-df 5 \
  --max-features 5000
```

The command creates:

```text
reports/sample/baselines/
├── baseline_metrics.csv
├── train_class_imbalance.csv
├── time_split_summary.csv
├── logistic_regression_top_positive_terms.csv
├── logistic_regression_top_negative_terms.csv
├── linear_svm_top_positive_terms.csv
├── linear_svm_top_negative_terms.csv
└── run_manifest.json
```

## Notebook analysis

The repository includes a polished analysis notebook at
`notebooks/01_complaint_intelligence_analysis.ipynb`.
It also includes:

- `notebooks/02_topic_deep_dive.ipynb`;
- `notebooks/03_supervised_baseline_preparation.ipynb`;
- `notebooks/04_model_comparison_and_thresholds.ipynb`;
- `notebooks/05_company_risk_benchmarking.ipynb`;
- `notebooks/06_topic_trend_monitoring.ipynb`;
- `notebooks/07_data_quality_and_schema_audit.ipynb`;
- `notebooks/08_response_quality_and_operational_playbook.ipynb`;
- `notebooks/09_weekly_monitoring_scorecard.ipynb`;
- `notebooks/10_executive_brief_generator.ipynb`.

Run Jupyter from the project root after installing dependencies:

```bash
poetry run jupyter notebook
```

Inside the notebook, load `data/raw/cfpb_sample.csv` and execute sections for:

- KPI snapshot;
- concentration by product/issue/company;
- monthly trend and risk trend;
- response-quality and risk distribution;
- evidence-based recommendations.

See [notebooks/README.md](notebooks/README.md) for the full suite description.

## Streamlit dashboard

An interactive companion to the notebooks lives under [src/cx_complaints/dashboard/](src/cx_complaints/dashboard/). Launch it after `analyze` has produced `reports/sample/`:

```bash
poetry run streamlit run src/cx_complaints/dashboard/app.py
```

The dashboard has five pages:

- **Overview** — headline KPIs, monthly trend, risk-score distribution.
- **Concentration** — top products, issues, and companies with a minimum-volume guardrail.
- **Topics** — topic leaderboard, volume-vs-risk bubble chart, deep-dive sampling.
- **Priority Queue** — filterable high-risk + untimely + weak-response queue with CSV download.
- **Weekly Brief** — live weekly KPIs, WoW deltas, and the auto-generated executive brief.

## Documentation site

The project ships a [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) site that renders the README, architecture notes, CLI reference, modelling notes, data-ethics policy, and the full notebook suite as a browsable HTML tree. Build it locally:

```bash
poetry run mkdocs serve         # live-reload at http://localhost:8000
poetry run mkdocs build --strict  # one-shot build into site/
```

The site sources live under [docs/](docs/), the build configuration is in [mkdocs.yml](mkdocs.yml), and a small hook at [scripts/mkdocs_hooks.py](scripts/mkdocs_hooks.py) mirrors the executed notebooks into the docs tree at build time. Both `site/` and `docs/notebooks/` are gitignored.

The data-loading layer (cached) and the pure aggregation helpers (`apply_filters`, `priority_queue`, `weekly_kpis`, `week_over_week`, `topic_weekly_movement`, `build_executive_brief`) are testable in isolation and shared across the pages.

## Example report

A pre-computed report from 1,000 June 2024 complaints with consumer narratives is checked into the repo under [reports/sample/](reports/sample/). The headline [executive_summary.md](reports/sample/executive_summary.md) covers top products, issues, companies, discovered topics, and recommended next actions.

The per-complaint `complaint_scores.csv` is intentionally not committed: it carries raw consumer narratives, and the data ethics note below applies. Regenerate it locally with the `analyze` command above.

## Main modelling choices

### Topic modelling

The first version uses TF-IDF and Non-negative Matrix Factorization. This is transparent, fast, and easy to explain in interviews.

Later versions should add BERTopic, sentence-transformer embeddings, and online topic drift detection.

### Severity/risk score

The first version uses a transparent heuristic score. It combines:

- untimely responses;
- weak response categories;
- high-risk products;
- high-risk issues;
- complaint narrative urgency terms;
- narrative length;
- vulnerable-consumer tags.

This is deliberately interpretable. In future versions, the score can become a calibrated supervised model.

### Supervised modelling scaffolding

The repository now includes reusable utilities for supervised model development:

- chronological train/test splitting (`time_aware_train_test_split`);
- class imbalance summaries (`summarize_class_imbalance`);
- standard binary metrics (`evaluate_binary_classifier`) with precision, recall, F1, and optional ROC-AUC.

These live in `src/cx_complaints/modelling/supervised.py` and are intended as the foundation for upcoming logistic regression and linear-SVM baselines.

The baseline model runners are now available in
`src/cx_complaints/modelling/baselines.py`:

- `run_text_baseline_models` for logistic regression and linear SVM over TF-IDF text;
- `baseline_metrics_table` for report-ready model metric comparisons.

## Example output

The executive summary is designed for a business audience. It focuses on:

- total complaints;
- share of high-risk complaints;
- top products;
- top issues;
- top companies by volume and risk;
- discovered topics;
- recommended next actions.

## Data ethics and limitations

Complaint narratives may contain sensitive situations even after CFPB scrubbing. Do not attempt to re-identify consumers. Do not use the project to rank individuals. Company comparisons should be normalised by market share or customer base before making public claims.

The risk score is not a legal or regulatory judgement. It is a prioritisation signal for analysis.

## DuckDB analytics storage

DuckDB support is available for local analytical tables in
`src/cx_complaints/data/duckdb_store.py`.

Example usage:

```python
from cx_complaints.data.duckdb_store import DuckDBStore

store = DuckDBStore("data/analytics/complaints.duckdb")
store.write_dataframe(frame=complaints_df, table_name="complaints", if_exists="replace")
summary = store.query("SELECT product, count(*) AS complaints FROM complaints GROUP BY product")
```

You can also persist pipeline-ready tables directly from the CLI:

```bash
poetry run cx-complaints persist-duckdb \
  --input data/raw/cfpb_sample.csv \
  --database data/analytics/complaints.duckdb \
  --reports-dir reports/sample \
  --if-exists replace
```

This writes the normalized `complaints` table and, when present in `--reports-dir`,
also persists `product_summary`, `issue_summary`, `company_summary`,
`topic_distribution`, `topic_terms`, and `baseline_metrics`.
