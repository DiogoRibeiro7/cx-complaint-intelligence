# CLI reference

The `cx-complaints` CLI is a thin `click` layer over the pipeline modules.
Every command writes a `run_manifest.json` alongside its output so runs are
auditable.

## `fetch`

Pull complaints from the public CFPB API and save them as CSV.

```bash
poetry run cx-complaints fetch \
    --output data/raw/cfpb_sample.csv \
    --limit 1000 \
    --page-size 100 \
    --has-narrative true \
    --date-min 2024-01-01
```

| Option | Default | What it does |
|---|---|---|
| `--output` | (required) | Destination CSV path. Parent directories are created. |
| `--limit` | 1000 | Maximum rows to retain. |
| `--page-size` | 100 | API page size for the underlying client. |
| `--has-narrative` | true | Only fetch complaints with a consumer narrative. |
| `--product` | None | Filter by product name. |
| `--company` | None | Filter by company name. |
| `--state` | None | Filter by US state. |
| `--date-min` | None | Inclusive lower bound on `date_received`. |
| `--date-max` | None | Inclusive upper bound on `date_received`. |
| `--search-term` | None | Narrative full-text filter. |

## `analyze`

Run the full pipeline against a saved CSV and write report artefacts plus a
run manifest.

```bash
poetry run cx-complaints analyze \
    --input data/raw/cfpb_sample.csv \
    --output-dir reports/sample \
    --topics 8 \
    --min-df 5 \
    --max-features 5000
```

Produces, under the output directory:

- `complaint_scores.csv` &mdash; per-complaint scored frame (gitignored).
- `topic_terms.csv` &mdash; top terms per discovered topic.
- `topic_distribution.csv`, `product_summary.csv`, `issue_summary.csv`, `company_summary.csv`.
- `executive_summary.md` &mdash; the stakeholder-ready markdown brief.
- `run_manifest.json`.

## `eda`

Generate the EDA chart pack.

```bash
poetry run cx-complaints eda \
    --input data/raw/cfpb_sample.csv \
    --output-dir reports/sample/eda \
    --top-n 10
```

Writes `product_share.png`, `issue_share.png`, `monthly_volume_trend.png`,
`risk_distribution.png`, `timely_response_share.png`, and a `run_manifest.json`.

## `validate`

Run dataset-quality checks. Fails with a non-zero exit code when any check
returns `status=fail`, unless `--no-fail-on-error` is passed.

```bash
poetry run cx-complaints validate \
    --input data/raw/cfpb_sample.csv \
    --min-narrative-share 0.2 \
    --max-missing-date-share 0.05
```

Writes the validation manifest to `reports/validation/run_manifest.json`.

## `train-baselines`

Train the logistic-regression and linear-SVM text baselines, with class
imbalance and time-split context.

```bash
poetry run cx-complaints train-baselines \
    --input data/raw/cfpb_sample.csv \
    --output-dir reports/sample/baselines \
    --train-fraction 0.8 \
    --min-df 5 \
    --max-features 5000
```

Writes per-model top positive/negative terms, the metrics table, the class
imbalance summary, the time-split summary, and a `run_manifest.json`.

## `persist-duckdb`

Materialise the normalised complaints frame and the available report tables
into a DuckDB analytics database.

```bash
poetry run cx-complaints persist-duckdb \
    --input data/raw/cfpb_sample.csv \
    --database data/analytics/complaints.duckdb \
    --reports-dir reports/sample \
    --if-exists replace
```

This persists the `complaints` table plus any of `product_summary`,
`issue_summary`, `company_summary`, `topic_distribution`, `topic_terms`, and
`baseline_metrics` that exist in `--reports-dir`.
