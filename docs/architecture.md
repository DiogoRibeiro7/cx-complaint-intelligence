# Architecture

The codebase is organised around the lifecycle of a complaint: **ingest →
normalise → clean → feature-engineer → model → score → report → serve**.

```text
src/cx_complaints/
├── cli.py                    # click-based CLI entry points
├── config.py                 # pydantic settings
├── data/
│   ├── cfpb_client.py        # HTTP client for the CFPB public API
│   ├── duckdb_store.py       # DuckDB analytics store
│   ├── schema.py             # canonical column contract + normalisation
│   └── validation.py         # dataset quality checks
├── features/
│   └── text_features.py      # word_count, urgency terms, money language…
├── modelling/
│   ├── baselines.py          # logistic regression / linear SVM runners
│   ├── risk.py               # interpretable risk score
│   └── supervised.py         # time-aware split, class imbalance, metrics
├── nlp/
│   └── topics.py             # TF-IDF + NMF topic discovery
├── pipeline/
│   ├── manifest.py           # run-manifest schema + validation
│   └── metadata.py           # run-manifest builders + writer
├── preprocessing/
│   └── text.py               # narrative cleaning (URLs/emails/redactions)
├── reports/
│   ├── eda.py                # chart artefacts (PNG)
│   ├── notebook_utils.py     # shared helpers for the notebook suite
│   ├── summary.py            # markdown executive summary builder
│   └── tables.py             # value_summary / topic_distribution
└── dashboard/
    ├── app.py                # Streamlit landing page
    ├── data.py               # cached loaders + pure aggregation helpers
    └── pages/                # 5 pages (Overview, Concentration, …)
```

## Contracts

Two contracts hold the system together:

| Contract | Where it lives | Why it matters |
| --- | --- | --- |
| Canonical complaint schema | `data/schema.CANONICAL_COLUMNS` | Notebooks, the dashboard, and the scoring code all assume these exact columns. Both API responses (snake_case) and CSV exports (Title Case) feed through `normalize_complaints`. |
| Run manifest | `pipeline/manifest.RunManifest` | Every CLI command writes a manifest with start/finish timestamps, parameters, output paths, input/output row counts, and validation status. This makes pipeline runs auditable. |

## Lifecycle of one complaint

1. `cfpb_client.CFPBClient.fetch` pulls raw rows from the public API and
   `save_complaints_csv` lands them under `data/raw/`.
2. `normalize_complaints` flattens column-name variants, parses dates,
   coerces booleans, and enforces the canonical column order.
3. `add_clean_text_column` strips URLs, emails, and the CFPB `XXXX`
   redaction placeholder while preserving structure.
4. `add_text_features` extracts `word_count`, urgency-term count, money
   language flags, and narrative availability.
5. `score_complaint_risk` combines those features with metadata (timely,
   company_response, tags) into an interpretable additive score and a
   discrete `risk_level`.
6. `fit_nmf_topics` runs TF-IDF + NMF on narratives that have meaningful
   length and merges the assignments back into the frame.
7. `write_report_outputs` emits the per-complaint scored table, the topic
   terms table, per-category summaries, and the markdown executive
   summary.
8. Notebooks under `notebooks/` and the Streamlit dashboard read these
   artefacts to surface answers.

## Engineering choices

- **Transparent baselines first.** TF-IDF + NMF for topics; an additive
  numeric score for risk; logistic regression and linear SVM for supervised
  baselines. Anything more complex is welcome (BERTopic, embeddings) as long
  as the baselines remain available, per the
  [contributor agreement](https://github.com/diogoribeiro/cx-complaint-intelligence/blob/main/AGENTS.md).
- **Time-aware splits.** `time_aware_train_test_split` is used wherever the
  goal is to predict future outcomes. Random splits would silently leak
  future information into training.
- **Audit trails by default.** Every command emits a `run_manifest.json`
  beside its outputs. Validation failures are surfaced via exit codes
  unless the caller explicitly opts out.
- **Data ethics by contract.** Narratives are not republished outside
  `data/raw/`; the per-complaint scored CSV is gitignored even though the
  rest of `reports/sample/` is committed.

## Where the pieces converge

```text
                       ┌─────────────────────────┐
                       │  data/raw/cfpb_sample…  │
                       └────────────┬────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ cli.analyze: normalize → clean → features → risk  │
        │              → topics → write_report_outputs      │
        └────────────┬───────────────────────────┬──────────┘
                     │                           │
                     ▼                           ▼
            reports/sample/*.csv          run_manifest.json
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Notebook suite             Streamlit dashboard
  (analysis, narrative)      (interactive triage)
```
