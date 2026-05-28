# Roadmap

This roadmap describes how the project has evolved from a small CFPB analysis into a portfolio-grade complaint intelligence platform, and where it still has room to grow.

## Current state — phases 0–9 shipped

The repository today provides:

- a normalised, validated CFPB pipeline with audit trails (`RunManifest`);
- two interpretable topic backends (TF-IDF + NMF baseline; opt-in sentence-transformer + KMeans + c-TF-IDF labelling) with schema-compatible outputs;
- a transparent risk score plus supervised text baselines (logistic regression, linear SVM) with threshold tooling;
- a 12-document analysis suite (11 notebooks + a shared README) executed end-to-end against the bundled sample;
- a five-page Streamlit dashboard backed by tested pure helpers;
- a MkDocs Material documentation site that renders the README, architecture, CLI reference, modelling notes, ethics policy, and the notebook outputs;
- DuckDB analytics storage, dataset-validation CLI, and a run-manifest schema with tests;
- a CI workflow (ruff + mypy + pytest) and an engineering checklist that governs PR contracts.

## Phase 0 — Foundation

- [x] CFPB API ingestion.
- [x] Stable complaint schema.
- [x] Narrative text cleaning.
- [x] Basic text and operational features.
- [x] TF-IDF + NMF topic modelling.
- [x] Interpretable severity/risk scoring.
- [x] Markdown and CSV reporting.
- [x] CLI entrypoint.
- [x] Unit tests for core modules.

## Phase 1 — Exploratory analysis and portfolio notebook

- [x] Notebook with a clear business question (`01_complaint_intelligence_analysis.ipynb`).
- [x] Compare complaint volume by product, issue, state, and company (NB 01 + NB 05).
- [x] Analyse trends over time (NB 01 + NB 06 + NB 09).
- [x] Analyse response quality and timely-response patterns (NB 08).
- [x] Add charts for product share, issue share, monthly trends, and risk distribution.
- [x] Write an executive narrative with business recommendations (NB 10 + per-notebook computed insights).

## Phase 2 — Better NLP

- [x] Add sentence-transformer embeddings (`nlp/embedding_topics.SentenceTransformerEncoder`).
- [x] Add clustering over embeddings with c-TF-IDF labelling (BERTopic-style).
- [x] Add representative complaint extraction for each topic (NB 02 + Topics dashboard page).
- [ ] Add phrase detection for recurring issue expressions.
- [ ] Add aspect-based sentiment analysis.
- [ ] Add semantic search over complaint narratives.
- [ ] Add topic labelling with an optional LLM.

Each topic card today already includes top terms, representative complaints, affected products, affected companies, and severity distribution. Suggested CX-action automation remains future work.

## Phase 3 — Supervised modelling

- [x] Create reproducible train/test splits by time (`modelling.supervised.time_aware_train_test_split`).
- [x] Add baseline logistic regression and linear SVM models (`modelling.baselines`).
- [x] Add model evaluation reports (NB 03 + NB 04 + `train-baselines` CLI).
- [ ] Add calibrated probabilities (`CalibratedClassifierCV`).
- [ ] Add confusion matrices and per-class metrics in the artefact pack.
- [ ] Add drift checks between train and recent data.
- [ ] Add feature importance and explainability (SHAP / per-token contribution beyond logistic coefficients).

## Phase 4 — Risk scoring v2

The current risk score is heuristic and intentionally transparent. Replacing it with a calibrated supervised model is reserved for when a reviewed-outcomes dataset exists.

- [ ] Define a target label using combinations of response category, timely response, disputed status, and relief indicators.
- [ ] Train a supervised model with text and metadata features.
- [ ] Calibrate probabilities.
- [ ] Compare model score with the initial heuristic score.
- [ ] Add threshold selection based on operational capacity (partial: NB 04 sweeps thresholds for the LR baseline).
- [ ] Add fairness checks across product, state, and vulnerable-consumer tags.

## Phase 5 — Monitoring and trend detection

- [x] Add monthly trend tables (NB 06).
- [x] Add per-week scorecard with WoW deltas and topic movement (NB 09).
- [x] Emerging-topic flag (delta-pp threshold) in NB 06.
- [ ] Add exponentially weighted moving averages.
- [ ] Add change-point detection for complaint volume and topic share.
- [ ] Add anomaly detection for sudden issue spikes.
- [ ] Add company-specific topic drift monitoring.
- [ ] Add alert generation for unusual complaint increases.

## Phase 6 — Dashboard

- [x] Executive overview (Overview page).
- [x] Product and issue explorer (Concentration page).
- [x] Company benchmarking with min-volume guardrail (Concentration page).
- [x] Topic explorer with bubble chart and deep-dive (Topics page).
- [x] High-risk complaint queue with CSV export (Priority Queue page).
- [x] Trend and weekly-brief view (Weekly Brief page).
- [ ] Search and free-text filter interface.
- [ ] Live deployment (Streamlit Cloud or HF Space) so reviewers can click without cloning.

## Phase 7 — Data engineering

- [x] Add DuckDB support for local analytics (`data/duckdb_store.py` + `persist-duckdb` CLI).
- [x] Add data validation checks (`data/validation.py` + `validate` CLI).
- [x] Add pipeline metadata (`pipeline/manifest.py` + per-command `run_manifest.json`).
- [ ] Add incremental ingestion (`fetch` is currently full-snapshot).
- [ ] Store raw API responses separately from cleaned analytical tables.
- [ ] Add GitHub Actions scheduled runs (`cron` workflow).
- [ ] Add artifact publishing for reports.

## Phase 8 — LLM-assisted reporting

The weekly executive brief (NB 10 + dashboard) builds its narrative from computed tables only, without LLM involvement, by design. The phase below would *optionally* add LLM-assisted summaries while keeping that traceability guarantee.

- [ ] Generate topic summaries from representative complaints.
- [ ] Generate company-level risk narratives.
- [ ] Generate issue-level action recommendations.
- [ ] Add citation links back to complaint IDs.
- [ ] Add guardrails to avoid unsupported claims.
- [ ] Keep all numerical claims computed by code, not by the LLM.

## Phase 9 — Production-quality package

- [x] Add full documentation site (MkDocs Material + mkdocs-jupyter).
- [x] Add CI with ruff, mypy, pytest.
- [x] Add example reports generated from a fixed sample (`reports/sample/`).
- [x] Add engineering checklist for contract-aware PR review.
- [ ] Add API reference (mkdocstrings).
- [ ] Add coverage upload + badge.
- [ ] Add security scanning (bandit / pip-audit) to CI.
- [ ] Add Dockerfile.
- [ ] Add Makefile or task runner.
- [ ] Add release workflow.
- [ ] Add pre-commit hooks enforcing ruff + a fast pytest subset.
- [ ] Add Dependabot / Renovate for dependency hygiene.

## Phase 10 — Research extensions

Possible research-grade directions:

- complaint topic drift over regulatory events;
- systematic anomaly detection in complaint narratives;
- measuring the effect of narrative clarity on company response outcomes;
- comparing complaint outcomes across products and response categories;
- studying LLM-style complaint narratives after late 2022;
- weak supervision for complaint severity labels;
- counterfactual analysis of timely-response interventions.
