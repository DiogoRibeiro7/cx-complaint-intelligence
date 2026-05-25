# Roadmap

This roadmap describes how to evolve the project from a strong portfolio project into a production-style customer experience intelligence platform.

## Phase 0 — Foundation

Status: implemented in this repository.

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

Goal: create a polished analytical notebook or article.

- [ ] Add a notebook with a clear business question.
- [ ] Compare complaint volume by product, issue, state, and company.
- [ ] Analyse trends over time.
- [ ] Analyse response quality and timely-response patterns.
- [x] Add charts for product share, issue share, monthly trends, and risk distribution.
- [ ] Write an executive narrative with business recommendations.

Suggested deliverable:

```text
notebooks/01_complaint_intelligence_analysis.ipynb
```

## Phase 2 — Better NLP

Goal: improve the quality of extracted themes.

- [ ] Add phrase detection for recurring issue expressions.
- [ ] Add aspect-based sentiment analysis.
- [ ] Add sentence-transformer embeddings.
- [ ] Add BERTopic or clustering over embeddings.
- [ ] Add semantic search over complaint narratives.
- [ ] Add representative complaint extraction for each topic.
- [ ] Add topic labelling with an optional LLM.

Possible output:

```text
reports/topics/topic_cards.md
```

Each topic card should include:

- topic label;
- top terms;
- representative complaint summaries;
- affected products;
- affected companies;
- severity distribution;
- suggested CX action.

## Phase 3 — Supervised modelling

Goal: move from descriptive analytics to prediction.

Candidate models:

- timely-response prediction;
- company-response-category prediction;
- high-risk complaint prediction;
- relief-likelihood prediction;
- product or issue classification from narrative text.

Technical tasks:

- [ ] Create reproducible train/test splits by time.
- [ ] Add baseline logistic regression and linear SVM models.
- [ ] Add calibrated probabilities.
- [ ] Add model evaluation reports.
- [ ] Add confusion matrices and per-class metrics.
- [ ] Add drift checks between train and recent data.
- [ ] Add feature importance and explainability.

## Phase 4 — Risk scoring v2

Goal: replace the heuristic score with a calibrated decision-support model.

- [ ] Define a target label using combinations of response category, timely response, disputed status, and relief indicators.
- [ ] Train a supervised model with text and metadata features.
- [ ] Calibrate probabilities.
- [ ] Compare model score with the initial heuristic score.
- [ ] Add threshold selection based on operational capacity.
- [ ] Add fairness checks across product, state, and vulnerable-consumer tags.

## Phase 5 — Monitoring and trend detection

Goal: identify emerging complaint themes.

- [ ] Add monthly trend tables.
- [ ] Add exponentially weighted moving averages.
- [ ] Add change-point detection for complaint volume and topic share.
- [ ] Add anomaly detection for sudden issue spikes.
- [ ] Add company-specific topic drift monitoring.
- [ ] Add alert generation for unusual complaint increases.

## Phase 6 — Dashboard

Goal: make the project usable by non-technical stakeholders.

Recommended first version: Streamlit.

Views:

- [ ] Executive overview.
- [ ] Product and issue explorer.
- [ ] Company benchmarking.
- [ ] Topic explorer.
- [ ] High-risk complaint queue.
- [ ] Trend and anomaly dashboard.
- [ ] Search and filter interface.

## Phase 7 — Data engineering

Goal: turn the project into a repeatable pipeline.

- [ ] Add incremental ingestion.
- [ ] Store raw API responses separately from cleaned analytical tables.
- [ ] Add DuckDB support for local analytics.
- [ ] Add data validation checks.
- [ ] Add pipeline metadata.
- [ ] Add GitHub Actions scheduled runs.
- [ ] Add artifact publishing for reports.

Suggested storage structure:

```text
data/
├── raw/cfpb/yyyy/mm/dd/
├── interim/
├── processed/
└── analytics/
```

## Phase 8 — LLM-assisted reporting

Goal: create useful summaries while keeping traceability.

- [ ] Generate topic summaries from representative complaints.
- [ ] Generate company-level risk narratives.
- [ ] Generate issue-level action recommendations.
- [ ] Add citation links back to complaint IDs.
- [ ] Add guardrails to avoid unsupported claims.
- [ ] Keep all numerical claims computed by code, not by the LLM.

## Phase 9 — Production-quality package

Goal: make the project maintainable and credible.

- [ ] Add full documentation site.
- [ ] Add API reference.
- [ ] Add CI with ruff, mypy, pytest, coverage, and security checks.
- [ ] Add Dockerfile.
- [ ] Add Makefile or task runner.
- [ ] Add release workflow.
- [ ] Add example reports generated from a fixed sample.

## Phase 10 — Research extensions

Possible research-grade directions:

- complaint topic drift over regulatory events;
- systematic anomaly detection in complaint narratives;
- measuring effect of narrative clarity on company response outcomes;
- comparing complaint outcomes across products and response categories;
- studying LLM-style complaint narratives after late 2022;
- weak supervision for complaint severity labels.
