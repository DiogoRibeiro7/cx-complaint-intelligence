# CX Complaint Intelligence

A portfolio-grade customer-experience intelligence platform built on the
[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
It turns raw complaint records into operational signals: which products and
companies concentrate risk, which complaint themes are emerging, which
complaints look operationally severe, and which response patterns predict weak
customer outcomes.

The project is intentionally end-to-end rather than a single notebook:

- A reproducible **data pipeline** with ingestion, normalisation, cleaning,
  feature engineering, topic modelling, and rule-based risk scoring.
- A **command-line interface** that produces auditable artefacts and run
  manifests for every stage.
- A polished **notebook suite** (10 notebooks, one per analytical question).
- A live **Streamlit dashboard** for operational triage.
- Tests, type checks, lint gates, and CI configuration.

## Why this exists

Most "complaint analysis" demos stop at sentiment over a small CSV. This
project deliberately mirrors what an applied data science team actually
ships: stable schema contracts, traceable artefacts, interpretable scoring,
time-aware modelling, monitoring, and operational hand-off.

## The five questions the project answers

1. **Where** do complaints concentrate &mdash; products, issues, companies, time?
2. **What** do customers complain about &mdash; discovered topics and their drift?
3. **Which** complaints look operationally severe and should be reviewed first?
4. **How** good is the response &mdash; timeliness, response codes, untimely segments?
5. **What changed this week** &mdash; weekly KPIs, deltas, escalation triggers, brief.

Each of these is backed by an analysis notebook and, where it makes sense,
exposed interactively in the dashboard.

## Where to start

- New to the project? Read [Getting started](getting-started.md).
- Want to see the analysis? Open the [notebook suite](notebooks/index.md).
- Want the codebase tour? See [Architecture](architecture.md).
- Want to try the live tool? See [Dashboard](dashboard.md).
- Curious about modelling choices? See [Modelling](modelling.md).
- Reviewing for compliance? See [Data ethics](ethics.md).

!!! info "Sample data"
    A small CFPB sample (1,000 June-2024 complaints with consumer narratives)
    is committed under `data/raw/cfpb_sample.csv` so the whole site can be
    reproduced from a fresh clone without hitting the public API.
