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

## Suggested portfolio story

A good article or README narrative would be:

> I built a complaint intelligence platform that turns public consumer complaints into a prioritisation system for CX, compliance, and operations teams. The project combines data ingestion, NLP, topic discovery, interpretable risk scoring, and executive reporting.
