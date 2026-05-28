# Getting started

The project ships with a bundled sample, so the full pipeline reproduces from
a fresh clone in under a minute.

## Prerequisites

- Python 3.10, 3.11, or 3.12.
- [Poetry](https://python-poetry.org/) for dependency management.

## Install

```bash
git clone https://github.com/diogoribeiro/cx-complaint-intelligence
cd cx-complaint-intelligence

# Light install: CLI + notebooks + NMF topic backend
poetry install

# Add features as needed
poetry install --extras dashboard           # Streamlit dashboard
poetry install --extras embeddings          # sentence-transformers + torch
poetry install --extras all                 # everything

# Contributors
poetry install --with dev --extras all
```

## Reproduce the canonical outputs

The bundled sample lives at `data/raw/cfpb_sample.csv`. Run the pipeline to
produce the artefacts the notebooks and dashboard depend on:

```bash
poetry run cx-complaints analyze \
    --input data/raw/cfpb_sample.csv \
    --output-dir reports/sample
```

That writes:

```text
reports/sample/
├── executive_summary.md
├── complaint_scores.csv          # gitignored: contains raw narratives
├── topic_terms.csv
├── topic_distribution.csv
├── product_summary.csv
├── issue_summary.csv
├── company_summary.csv
└── run_manifest.json
```

## Run the quality gates

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
```

These are the same three commands the [engineering checklist](https://github.com/diogoribeiro/cx-complaint-intelligence/blob/main/ENGINEERING_CHECKLIST.md)
asks for before any PR or commit.

## Refresh from the live API

To pull new complaints from CFPB:

```bash
poetry run cx-complaints fetch \
    --output data/raw/cfpb_sample.csv \
    --limit 1000 \
    --has-narrative true \
    --date-min 2024-01-01
```

Useful options:

- `--product` &mdash; constrain to a specific product line.
- `--company` &mdash; constrain to a specific company.
- `--state` &mdash; constrain to a US state.
- `--search-term` &mdash; narrative full-text filter.

See the [CLI reference](cli.md) for the full surface area.

## Open the notebooks

```bash
poetry run jupyter notebook notebooks/
```

Every notebook starts with a `setup_notebook("<id>")` cell that resolves the
project root from whichever directory you launched the kernel in, so it does
not matter whether you launch Jupyter from the repo root or from the
`notebooks/` folder.

## Launch the dashboard

```bash
poetry run streamlit run src/cx_complaints/dashboard/app.py
```

See [Dashboard](dashboard.md) for what each page does.

## Build this documentation site locally

```bash
poetry run mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000). The `docs/hooks.py`
script copies the executed notebooks into the docs tree at build time so they
render in the **Notebooks** section.
