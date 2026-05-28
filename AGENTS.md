# AGENTS.md

This file gives coding agents clear instructions for contributing to this repository.

## Project mission

Build a customer experience intelligence platform around CFPB consumer complaint data.

The project should demonstrate applied data science skills across:

- data ingestion;
- data cleaning;
- NLP;
- topic discovery;
- interpretable risk scoring;
- supervised modelling;
- reporting;
- production-style Python engineering.

## General rules

1. Prefer complete, maintainable implementations over small demos.
2. Keep the public API stable unless a change is documented.
3. Use English for all code, comments, documentation, and commit messages.
4. Use type hints for all public functions.
5. Add docstrings to public classes and functions.
6. Add inline comments where the logic is not obvious.
7. Keep functions small and testable.
8. Do not silently swallow exceptions.
9. Do not add heavy dependencies unless they are justified in the roadmap or README.
10. Do not commit raw CFPB data or generated reports unless they are explicitly small sample fixtures.

## Python style

Use:

- Python 3.10+;
- Poetry for dependency management;
- pandas and NumPy for tabular work;
- scikit-learn for baseline NLP and ML;
- pydantic for configuration and validation where useful;
- click for CLI commands;
- rich for user-facing terminal output.

Run before finishing any task:

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
```

## Data handling rules

The CFPB dataset is public, but complaint narratives can still describe sensitive situations.

Agents must:

- avoid attempts to identify consumers;
- avoid enriching rows with personal data;
- avoid adding geolocation precision beyond what CFPB already publishes;
- preserve complaint IDs for traceability;
- keep raw and processed data separate;
- avoid unsupported claims about companies without normalisation or caveats.

## Modelling rules

The first implementation uses transparent methods. Do not replace them without preserving explainability.

For topic modelling:

- keep TF-IDF + NMF as the baseline;
- if adding embeddings or BERTopic, keep the baseline available;
- save topic terms and topic assignments as separate outputs.

For severity/risk scoring:

- keep the heuristic score transparent;
- document every scoring component;
- do not present risk score as a legal judgement;
- add tests when scoring rules change.

For supervised modelling:

- use time-aware train/test splits when predicting future outcomes;
- report class imbalance;
- include precision, recall, F1, ROC-AUC where applicable;
- calibrate probabilities when the score is used for prioritisation;
- include model limitations.

## Reporting rules

Reports should be useful for business stakeholders.

A good report should answer:

- what changed;
- where the problem is concentrated;
- which customers or products are affected;
- what the likely operational action is;
- what evidence supports the conclusion.

Numerical claims must come from computed tables. Do not let an LLM invent numbers.

## Suggested task order for agents

1. Improve tests around schema normalisation.
2. Add EDA charts.
3. Add a polished analysis notebook.
4. Add supervised model baselines.
5. Add DuckDB storage.
6. Add Streamlit dashboard.
7. Add CI workflow.
8. Add documentation site.

## Files and responsibilities

```text
src/cx_complaints/data/
```

Data ingestion, API clients, schema normalisation.

```text
src/cx_complaints/preprocessing/
```

Text cleaning and data preparation.

```text
src/cx_complaints/features/
```

Feature engineering for text and metadata.

```text
src/cx_complaints/nlp/
```

Topic modelling, text representations, future embeddings.

```text
src/cx_complaints/modelling/
```

Risk scoring and future supervised models.

```text
src/cx_complaints/reports/
```

Report generation and export logic.

```text
tests/
```

Unit tests. Every non-trivial new function should have tests.

## Definition of done

A task is done when:

- the implementation works from the CLI or documented API;
- public functions have type hints and docstrings;
- tests cover the main behaviour;
- existing tests pass;
- the README or ROADMAP is updated if the workflow changes;
- generated outputs are not committed unless they are intended fixtures.
