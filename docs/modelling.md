# Modelling

The project deliberately starts with **transparent baselines**. The
contributor agreement requires those baselines to remain available even when
heavier models are added later.

## Risk score

The risk score is a small additive function of seven interpretable
components. Higher means more attention is warranted.

| Component | Weight | When it adds |
|---|---|---|
| Not timely (`timely != "yes"`) | `+2.0` | Always when timely is anything other than "yes". |
| High-risk product | `+1.5` | Product is in the curated `HIGH_RISK_PRODUCTS` set (debt collection, mortgage, credit reporting, etc.). |
| High-risk issue terms | `+1.5` | Issue or sub-issue mentions identity theft, fraud, foreclosure, lawsuit, threat, scam, etc. |
| Weak response code | `+0.75` | Response is `closed`, `closed with explanation`, or `in progress`. |
| Vulnerable consumer tag | `+1.0` | CFPB tags include "Older American" or "Servicemember". |
| Urgency terms in narrative | `+0.75` × (count up to 3) | Narrative contains words like `fraud`, `lawsuit`, `harassment`, `eviction`. |
| Money language | `+0.5` | Narrative mentions `[MONEY]`, `$`, `dollar`, `fee`, `refund`, etc. |
| Long narrative (>= 250 words) | `+0.5` | Long narratives correlate with detailed grievances. |

The numeric score is bucketed into `low` (< 2.5), `medium` (2.5–5.0), and
`high` (>= 5.0).

!!! warning "The score is not a legal finding"
    It exists to triage analyst attention. Use it to decide what to read
    first, not to make public claims about companies.

The full implementation is in [`modelling/risk.py`](https://github.com/diogoribeiro/cx-complaint-intelligence/blob/main/src/cx_complaints/modelling/risk.py)
and is covered by unit tests in
[`tests/test_risk.py`](https://github.com/diogoribeiro/cx-complaint-intelligence/blob/main/tests/test_risk.py).

## Topic discovery

TF-IDF + NMF over the cleaned narratives. Default configuration:

- `lowercase=True`, English stopwords, n-grams (1, 2), `min_df=5`, `max_df=0.85`.
- `NMF(n_components=8, init="nndsvda", max_iter=500, random_state=42)`.

NMF is chosen for two reasons:

1. **Interpretability.** Each topic is a non-negative combination of
   vocabulary terms; the top terms read as a topic label.
2. **Stability.** With a fixed seed and `nndsvda` initialisation, repeated
   runs on the same corpus produce the same topics.

The pipeline produces a `topic_terms.csv` with the highest-weighted terms per
topic and assigns each complaint to its dominant topic.

## Supervised baselines

Two interpretable models, both trained over TF-IDF features:

- **Logistic regression** (`class_weight="balanced"`).
- **Linear SVM** (`class_weight="balanced"`).

The label is the heuristic `risk_level == "high"`. That is **not** a
validated outcome &mdash; the baseline therefore mostly recovers the
heuristic's vocabulary. It is useful as a triage assistant, not as a
substitute for analyst review. Replacing the label with a reviewed outcome
is the natural next step once enough labelled examples exist.

### Time-aware splits

`time_aware_train_test_split` orders rows by `date_received` and slices the
trailing fraction as the test set. Random splits would leak future
information.

### Metrics

`evaluate_binary_classifier` returns precision, recall, F1, and ROC-AUC
(when the target is non-degenerate). Threshold sweeps in notebook 04 show
the operational trade-off between precision and the analyst-review budget.

## What is intentionally not here

- **No deep learning models** in the baseline. The contributor agreement
  permits adding embeddings or BERTopic later, but only as additions
  alongside the transparent baselines.
- **No automated company ranking.** Company comparisons are presented with a
  minimum-volume guardrail and explicit narrative caveats &mdash; never
  surfaced as a "naughty list".
- **No automated remediation recommendations.** The pipeline surfaces
  evidence; the operational decisions stay with humans.
