# Data ethics

CFPB consumer complaints are public, but the narratives still describe
sensitive situations &mdash; financial distress, identity theft, eviction,
harassment. The project handles them under a small set of explicit rules.

## What this project will not do

- **No re-identification of consumers.** Names, addresses, and account
  numbers are never extracted, even though some narratives mention them.
  The cleaning step explicitly removes URLs, emails, and money values.
- **No enrichment with personal data.** The pipeline only uses the columns
  CFPB publishes. No third-party data is joined onto complaints.
- **No public "naughty list" of companies.** Company comparisons are
  presented with a minimum-volume guardrail (the smaller-sample distortions
  it prevents are illustrated in notebook 05) and with explicit caveats
  about composition bias.
- **No legal claims.** The risk score is a prioritisation signal. It does
  not appear in any artefact without an accompanying disclaimer.

## What stays inside the repo and what does not

| Artefact | Committed? | Why |
|---|---|---|
| `data/raw/cfpb_sample.csv` | **Yes** (small fixture) | Lets a reviewer reproduce the pipeline without API access. |
| `reports/sample/complaint_scores.csv` | No (gitignored) | Contains the raw narratives joined with the scoring columns &mdash; even though CFPB publishes the narratives, we do not republish them in this repo. |
| `reports/sample/*.csv` (summaries) | Yes | Aggregated tables only; no narratives. |
| `reports/sample/executive_summary.md` | Yes | Stakeholder-ready summary, no narratives. |
| `reports/notebook/<id>/*` | No (gitignored) | Per-notebook artefacts. Regenerate locally. |
| Run manifests (`run_manifest.json`) | No (gitignored) | Local audit trail; not stable across machines. |

## Narrative redaction

CFPB pre-processes published narratives by replacing PII with `XXXX`. The
project keeps a generic `[REDACTED]` marker for those runs because the
**presence** of redaction is informative about complaint structure, but the
specific values are not retained.

## When in doubt

The contributor agreement at
[AGENTS.md](https://github.com/diogoribeiro/cx-complaint-intelligence/blob/main/AGENTS.md)
lays out the data-handling rules that govern every contribution to this
repo. Read it before adding features that touch raw narratives or company
identifiers.
