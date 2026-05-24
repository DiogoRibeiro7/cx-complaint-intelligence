#!/usr/bin/env bash
set -euo pipefail

poetry run cx-complaints fetch \
  --output data/raw/cfpb_sample.csv \
  --limit 1000 \
  --has-narrative true \
  --date-min 2024-01-01

poetry run cx-complaints analyze \
  --input data/raw/cfpb_sample.csv \
  --output-dir reports/sample \
  --topics 8
