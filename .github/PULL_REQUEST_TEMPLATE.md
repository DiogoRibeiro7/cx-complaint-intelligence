## Summary

Describe what changed and why.

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] Test-only

## Scope

- Affected area(s): data / preprocessing / features / NLP / modelling / reports / CLI / docs
- Public API changes: yes/no
- Backward-compatible: yes/no

## Validation

- [ ] `poetry run ruff check .`
- [ ] `poetry run mypy src tests`
- [ ] `poetry run pytest`

Paste key outputs or a short summary:

```text
# validation notes
```

## Data and modelling checks

- [ ] No raw sensitive complaint narratives were newly committed.
- [ ] `complaint_id` traceability preserved where relevant.
- [ ] Topic baseline (TF-IDF + NMF) remains available when touching topic modelling.
- [ ] Risk-score logic changes are documented and tested when applicable.

## Evidence

Attach screenshots, report snippets, or metric tables when relevant.

## Related issues

Closes #
