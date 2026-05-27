# Engineering Checklist

Use this checklist before opening a PR or asking for a commit.

## Scope discipline

- Keep changes scoped to the requested task.
- If extra fixes are needed, separate them into a follow-up commit.

## Contract checks

- Verify new tests match the command/function contract.
- For non-validation commands, do not assert validation-specific outcomes.
- Add fallback paths for partial/missing data in UI formatters.

## Test strategy

- Add regression tests for every bug fixed.
- Cover helper-level logic and at least one page-level edge case for dashboards.
- Prefer deterministic tests with small fixtures.

## Quality gate

Run all three and confirm success:

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest -q
```

## Data/metrics integrity

- Ensure all displayed metrics come from computed tables only.
- Avoid assumptions that can silently drop or misclassify rows (e.g., trim/normalize strings before comparisons).
