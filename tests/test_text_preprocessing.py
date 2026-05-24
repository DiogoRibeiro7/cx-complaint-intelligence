from __future__ import annotations

import pandas as pd

from cx_complaints.preprocessing.text import add_clean_text_column, normalize_text


def test_normalize_text_handles_redactions_and_whitespace() -> None:
    text = "I paid $250 to XXXX Bank.\n\nEmail me at test@example.com"

    result = normalize_text(text)

    assert "[MONEY]" in result
    assert "[REDACTED]" in result
    assert "[EMAIL]" in result
    assert "\n" not in result


def test_add_clean_text_column() -> None:
    frame = pd.DataFrame({"complaint_what_happened": ["Hello   XXXX"]})

    result = add_clean_text_column(frame)

    assert result.loc[0, "clean_text"] == "Hello [REDACTED]"
