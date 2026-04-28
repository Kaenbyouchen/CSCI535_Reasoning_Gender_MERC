"""Exact-token gender lexicon for attribution + plots (no torch / no MELD imports)."""
from __future__ import annotations

import re

# Exact-token match only — no substring rules ("he" must not match "the").
_gender_words = frozenset({
    "he", "she", "her", "his", "him", "himself", "herself",
    "man", "men", "woman", "women", "boy", "girl", "boys", "girls", "male", "female", "males", "females",
    "guy", "guys", "husband", "wife", "mother", "father", "mom", "dad", "son", "daughter", "sister", "brother",
    "lady", "ladies", "gentleman", "gentlemen", "girly", "boyish",
    "feminine", "masculine", "nonbinary", "non-binary",
    "gender", "gendered", "womanhood", "manhood",
    "mr", "mrs", "ms", "miss", "sir", "madam", "queer",
})


def is_gender_token(s: str) -> bool:
    """True iff normalized token equals an entry in the lexicon (exact match only)."""
    t = s.replace("Ġ", "").replace("▁", "").strip().lower()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    if not t:
        return False
    if t.endswith("'s") or t.endswith("'re") or t.endswith("'ve") or t.endswith("'ll") or t.endswith("'d"):
        t = re.sub(r"'(?:s|re|ve|ll|d)$", "", t)
    return t in _gender_words


def gender_lexicon_words() -> frozenset[str]:
    """Read-only set of normalized forms treated as gender-related (for docs / debugging)."""
    return _gender_words
