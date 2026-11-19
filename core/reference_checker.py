"""Step 7: Flag misleading or non-existent references (adapted from RefChecker concepts)."""
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple
from core.models import BiblioEntry


def _extract_surnames(names: List[str]) -> set[str]:
    """Extracts lowercase last names from full author strings."""
    surnames = set()
    for name in names:
        cleaned = re.sub(r"[^\w\s-]", "", name).strip()
        parts = cleaned.split()
        if parts:
            surnames.add(parts[-1].lower())
    return surnames


def check_metadata_consistency(
    entry: BiblioEntry,
    retrieved_meta: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Compares cited paper metadata (from PDF bibliography) against
    retrieved database metadata (from Semantic Scholar, CrossRef, arXiv).
    Returns: (is_consistent, flag_type, reason)
    """
    if not retrieved_meta.get("exists", False):
        return (
            False,
            "Non-Existent Reference",
            "Paper could not be found in scholarly databases (possible fabricated or hallucinated citation).",
        )

    # 1. Author Overlap Check (RefChecker policy: flag if overlap is low)
    cited_surnames = _extract_surnames(entry.authors)
    db_surnames = _extract_surnames(retrieved_meta.get("authors", []))

    if cited_surnames and db_surnames:
        overlap = cited_surnames.intersection(db_surnames)
        overlap_ratio = len(overlap) / max(len(cited_surnames), 1)
        if overlap_ratio < 0.40:
            cited_str = ", ".join(entry.authors[:2])
            db_str = ", ".join(retrieved_meta.get("authors", [])[:2])
            return (
                False,
                "Metadata Mismatch (Authors)",
                f"Author mismatch: cited authors ({cited_str}) do not match database records ({db_str}).",
            )

    # 2. Publication Year Check
    if entry.year and retrieved_meta.get("year"):
        try:
            diff = abs(int(entry.year) - int(retrieved_meta["year"]))
            if diff > 2:
                return (
                    False,
                    "Metadata Mismatch (Year)",
                    f"Year mismatch: cited year ({entry.year}) differs from database ({retrieved_meta['year']}).",
                )
        except ValueError:
            pass

    # 3. Title Check (detect if DOI pointed to completely different paper)
    cited_title = entry.title.lower().strip()
    db_title = retrieved_meta.get("title", "").lower().strip()
    if cited_title and db_title and len(cited_title) > 10 and len(db_title) > 10:
        words_cited = set(re.findall(r"\b[a-z]{3,}\b", cited_title))
        words_db = set(re.findall(r"\b[a-z]{3,}\b", db_title))
        if words_cited and words_db:
            title_overlap = len(words_cited.intersection(words_db)) / max(len(words_cited), 1)
            if title_overlap < 0.25:
                return (
                    False,
                    "Metadata Mismatch (Title)",
                    f"Title mismatch: cited '{entry.title[:40]}...' differs from database '{retrieved_meta.get('title','')[:40]}...'",
                )

    return True, "Verified", "Paper exists and metadata matches database records."


def evaluate_reference_flag(
    entry: BiblioEntry,
    retrieved_meta: Dict[str, Any],
    verdict: str,
) -> Tuple[str, str]:
    """
    Assigns an overall verification flag to the citation:
    - 'Verified'
    - 'Misleading Citation' (Contradicted evidence)
    - 'Non-Existent Reference' (Not found in any database)
    - 'Metadata Mismatch' (Author, Year, or Title conflict)
    - 'Unverified / Weak Evidence' (Evidence abstract lacks support)
    """
    if not retrieved_meta.get("exists", False):
        return (
            "Non-Existent Reference",
            "Cited paper was not found in Semantic Scholar, CrossRef, or arXiv.",
        )

    is_meta_ok, flag_type, meta_reason = check_metadata_consistency(entry, retrieved_meta)
    if not is_meta_ok:
        return flag_type, meta_reason

    if verdict == "Contradicted":
        return (
            "Misleading Citation",
            "Cited evidence directly refutes or contradicts the claim made in the text.",
        )

    if verdict == "Not found":
        return (
            "Unverified / Weak Evidence",
            "Cited paper was found in databases, but its text does not substantiate this claim.",
        )

    return "Verified", "Paper exists, metadata matches, and evidence substantiates the claim."
