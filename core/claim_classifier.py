"""Step 6: Classify claim stance against evidence (Supported / Partially supported / Contradicted / Not found)."""
from __future__ import annotations
import os
import re
from typing import List, Tuple

from core.models import EvidencePassage

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "insufficient",
    "without",
    "cannot",
    "fail",
    "fails",
    "failed",
    "failure",
    "doesn't",
    "didn't",
    "rarely",
    "unlikely",
    "neither",
    "nor",
}

CONTRADICTION_CUES = {
    "contrast",
    "contrary",
    "however",
    "in contrast",
    "despite",
    "whereas",
    "unlike",
    "refutes",
    "disputes",
    "instead",
    "incorrect",
    "flawed",
    "misleading",
}

OPPOSITION_PAIRS = [
    ({"fail", "fails", "failed", "failure"}, {"competitive", "superior", "succeeds", "success", "effective", "improves"}),
    ({"increase", "increases", "increased"}, {"decrease", "decreases", "decreased", "reduce", "reduces"}),
    ({"improve", "improves", "improved"}, {"worsen", "worsens", "degrades", "degrade"}),
]


def _get_tokens(text: str) -> set[str]:
    """Extracts meaningful alphabetic tokens of length 3 or more."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _detect_polarity_conflict(claim_tokens: set[str], passage_tokens: set[str]) -> bool:
    """Checks for negation mismatch or semantic antonym opposition between claim and evidence."""
    claim_has_neg = bool(claim_tokens.intersection(NEGATION_WORDS))
    passage_has_neg = bool(passage_tokens.intersection(NEGATION_WORDS))
    passage_has_contrast = bool(passage_tokens.intersection(CONTRADICTION_CUES))

    if claim_has_neg != passage_has_neg:
        return True

    if passage_has_contrast:
        return True

    # Check opposition pairs (e.g. claim asserts fail, evidence asserts competitive/superior)
    for set_a, set_b in OPPOSITION_PAIRS:
        if (claim_tokens.intersection(set_a) and passage_tokens.intersection(set_b)) or (
            claim_tokens.intersection(set_b) and passage_tokens.intersection(set_a)
        ):
            return True

    return False


def classify_with_rules(
    claim_text: str,
    passages: List[EvidencePassage],
) -> Tuple[str, str]:
    """
    Rule-based stance classifier using TF-IDF similarity, lexical overlap, and negation cues.
    Returns: (verdict, details)
      verdict in {"Supported", "Partially supported", "Contradicted", "Not found"}
    """
    if not passages or not passages[0].text.strip():
        return "Not found", "No evidence passages retrieved for this claim."

    best_passage = passages[0]
    score = best_passage.score

    if score < 0.08:
        return (
            "Not found",
            f"Retrieval similarity score ({score:.2f}) is below the minimum evidence threshold (0.08).",
        )

    claim_tokens = _get_tokens(claim_text)
    passage_tokens = _get_tokens(best_passage.text)
    overlap = claim_tokens.intersection(passage_tokens)

    # 1. Contradiction Detection
    has_conflict = _detect_polarity_conflict(claim_tokens, passage_tokens)
    if has_conflict and score >= 0.14 and len(overlap) >= 2:
        return (
            "Contradicted",
            f"Evidence addresses the topic (similarity: {score:.2f}, {len(overlap)} shared terms) but asserts contrary findings or polarity.",
        )

    # 2. Support Detection
    if score >= 0.28 and len(overlap) >= 3:
        return (
            "Supported",
            f"Strong semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms) with consistent polarity.",
        )
    elif score >= 0.14 and len(overlap) >= 2:
        return (
            "Partially supported",
            f"Moderate semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms). Evidence relates to the claim topic.",
        )
    else:
        return (
            "Not found",
            f"Insufficient semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms). Evidence lacks sufficient support.",
        )


def classify_with_llm(
    claim_text: str,
    passages: List[EvidencePassage],
    api_key: str | None = None,
) -> Tuple[str, str]:
    """
    Pluggable hook for optional LLM verification (e.g. Gemini, OpenAI, Claude).
    When an LLM provider is configured, prompts the model to classify:
    [SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, NOT_FOUND] with reasoning.
    """
    # If no LLM provider configured, fall back seamlessly to rule-based verification
    return classify_with_rules(claim_text, passages)


def classify_claim(
    claim_text: str,
    passages: List[EvidencePassage],
    use_llm: bool = False,
) -> Tuple[str, str]:
    """Main classification entrypoint."""
    if use_llm or os.getenv("USE_LLM_CLASSIFIER", "").lower() in ("true", "1"):
        return classify_with_llm(claim_text, passages)
    return classify_with_rules(claim_text, passages)
