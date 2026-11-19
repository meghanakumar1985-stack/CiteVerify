"""Step 3: Extract scientific claims from citation sentences (adapted from ClaimScope)."""
from __future__ import annotations
import re
from typing import List

from core.models import CitationClaim

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r"[;:]\s+|\s+(?:whereas|however|which|while)\s+")

CLAIM_CUES = (
    "found",
    "shows",
    "suggests",
    "indicates",
    "demonstrates",
    "argues",
    "concludes",
    "states",
    "associated",
    "insufficient",
    "supports",
    "does not",
    "cannot",
    "should",
    "outperforms",
    "improves",
    "significantly",
    "reveals",
    "proves",
    "fails",
    "achieves",
    "reduces",
    "increases",
    "leads to",
    "consists in",
    "scales",
)

# Matches citation brackets like [1], [1, 2], [1-4], (Vaswani et al., 2017), (2019)
CITATION_PATTERNS = [
    re.compile(r"\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]"),
    re.compile(r"\(\s*[A-Z][a-zA-Z\s&.,]+,\s*\d{4}[a-z]?\s*\)"),
    re.compile(r"\(\s*\d{4}\s*\)"),
    re.compile(r"\[[A-Za-z\s.,0-9]+\]"),
]

LEADING_PHRASES = (
    "in their view,",
    "according to the authors,",
    "according to the author,",
    "the authors argue that",
    "the author argues that",
    "prior studies suggest that",
    "recent experiments show that",
    "studies have shown that",
    "previous work indicates that",
    "research shows that",
    "results show that",
    "it was observed that",
    "we found that",
    "the authors observed that",
)


def remove_citation_markers(text: str) -> str:
    """Removes bracketed citations like [1], [1, 2], (Author, 2020) from the sentence."""
    cleaned = text
    for pattern in CITATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Clean up any leftover double spaces or orphaned punctuation
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    return " ".join(cleaned.split())


def clean_claim_text(text: str) -> str:
    """Removes framing phrases and citations to isolate the core scientific claim."""
    cleaned = remove_citation_markers(text)
    lowered = cleaned.lower()

    for phrase in LEADING_PHRASES:
        if lowered.startswith(phrase):
            cleaned = cleaned[len(phrase) :].strip()
            break

    parts = [p.strip() for p in CLAUSE_SPLIT.split(cleaned) if p.strip()]
    if parts:
        # Prefer the clause with the highest density of claim verbs/cues
        ranked = sorted(
            parts,
            key=lambda part: (sum(cue in part.lower() for cue in CLAIM_CUES), len(part)),
            reverse=True,
        )
        cleaned = ranked[0]

    return cleaned.strip(" \"'")


def is_claim_bearing(text: str) -> bool:
    """Returns True if the text contains verbs/phrases typical of scientific claims."""
    lowered = text.lower()
    return any(cue in lowered for cue in CLAIM_CUES)


def extract_claims(citation_claims: List[CitationClaim]) -> List[CitationClaim]:
    """Processes citation sentences and populates claim_text with the clean claim."""
    extracted: List[CitationClaim] = []
    for item in citation_claims:
        cleaned = clean_claim_text(item.sentence)
        final_claim = cleaned if len(cleaned.split()) >= 4 else remove_citation_markers(item.sentence)
        extracted.append(
            CitationClaim(
                sentence=item.sentence,
                target_ref_ids=item.target_ref_ids,
                claim_text=final_claim,
                section=item.section,
            )
        )
    return extracted
