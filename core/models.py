"""Data structures for CiteVerify."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BiblioEntry:
    """Represents a bibliography entry parsed from TEI XML."""
    ref_id: str                      # e.g. "b0"
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    raw_text: str = ""


@dataclass
class CitationClaim:
    """Represents a citation sentence and its extracted claim."""
    sentence: str
    target_ref_ids: List[str]       # e.g. ["b0"]
    claim_text: str                 # Extracted core claim
    section: Optional[str] = None


@dataclass
class EvidencePassage:
    """Represents a chunk of evidence from the cited paper."""
    passage_id: str
    ref_id: str
    text: str
    score: float = 0.0


@dataclass
class VerificationItem:
    """Final result row for a claim and its citation."""
    claim: str
    citation: str                   # Cited reference label or title
    cited_title: str
    cited_authors: str
    cited_year: str
    cited_doi: str
    evidence_passage: str           # Best retrieved evidence snippet
    evidence_score: float           # TF-IDF similarity score or confidence float
    verdict: str                    # Supported / Partially supported / Contradicted / Potentially outdated / Unsupported
    flag: str                       # Verified / Misleading Citation / Non-Existent Reference / Metadata Mismatch / Outdated Reference / Unverified
    confidence: str = "High"        # High / Medium / Low
    needs_review: bool = False      # True if requiring human review per Hackathon rubric
    source_url: str = ""            # Direct clickable link (DOI, Semantic Scholar, arXiv)
    details: str = ""               # Explanation or extra context
