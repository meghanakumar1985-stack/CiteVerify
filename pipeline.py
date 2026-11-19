"""End-to-end scientific claim and citation verification orchestrator."""
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv

from core.models import BiblioEntry, CitationClaim, EvidencePassage, VerificationItem
from core.pdf_parser import parse_pdf
from core.claim_extractor import extract_claims
from core.evidence_retriever import retrieve_cited_paper
from core.passage_ranker import rank_passages
from core.claim_classifier import classify_claim
from core.reference_checker import evaluate_reference_flag

load_dotenv()


def run_pipeline(
    pdf_bytes: bytes,
    use_stub: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end claim and citation verification workflow:
    - Parses document PDF bytes directly using lightweight PyMuPDF extraction.
    - Resolves in-text citation markers to bibliography entries.
    - Isolates factual assertions and target citation contexts.
    - Retrieves primary source evidence via scholarly APIs (Semantic Scholar, CrossRef, arXiv).
    - Ranks evidence passages using TF-IDF cosine similarity.
    - Performs dual NLI stance classification (Supported, Partially supported, Contradicted, Potentially outdated, Unsupported).
    - Audits bibliographic metadata integrity for phantom citations and drift.
    - Returns structured DataFrame and runtime diagnostics.
    """
    diagnostics: Dict[str, Any] = {
        "engine_mode": "native_pdf_parser",
        "used_stub": use_stub,
        "total_references": 0,
        "total_citations": 0,
        "warnings": [],
    }

    # 1. Parse PDF directly (in-memory, no external GROBID service required)
    bib_map, raw_claims = parse_pdf(pdf_bytes, use_stub=use_stub)
    diagnostics["total_references"] = len(bib_map)
    diagnostics["total_citations"] = len(raw_claims)

    if not raw_claims:
        return pd.DataFrame(), diagnostics

    # 3. Extract Clean Claims
    claims = extract_claims(raw_claims)

    # Cache retrieved evidence only for references that are actually cited
    needed_ref_ids = set()
    for item in claims:
        needed_ref_ids.update(item.target_ref_ids)

    retrieved_papers: Dict[str, Dict[str, Any]] = {}
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_one(r_id: str):
        entry = bib_map.get(r_id, BiblioEntry(ref_id=r_id, title=f"Reference {r_id}"))
        return r_id, retrieve_cited_paper(entry, use_stub=diagnostics["used_stub"])

    with ThreadPoolExecutor(max_workers=6) as executor:
        for r_id, p_info in executor.map(_fetch_one, needed_ref_ids):
            retrieved_papers[r_id] = p_info

    # 5, 6, 7. Rank passages, Classify Claim, and Flag Reference concurrently
    items: List[VerificationItem] = []
    selected_claims = claims[:8]

    def _verify_one_target(pair):
        item, ref_id = pair
        bib_entry = bib_map.get(
            ref_id,
            BiblioEntry(ref_id=ref_id, title=f"Reference {ref_id}"),
        )
        paper_info = retrieved_papers.get(
            ref_id,
            {"exists": False, "abstract": "", "authors": [], "year": None},
        )

        ranked_passages = rank_passages(
            claim_text=item.claim_text,
            evidence_text=paper_info.get("abstract", ""),
            ref_id=ref_id,
            top_k=3,
        )

        cited_year_str = str(bib_entry.year or paper_info.get("year") or "")

        verdict, details, confidence = classify_claim(
            claim_text=item.claim_text,
            passages=ranked_passages,
            cited_year=cited_year_str,
        )

        flag, flag_reason, needs_review = evaluate_reference_flag(bib_entry, paper_info, verdict)

        best_evidence = ranked_passages[0].text if ranked_passages else "No evidence found."
        best_score = ranked_passages[0].score if ranked_passages else 0.0

        authors_str = ", ".join(bib_entry.authors) if bib_entry.authors else "Unknown"

        source_url = paper_info.get("url") or ""
        if not source_url:
            if bib_entry.doi:
                source_url = f"https://doi.org/{bib_entry.doi}"
            elif bib_entry.title:
                import urllib.parse
                source_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(bib_entry.title)}"

        return VerificationItem(
            claim=item.claim_text,
            citation=f"[{ref_id}] {bib_entry.title or 'Unknown Title'}",
            cited_title=bib_entry.title,
            cited_authors=authors_str,
            cited_year=cited_year_str,
            cited_doi=str(bib_entry.doi or ""),
            evidence_passage=best_evidence,
            evidence_score=best_score,
            verdict=verdict,
            flag=flag,
            confidence=confidence,
            needs_review=needs_review,
            source_url=source_url,
            details=f"{details} | {flag_reason}",
        )

    targets = []
    for item in selected_claims:
        for ref_id in item.target_ref_ids:
            targets.append((item, ref_id))

    with ThreadPoolExecutor(max_workers=5) as executor:
        items = list(executor.map(_verify_one_target, targets))

    # Format table for report adhering to Problem Statement 03 fields
    df = pd.DataFrame(
        [
            {
                "Claim": it.claim,
                "Citation": it.citation,
                "Verdict": it.verdict,
                "Confidence": it.confidence,
                "Review Status": "[!] Review Required" if it.needs_review else "[✓] Verified",
                "Flag": it.flag,
                "Source Link": it.source_url,
                "Similarity": round(it.evidence_score, 3),
                "Top Evidence Passage": it.evidence_passage,
                "Details": it.details,
            }
            for it in items
        ]
    )

    return df, diagnostics
