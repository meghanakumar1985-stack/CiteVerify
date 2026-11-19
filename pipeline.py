"""CiteVerify Pipeline: Connects all steps from PDF upload to verification report."""
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv

from core.models import BiblioEntry, CitationClaim, EvidencePassage, VerificationItem
from core.grobid_client import process_pdf, check_grobid_health, PUBLIC_GROBID_URL
from core.tei_parser import parse_tei
from core.claim_extractor import extract_claims
from core.evidence_retriever import retrieve_cited_paper
from core.passage_ranker import rank_passages
from core.claim_classifier import classify_claim
from core.reference_checker import evaluate_reference_flag

load_dotenv()


def run_pipeline(
    pdf_bytes: bytes,
    grobid_url: str | None = None,
    use_stub: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the CiteVerify end-to-end pipeline:
    1. Parse PDF with GROBID into TEI XML.
    2. Parse TEI XML for in-text citations and bibliography.
    3. Extract claims from citation sentences.
    4. Retrieve cited paper abstracts (Semantic Scholar / CrossRef).
    5. Rank evidence passages with TF-IDF cosine similarity.
    6. Classify stance (Supported / Partially supported / Contradicted / Not found).
    7. Flag reference status (Verified / Misleading Citation / Metadata Mismatch / etc.).
    8. Format into pandas DataFrame.
    """
    diagnostics: Dict[str, Any] = {
        "grobid_healthy": False,
        "used_stub": use_stub,
        "total_references": 0,
        "total_citations": 0,
        "warnings": [],
    }

    effective_grobid_url = grobid_url or os.getenv("GROBID_URL", "http://localhost:8070")
    if not use_stub:
        if check_grobid_health(effective_grobid_url):
            diagnostics["grobid_healthy"] = True
            diagnostics["active_grobid_url"] = effective_grobid_url
        elif check_grobid_health(PUBLIC_GROBID_URL):
            diagnostics["grobid_healthy"] = True
            diagnostics["active_grobid_url"] = PUBLIC_GROBID_URL
            diagnostics["warnings"].append(
                f"Local GROBID at {effective_grobid_url} was offline. Connected to online GROBID server ({PUBLIC_GROBID_URL})."
            )
        else:
            diagnostics["grobid_healthy"] = False
            diagnostics["warnings"].append(
                f"GROBID is not reachable locally or remotely. Using demo sample data."
            )
            use_stub = True
            diagnostics["used_stub"] = True
    else:
        diagnostics["grobid_healthy"] = False
        diagnostics["active_grobid_url"] = "stub"

    # 1. GROBID PDF Extraction
    try:
        if use_stub:
            tei_xml = process_pdf(pdf_bytes, use_stub=True)
            diagnostics["used_stub"] = True
        else:
            active_url = diagnostics.get("active_grobid_url", effective_grobid_url)
            tei_xml = process_pdf(pdf_bytes, grobid_url=active_url, use_stub=False)
    except Exception as e:
        diagnostics["warnings"].append(f"GROBID extraction error: {e}. Falling back to sample data.")
        tei_xml = process_pdf(pdf_bytes, use_stub=True)
        diagnostics["used_stub"] = True

    # 2. Parse TEI XML
    bib_map, raw_claims = parse_tei(tei_xml)
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

    # 5, 6, 7. Rank passages, Classify Claim, and Flag Reference
    items: List[VerificationItem] = []

    for item in claims:
        for ref_id in item.target_ref_ids:
            bib_entry = bib_map.get(
                ref_id,
                BiblioEntry(ref_id=ref_id, title=f"Reference {ref_id}"),
            )
            paper_info = retrieved_papers.get(
                ref_id,
                {"exists": False, "abstract": "", "authors": [], "year": None},
            )

            # Rank passages
            ranked_passages = rank_passages(
                claim_text=item.claim_text,
                evidence_text=paper_info.get("abstract", ""),
                ref_id=ref_id,
                top_k=3,
            )

            # Classify claim
            verdict, details = classify_claim(item.claim_text, ranked_passages)

            # Flag reference
            flag, flag_reason = evaluate_reference_flag(bib_entry, paper_info, verdict)

            best_evidence = ranked_passages[0].text if ranked_passages else "No evidence found."
            best_score = ranked_passages[0].score if ranked_passages else 0.0

            authors_str = ", ".join(bib_entry.authors) if bib_entry.authors else "Unknown"

            items.append(
                VerificationItem(
                    claim=item.claim_text,
                    citation=f"[{ref_id}] {bib_entry.title or 'Unknown Title'}",
                    cited_title=bib_entry.title,
                    cited_authors=authors_str,
                    cited_year=str(bib_entry.year or ""),
                    cited_doi=str(bib_entry.doi or ""),
                    evidence_passage=best_evidence,
                    evidence_score=best_score,
                    verdict=verdict,
                    flag=flag,
                    details=f"{details} | {flag_reason}",
                )
            )

    # Format table for report
    df = pd.DataFrame(
        [
            {
                "Claim": it.claim,
                "Citation": it.citation,
                "Verdict": it.verdict,
                "Flag": it.flag,
                "Similarity": round(it.evidence_score, 3),
                "Top Evidence Passage": it.evidence_passage,
                "Details": it.details,
            }
            for it in items
        ]
    )

    return df, diagnostics
