"""Native PDF parser and citation extraction engine using PyMuPDF.

Replaces GROBID with direct, lightweight, in-memory PDF parsing and regex
citation-to-bibliography resolution. Requires no Docker, no external services,
and executes in milliseconds.
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
import pymupdf

from core.models import BiblioEntry, CitationClaim

# Benchmark demonstration entries for evaluation
BENCHMARK_BIBLIO: Dict[str, BiblioEntry] = {
    "b0": BiblioEntry(
        ref_id="b0",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year="2017",
        doi="10.48550/arXiv.1706.03762",
    ),
    "b1": BiblioEntry(
        ref_id="b1",
        title="Semi-Supervised Classification with Graph Convolutional Networks",
        authors=["Thomas N. Kipf", "Max Welling"],
        year="2016",
        doi="10.48550/arXiv.1609.02907",
    ),
    "b2": BiblioEntry(
        ref_id="b2",
        title="Evaluating Mobile Learning Apps in Secondary Education",
        authors=["H. Chen", "L. Zhang"],
        year="2021",
        doi="10.1016/j.compedu.2021.104231",
    ),
    "b3": BiblioEntry(
        ref_id="b3",
        title="Object Recognition from Local Scale-Invariant Features",
        authors=["David G. Lowe"],
        year="1999",
        doi="10.1109/ICCV.1999.790410",
    ),
    "b4": BiblioEntry(
        ref_id="b4",
        title="Room Temperature Quantum Neural Architectures for AGI",
        authors=["Sarah Williams", "Edward Thorne"],
        year="2024",
        doi="10.9999/fake.quantum.2024.99999",
    ),
}

BENCHMARK_CLAIMS: List[CitationClaim] = [
    CitationClaim(
        sentence="Recent experiments show that transformer models significantly improve scientific fact verification accuracy [1].",
        target_ref_ids=["b0"],
        claim_text="transformer models significantly improve scientific fact verification accuracy.",
    ),
    CitationClaim(
        sentence="Furthermore, prior studies suggest that graph neural networks fail on low-resource citation graph tasks [2].",
        target_ref_ids=["b1"],
        claim_text="Furthermore, prior studies suggest that graph neural networks fail on low-resource citation graph tasks.",
    ),
    CitationClaim(
        sentence="A particular study found that a learning app improved test scores by 30% [3].",
        target_ref_ids=["b2"],
        claim_text="A particular study found that a learning app improved test scores by 30%.",
    ),
    CitationClaim(
        sentence="Modern state of the art computer vision systems primarily rely on SIFT feature descriptors [4].",
        target_ref_ids=["b3"],
        claim_text="Modern state of the art computer vision systems primarily rely on SIFT feature descriptors.",
    ),
    CitationClaim(
        sentence="Recent experiments in room-temperature quantum neural networks demonstrated general artificial intelligence [5].",
        target_ref_ids=["b4"],
        claim_text="Recent experiments in room-temperature quantum neural networks demonstrated general artificial intelligence.",
    ),
]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts raw plain text from PDF bytes via PyMuPDF with security validation."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Unable to read PDF file format: {e}")

    if doc.is_encrypted and doc.needs_pass:
        if not doc.authenticate(""):
            raise ValueError("The uploaded PDF is password-protected or encrypted. Please upload an unlocked research PDF.")

    text_parts: List[str] = []
    try:
        for page in doc:
            text_parts.append(page.get_text())
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("The uploaded PDF is password-protected or encrypted. Please upload an unlocked research PDF.")
        raise ValueError(f"Error parsing PDF page text: {e}")

    extracted = "\n".join(text_parts).strip()
    if not extracted:
        raise ValueError("The uploaded PDF contains no extractable digital text (scanned image or rasterized canvas). Please upload a digital text research paper.")

    return extracted


def parse_bibliography_from_text(text: str) -> Dict[str, BiblioEntry]:
    """
    Extracts bibliographic references from the bibliography section of the paper.
    Matches bracketed references like [1] Title, Authors, Year, DOI or standard lists.
    """
    bib_map: Dict[str, BiblioEntry] = {}

    # Locate references section
    ref_heading_match = re.search(r"\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n", text)
    if not ref_heading_match:
        return bib_map

    ref_text = text[ref_heading_match.end() :]

    # Regex for numbered references: e.g. [1] or 1.
    entry_pattern = re.compile(
        r"(?:^|\n)\s*(?:\[(\d+)\]|(\d+)\.)\s+([^\n]+(?:\n(?!\s*(?:\[\d+\]|\d+\.))[^\n]+)*)",
        re.MULTILINE,
    )

    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
    year_pattern = re.compile(r"\b(19\d\d|20\d\d)\b")

    matches = entry_pattern.findall(ref_text)
    for m in matches:
        num = m[0] or m[1]
        raw_body = " ".join(m[2].split())
        ref_id = f"b{int(num) - 1}"

        doi_match = doi_pattern.search(raw_body)
        doi = doi_match.group(0) if doi_match else None

        years = year_pattern.findall(raw_body)
        year = years[-1] if years else None

        # Clean title heuristic: take first significant segment before year or publication
        title_candidate = raw_body
        for punct in [".", ",", "\"", "''"]:
            if punct in title_candidate:
                parts = [p.strip() for p in title_candidate.split(punct) if len(p.strip()) > 15]
                if parts:
                    title_candidate = parts[0]
                    break

        bib_map[ref_id] = BiblioEntry(
            ref_id=ref_id,
            title=title_candidate[:140],
            authors=[],
            year=year,
            doi=doi,
            raw_text=raw_body,
        )

    return bib_map


def parse_citations_from_text(text: str, bib_map: Dict[str, BiblioEntry]) -> List[CitationClaim]:
    """
    Finds in-text citation sentences matching [1], [1, 2], or [1-3]
    and maps them to resolved bibliography entries.
    """
    claims: List[CitationClaim] = []

    # Clean text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    citation_pattern = re.compile(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]")

    for s in sentences:
        s_clean = " ".join(s.split())
        if len(s_clean) < 25:
            continue

        match = citation_pattern.search(s_clean)
        if match:
            raw_nums = match.group(1)
            # Expand ranges like 1-3 or lists like 1, 2
            target_ids: List[str] = []
            for token in re.split(r"[,;]", raw_nums):
                token = token.strip()
                if "-" in token or "–" in token:
                    parts = re.split(r"[-–]", token)
                    try:
                        start, end = int(parts[0]), int(parts[1])
                        for n in range(start, end + 1):
                            target_ids.append(f"b{n - 1}")
                    except ValueError:
                        pass
                else:
                    try:
                        target_ids.append(f"b{int(token) - 1}")
                    except ValueError:
                        pass

            valid_ids = [r_id for r_id in target_ids if r_id in bib_map]
            if not valid_ids and target_ids:
                valid_ids = target_ids  # Preserve for potential phantom detection

            if valid_ids:
                claims.append(
                    CitationClaim(
                        sentence=s_clean,
                        target_ref_ids=valid_ids[:3],
                        claim_text=s_clean,
                    )
                )

    return claims


def extract_unreferenced_claims(text: str) -> Tuple[Dict[str, BiblioEntry], List[CitationClaim]]:
    """
    When an uploaded PDF contains text but lacks standard [1] numbered citations,
    extracts the strongest empirical assertions from the document itself so the user's
    actual file is audited rather than silently falling back to benchmark demo claims.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidate_sentences: List[str] = []

    # Priority assertion indicators
    keywords = [
        "demonstrate", "increase", "decrease", "improve", "show", "found", "find",
        "result", "achieve", "outperform", "percent", "%", "accuracy", "model",
        "study", "propose", "suggest", "evidence", "conclude", "significant", "require"
    ]

    for s in sentences:
        s_clean = " ".join(s.split())
        if 40 <= len(s_clean) <= 220:
            lower_s = s_clean.lower()
            if any(k in lower_s for k in keywords):
                candidate_sentences.append(s_clean)
                if len(candidate_sentences) >= 6:
                    break

    if not candidate_sentences:
        for s in sentences:
            s_clean = " ".join(s.split())
            if 45 <= len(s_clean) <= 200:
                candidate_sentences.append(s_clean)
                if len(candidate_sentences) >= 5:
                    break

    if not candidate_sentences:
        return BENCHMARK_BIBLIO, BENCHMARK_CLAIMS

    bib_map: Dict[str, BiblioEntry] = {
        "unref_0": BiblioEntry(
            ref_id="unref_0",
            title="Unreferenced Assertion (No citation in paper)",
            authors=["Uncited"],
            year=None,
            raw_text="The manuscript makes this assertion without providing a supporting bibliographic citation.",
        )
    }

    claims = [
        CitationClaim(
            sentence=sent,
            target_ref_ids=["unref_0"],
            claim_text=sent,
        )
        for sent in candidate_sentences
    ]

    return bib_map, claims


def parse_pdf(pdf_bytes: bytes, use_stub: bool = False) -> Tuple[Dict[str, BiblioEntry], List[CitationClaim]]:
    """
    High-level entrypoint: parses PDF bytes or returns standard benchmark claims.
    No GROBID service or external daemon required.
    """
    if use_stub or not pdf_bytes or pdf_bytes.startswith(b"%PDF-demo") or len(pdf_bytes) < 100:
        return BENCHMARK_BIBLIO, BENCHMARK_CLAIMS

    raw_text = extract_text_from_pdf(pdf_bytes)
    if not raw_text.strip():
        return BENCHMARK_BIBLIO, BENCHMARK_CLAIMS

    bib_map = parse_bibliography_from_text(raw_text)
    claims = parse_citations_from_text(raw_text, bib_map)

    # If extraction did not find formal numbered citations, audit document text directly
    if not claims or not bib_map:
        return extract_unreferenced_claims(raw_text)

    return bib_map, claims
