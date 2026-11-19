"""Step 2: Parse TEI XML for in-text citations and bibliography entries."""
from __future__ import annotations
import re
from typing import Dict, List, Tuple

try:
    from lxml import etree
    USE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    USE_LXML = False

from core.models import BiblioEntry, CitationClaim

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _get_text(element) -> str:
    """Helper to extract and normalize text from an XML element."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_bibliography(root) -> Dict[str, BiblioEntry]:
    """Extracts all bibliography entries from TEI XML."""
    bib_map: Dict[str, BiblioEntry] = {}
    if USE_LXML:
        bibl_structs = root.xpath(".//tei:listBibl/tei:biblStruct | .//tei:listBibl/tei:bibl", namespaces=TEI_NS)
    else:
        bibl_structs = root.findall(".//{http://www.tei-c.org/ns/1.0}listBibl/{http://www.tei-c.org/ns/1.0}biblStruct")
        bibl_structs.extend(root.findall(".//{http://www.tei-c.org/ns/1.0}listBibl/{http://www.tei-c.org/ns/1.0}bibl"))

    for idx, bibl in enumerate(bibl_structs):
        # Get xml:id (e.g., 'b0')
        ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id") or bibl.get("id") or f"b{idx}"
        ref_id = ref_id.lstrip("#")

        # Extract title (article title level="a", or monograph title level="m")
        title = ""
        if USE_LXML:
            title_el = (
                bibl.find(".//tei:analytic/tei:title", namespaces=TEI_NS)
                or bibl.find(".//tei:monogr/tei:title", namespaces=TEI_NS)
                or bibl.find(".//tei:title", namespaces=TEI_NS)
            )
        else:
            title_el = (
                bibl.find(".//{http://www.tei-c.org/ns/1.0}analytic/{http://www.tei-c.org/ns/1.0}title")
                or bibl.find(".//{http://www.tei-c.org/ns/1.0}monogr/{http://www.tei-c.org/ns/1.0}title")
                or bibl.find(".//{http://www.tei-c.org/ns/1.0}title")
            )
        if title_el is not None:
            title = _get_text(title_el)

        # Extract authors
        authors = []
        if USE_LXML:
            author_els = bibl.findall(".//tei:analytic/tei:author", namespaces=TEI_NS)
            if not author_els:
                author_els = bibl.findall(".//tei:monogr/tei:author", namespaces=TEI_NS)
            if not author_els:
                author_els = bibl.findall(".//tei:author", namespaces=TEI_NS)
        else:
            author_els = bibl.findall(".//{http://www.tei-c.org/ns/1.0}analytic/{http://www.tei-c.org/ns/1.0}author")
            if not author_els:
                author_els = bibl.findall(".//{http://www.tei-c.org/ns/1.0}monogr/{http://www.tei-c.org/ns/1.0}author")
            if not author_els:
                author_els = bibl.findall(".//{http://www.tei-c.org/ns/1.0}author")

        for auth in author_els:
            if USE_LXML:
                forenames = [
                    _get_text(fn) for fn in auth.findall(".//tei:persName/tei:forename", namespaces=TEI_NS)
                ]
                surname = _get_text(auth.find(".//tei:persName/tei:surname", namespaces=TEI_NS))
            else:
                forenames = [
                    _get_text(fn) for fn in auth.findall(".//{http://www.tei-c.org/ns/1.0}persName/{http://www.tei-c.org/ns/1.0}forename")
                ]
                surname = _get_text(auth.find(".//{http://www.tei-c.org/ns/1.0}persName/{http://www.tei-c.org/ns/1.0}surname"))
            name = " ".join([*forenames, surname]).strip()
            if not name:
                name = _get_text(auth).strip()
            if name:
                authors.append(name)

        # Extract published year
        year = None
        if USE_LXML:
            date_el = bibl.find(".//tei:date[@when]", namespaces=TEI_NS)
        else:
            date_el = bibl.find(".//{http://www.tei-c.org/ns/1.0}date[@when]")
        if date_el is not None:
            when = date_el.get("when", "")
            match = re.search(r"\b(19\d\d|20\d\d)\b", when)
            if match:
                year = match.group(1)
        if not year:
            date_any = bibl.find(".//tei:date", namespaces=TEI_NS) if USE_LXML else bibl.find(".//{http://www.tei-c.org/ns/1.0}date")
            if date_any is not None:
                match = re.search(r"\b(19\d\d|20\d\d)\b", _get_text(date_any))
                if match:
                    year = match.group(1)

        # Extract DOI
        doi = None
        if USE_LXML:
            doi_el = (
                bibl.find(".//tei:idno[@type='DOI']", namespaces=TEI_NS)
                or bibl.find(".//tei:idno[@type='doi']", namespaces=TEI_NS)
            )
        else:
            doi_el = (
                bibl.find(".//{http://www.tei-c.org/ns/1.0}idno[@type='DOI']")
                or bibl.find(".//{http://www.tei-c.org/ns/1.0}idno[@type='doi']")
            )
        if doi_el is not None:
            doi = _get_text(doi_el)
        elif "doi.org/" in _get_text(bibl):
            match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", _get_text(bibl))
            if match:
                doi = match.group(0)

        raw_text = _get_text(bibl)

        bib_map[ref_id] = BiblioEntry(
            ref_id=ref_id,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            raw_text=raw_text,
        )

    return bib_map


def parse_in_text_citations(root) -> List[CitationClaim]:
    """
    Finds paragraphs, splits into sentences, and identifies sentences containing
    citation references (<ref type='bibr' target='#bN'>).
    """
    citation_claims: List[CitationClaim] = []
    if USE_LXML:
        paragraphs = root.xpath(".//tei:body//tei:p", namespaces=TEI_NS)
    else:
        paragraphs = root.findall(".//{http://www.tei-c.org/ns/1.0}body//{http://www.tei-c.org/ns/1.0}p")

    for p in paragraphs:
        if USE_LXML:
            refs = p.xpath(".//tei:ref[@type='bibr']", namespaces=TEI_NS)
        else:
            refs = [
                el for el in p.findall(".//{http://www.tei-c.org/ns/1.0}ref")
                if el.get("type") == "bibr"
            ]
        if not refs:
            continue

        p_text = _get_text(p)
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(p_text) if s.strip()]

        for ref in refs:
            raw_target = ref.get("target", "")
            # Handle multiple targets e.g. target="#b0 #b1"
            targets = [t.lstrip("#") for t in raw_target.split() if t.lstrip("#")]
            ref_text = _get_text(ref)
            if not targets:
                continue

            # Find matching sentence
            matched_sentence = ""
            for s in sentences:
                if ref_text and ref_text in s:
                    matched_sentence = s
                    break

            if not matched_sentence and sentences:
                matched_sentence = sentences[0]

            if matched_sentence:
                citation_claims.append(
                    CitationClaim(
                        sentence=matched_sentence,
                        target_ref_ids=targets,
                        claim_text=matched_sentence,
                    )
                )

    return citation_claims


def parse_tei(tei_xml: str) -> Tuple[Dict[str, BiblioEntry], List[CitationClaim]]:
    """Main parsing function that returns (biblio_map, citation_claims)."""
    if USE_LXML:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(tei_xml.encode("utf-8"), parser=parser)
    else:
        root = etree.fromstring(tei_xml)
    bib_map = parse_bibliography(root)
    claims = parse_in_text_citations(root)
    return bib_map, claims
