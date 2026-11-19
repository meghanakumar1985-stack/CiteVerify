"""Step 1: GROBID PDF processing client."""
from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
PUBLIC_GROBID_URL = "https://grobidOrg-grobid-crf.hf.space"

# Minimal sample TEI XML for stub testing
SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div xmlns="http://www.tei-c.org/ns/1.0">
        <head>1. Introduction</head>
        <p>Recent experiments show that transformer models significantly improve scientific fact verification accuracy <ref type="bibr" target="#b0">(Vaswani et al., 2017)</ref>. Furthermore, prior studies suggest that graph neural networks fail on low-resource citation graph tasks <ref type="bibr" target="#b1">(Kipf and Welling, 2016)</ref>.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <title level="a" type="main">Attention Is All You Need</title>
              <author><persName><forename type="first">Ashish</forename><surname>Vaswani</surname></persName></author>
              <author><persName><forename type="first">Noam</forename><surname>Shazeer</surname></persName></author>
            </analytic>
            <monogr>
              <imprint>
                <date type="published" when="2017">2017</date>
              </imprint>
            </monogr>
            <idno type="DOI">10.48550/arXiv.1706.03762</idno>
          </biblStruct>
          <biblStruct xml:id="b1">
            <analytic>
              <title level="a" type="main">Semi-Supervised Classification with Graph Convolutional Networks</title>
              <author><persName><forename type="first">Thomas</forename><forename type="middle">N</forename><surname>Kipf</surname></persName></author>
              <author><persName><forename type="first">Max</forename><surname>Welling</surname></persName></author>
            </analytic>
            <monogr>
              <imprint>
                <date type="published" when="2016">2016</date>
              </imprint>
            </monogr>
            <idno type="DOI">10.48550/arXiv.1609.02907</idno>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def check_grobid_health(grobid_url: str | None = None) -> bool:
    """Checks if the GROBID service is running and responsive."""
    url = grobid_url or os.getenv("GROBID_URL", DEFAULT_GROBID_URL)
    health_endpoint = f"{url.rstrip('/')}/api/isalive"
    try:
        resp = requests.get(health_endpoint, timeout=(1.0, 2.0))
        return resp.status_code == 200
    except Exception:
        return False


def process_pdf(pdf_bytes: bytes, grobid_url: str | None = None, use_stub: bool = False) -> str:
    """
    Sends PDF bytes to GROBID processFulltextDocument endpoint and returns TEI XML.
    Falls back to sample TEI if use_stub=True.
    Tries primary URL first, then public demo server, then stub if configured.
    """
    if use_stub:
        return SAMPLE_TEI_XML

    primary_url = grobid_url or os.getenv("GROBID_URL", DEFAULT_GROBID_URL)
    urls_to_try = [primary_url]

    # If primary is localhost and not reachable, try public demo as fallback
    if "localhost" in primary_url or "127.0.0.1" in primary_url:
        if not check_grobid_health(primary_url):
            urls_to_try.append(PUBLIC_GROBID_URL)

    last_error = None
    for url in urls_to_try:
        endpoint = f"{url.rstrip('/')}/api/processFulltextDocument"
        try:
            files = {"input": ("document.pdf", pdf_bytes, "application/pdf")}
            data = {
                "generateIDs": "1",
                "consolidateHeader": "1",
                "consolidateCitations": "1",
            }
            resp = requests.post(endpoint, files=files, data=data, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except Exception as exc:
            last_error = str(exc)

    raise ConnectionError(
        f"Could not process PDF via GROBID ({last_error}). "
        f"Checked: {', '.join(urls_to_try)}."
    )
