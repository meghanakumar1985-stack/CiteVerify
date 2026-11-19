# 🔬 CiteVerify

> **CiteVerify** is a lightweight scientific claim and citation verification tool. It takes an uploaded research paper PDF, parses its citations and extracted claims, retrieves the cited evidence from external scholarly databases, and verifies whether the cited paper actually supports the claims made about it.

---

## 🚀 How It Works (The 8-Step Pipeline)

1. **PDF Upload & GROBID:** Upload a PDF. GROBID parses it into structured TEI-XML (`/api/processFulltextDocument`).
2. **TEI XML Parsing:** Identifies in-text citations (`<ref type="bibr" target="#bN">`) and maps them to bibliography entries (`<biblStruct xml:id="bN">`: title, authors, year, DOI).
3. **Claim Extraction:** Extracts the core scientific assertion from citation sentences using cue-word filters (adapted from `claimscope`).
4. **Evidence Retrieval:** Fetches the cited paper's abstract and metadata using Semantic Scholar or CrossRef.
5. **Passage Ranking:** Chunks the evidence text into passages and ranks them with **TF-IDF cosine similarity** (`scikit-learn`).
6. **Stance Classification:** Categorizes the claim as **Supported**, **Partially supported**, **Contradicted**, or **Not found** using lexical overlap, negation matching, and contrast cues.
7. **Reference Checking:** Identifies phantom citations (non-existent papers), author/year metadata mismatches, and misleading references (adapted from `refchecker`).
8. **Streamlit Report & CSV:** Presents an interactive dashboard with summary metrics, color-coded verdicts, deep-dive passage inspections, and a CSV download.

---

## 🛠️ Project Structure

```text
citeverify/
├── app.py                     # Streamlit web application interface
├── pipeline.py                # End-to-end pipeline runner
├── core/
│   ├── models.py              # Data structures (BiblioEntry, CitationClaim, etc.)
│   ├── grobid_client.py       # Step 1: GROBID client and health check
│   ├── tei_parser.py          # Step 2: TEI-XML parser for citations & bibliography
│   ├── claim_extractor.py     # Step 3: Scientific claim extraction
│   ├── evidence_retriever.py  # Step 4: Semantic Scholar / CrossRef API retriever
│   ├── passage_ranker.py      # Step 5: Evidence passage chunking & TF-IDF ranking
│   ├── claim_classifier.py    # Step 6: Stance classification (Support/Contradict)
│   └── reference_checker.py   # Step 7: Reference validity & metadata consistency check
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration (GROBID URL, API keys)
├── .env.example               # Example configuration template
└── README.md
```

---

## ⚡ Quick Start

### 1. Activate the Virtual Environment
```bash
cd /Users/meghanakumar/Desktop/citeverify
source .venv/bin/activate
```

### 2. Configure Environment (Optional)
CiteVerify runs out of the box with built-in fallback demo data.
If you have a local GROBID instance running (e.g. via Docker):
```bash
# Optional: Start GROBID in Docker on port 8070
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.1
```
You can configure `GROBID_URL` and `SEMANTIC_SCHOLAR_API_KEY` in `.env`:
```text
GROBID_URL=http://localhost:8070
SEMANTIC_SCHOLAR_API_KEY=
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Pipeline via Command Line

You can also run a quick standalone test directly in Python:
```bash
python3 -c "
from pipeline import run_pipeline
df, diag = run_pipeline(b'%PDF-demo', use_stub=True)
print(df[['Claim', 'Citation', 'Verdict', 'Flag', 'Similarity']])
"
```
