"""CiteVerify Streamlit Web Application."""
from __future__ import annotations
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.grobid_client import check_grobid_health
from pipeline import run_pipeline

load_dotenv()

st.set_page_config(
    page_title="CiteVerify - Scientific Citation & Claim Verifier",
    page_icon="🔬",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stBadge {
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🔬 CiteVerify</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated verification of scientific claims, citation grounding, and reference validity in research papers.</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
env_grobid_url = os.getenv("GROBID_URL", "http://localhost:8070")
grobid_url_input = st.sidebar.text_input("GROBID Server URL", value=env_grobid_url)

# Health Check for GROBID
grobid_online = check_grobid_health(grobid_url_input)
if grobid_online:
    st.sidebar.success(f"🟢 Connected to GROBID at `{grobid_url_input}`")
elif check_grobid_health("https://grobidOrg-grobid-crf.hf.space"):
    st.sidebar.info(
        f"🌐 Local GROBID (`{grobid_url_input}`) is offline.\n\n"
        "Automatic fallback to online GROBID server (`https://grobidOrg-grobid-crf.hf.space`) is active."
    )
else:
    st.sidebar.warning(
        f"⚠️ GROBID is offline at `{grobid_url_input}`.\n\n"
        "CiteVerify will fall back to sample demo data."
    )

force_demo = st.sidebar.checkbox(
    "Use Demo Data (offline stub mode)",
    value=False,
    help="Enable to test the entire verification pipeline immediately with sample claims.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ How CiteVerify Works")
st.sidebar.markdown(
    """
1. **GROBID**: Parses PDF into TEI-XML structure.
2. **Citation Linker**: Maps in-text citations (`#b0`) to bibliography entries.
3. **Claim Extractor**: Extracts assertion clauses using cue words.
4. **Evidence Retrieval**: Fetches cited paper abstracts (Semantic Scholar / CrossRef).
5. **Passage Ranking**: TF-IDF cosine similarity over evidence chunks.
6. **Stance Classifier**: Categorizes support vs contradiction & negation mismatch.
7. **RefChecker**: Detects phantom citations & metadata inconsistencies.
"""
)

# Main Upload Area
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload a scientific paper (PDF)",
        type=["pdf"],
        help="Upload an academic research paper PDF to verify its citations and claims.",
    )
with col2:
    st.write("")
    st.write("")
    demo_button = st.button("🚀 Run Demo on Sample Paper", use_container_width=True)

if uploaded_file or demo_button:
    pdf_bytes = uploaded_file.getvalue() if uploaded_file else b"%PDF-1.4 demo stub"
    is_stub = force_demo or (demo_button and not uploaded_file)

    with st.spinner("Running CiteVerify pipeline..."):
        df_results, diagnostics = run_pipeline(
            pdf_bytes=pdf_bytes,
            grobid_url=grobid_url_input,
            use_stub=is_stub,
        )

    if diagnostics.get("warnings"):
        for w in diagnostics["warnings"]:
            st.info(f"ℹ️ {w}")

    if df_results.empty:
        st.warning("No in-text citations or claims could be identified in the document.")
    else:
        st.success(f"✅ Analysis complete! Verified {len(df_results)} citation claims.")

        # Metric Summary Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        total_claims = len(df_results)
        supported_cnt = len(df_results[df_results["Verdict"].str.contains("Supported", case=False)])
        contradicted_cnt = len(df_results[df_results["Verdict"].str.contains("Contradicted", case=False)])
        flagged_cnt = len(df_results[df_results["Flag"] != "Verified"])

        mcol1.metric("Claims Analyzed", total_claims)
        mcol2.metric("Supported / Partial", supported_cnt)
        mcol3.metric("Contradicted", contradicted_cnt)
        mcol4.metric("Flagged Issues", flagged_cnt)

        st.markdown("### 📋 Verification Report")

        # Interactive Filtering Controls
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            verdict_filter = st.selectbox(
                "Filter by Verdict",
                options=["All Verdicts", "Supported", "Partially supported", "Contradicted", "Not found"],
            )
        with fcol2:
            flag_filter = st.selectbox(
                "Filter by Reference Flag",
                options=["All Citations", "Verified Only", "Flagged Issues Only"],
            )

        filtered_df = df_results.copy()
        if verdict_filter != "All Verdicts":
            filtered_df = filtered_df[filtered_df["Verdict"] == verdict_filter]

        if flag_filter == "Verified Only":
            filtered_df = filtered_df[filtered_df["Flag"] == "Verified"]
        elif flag_filter == "Flagged Issues Only":
            filtered_df = filtered_df[filtered_df["Flag"] != "Verified"]

        # Color-coded display table
        def highlight_verdict(val):
            if val == "Supported":
                return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
            elif val == "Partially supported":
                return "background-color: #dbeafe; color: #1e40af; font-weight: bold;"
            elif val == "Contradicted":
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            else:
                return "background-color: #f3f4f6; color: #374151;"

        def highlight_flag(val):
            if val == "Verified":
                return "background-color: #d1fae5; color: #065f46;"
            elif "Misleading" in str(val) or "Contradiction" in str(val):
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            elif "Non-Existent" in str(val) or "Mismatch" in str(val):
                return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
            return "background-color: #f3f4f6; color: #374151;"

        styler = filtered_df.style
        style_mapper = getattr(styler, "map", getattr(styler, "applymap", None))
        if style_mapper is not None:
            styled_df = style_mapper(highlight_verdict, subset=["Verdict"])
            styled_df = style_mapper(highlight_flag, subset=["Flag"])
        else:
            styled_df = filtered_df

        st.dataframe(styled_df, use_container_width=True, height=350)

        # CSV Download Button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Verification Report (CSV)",
            data=csv_data,
            file_name="citeverify_report.csv",
            mime="text/csv",
        )

        # Detailed Inspection Accordion
        with st.expander("🔍 Deep-Dive: Evidence Passages & Citation Match Details", expanded=False):
            for idx, row in filtered_df.iterrows():
                st.markdown(f"#### #{idx+1}: {row['Citation']}")
                st.markdown(f"**Extracted Claim:** *\"{row['Claim']}\"*")
                st.markdown(
                    f"**Verdict:** `{row['Verdict']}` &nbsp;|&nbsp; "
                    f"**Flag:** `{row['Flag']}` &nbsp;|&nbsp; "
                    f"**TF-IDF Similarity:** `{row['Similarity']:.3f}`"
                )
                st.markdown(f"**Retrieved Evidence Passage:**")
                st.info(f"\"{row['Top Evidence Passage']}\"")
                st.caption(f"Reasoning: {row['Details']}")
                st.markdown("---")
else:
    st.info("👆 Upload a research paper PDF or click **'Run Demo on Sample Paper'** to see CiteVerify in action.")
