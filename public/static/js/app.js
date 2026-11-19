/**
 * CiteVerify — Production Web Client
 * Clean, safe, modular JavaScript for real-time claim & citation verification.
 * Zero client-side API keys; communicates strictly with the backend API.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  let selectedFile = null;
  let verificationData = [];
  let currentFilter = "all";
  let activeDiagnostics = null;

  // Toast Notification
  const toast = (message) => {
    const t = $("toast");
    if (!t) return;
    t.textContent = message;
    t.classList.add("show");
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove("show"), 2800);
  };

  // File size formatting
  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return "0 KB";
    if (bytes > 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    return Math.ceil(bytes / 1024) + " KB";
  };

  // Escape HTML helper for security
  const esc = (str) => {
    if (str === null || str === undefined) return "";
    return String(str).replace(/[&<>"']/g, (char) => {
      switch (char) {
        case "&": return "&amp;";
        case "<": return "&lt;";
        case ">": return "&gt;";
        case '"': return "&quot;";
        case "'": return "&#39;";
        default: return char;
      }
    });
  };

  // File selection
  function handleFileSelect(file) {
    if (!file) return;
    const errEl = $("err");
    errEl.textContent = "";

    const isPdf = /\.pdf$/i.test(file.name) || file.type === "application/pdf";
    if (!isPdf) {
      errEl.textContent = "Please select a valid PDF research paper.";
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      errEl.textContent = "PDF file size must be less than 25 MB.";
      return;
    }

    selectedFile = file;
    $("fName").textContent = file.name;
    $("fSize").textContent = formatBytes(file.size);
    $("fileRow").classList.add("show");
    $("runBtn").disabled = false;
    $("runBtn").textContent = "Run Review (" + (file.name.length > 18 ? file.name.substring(0, 16) + "..." : file.name) + ")";
    $("status").textContent = "Ready to review: " + file.name;
    toast("Paper loaded: " + file.name);
  }

  // Clear selected file
  function clearFile() {
    selectedFile = null;
    const fileInput = $("file");
    if (fileInput) fileInput.value = "";
    $("fileRow").classList.remove("show");
    $("runBtn").disabled = true;
    $("runBtn").textContent = "Run Review";
    $("status").textContent = "No document selected";
  }

  // Bind drag-and-drop & file input
  const fileInput = $("file");
  const dropZone = $("drop");

  if (fileInput) {
    fileInput.onchange = (e) => handleFileSelect(e.target.files[0]);
  }

  if ($("removeBtn")) {
    $("removeBtn").onclick = clearFile;
  }

  if (dropZone) {
    dropZone.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput?.click();
      }
    };

    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("drag");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer?.files?.length) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
  }

  // Call API for real verification
  async function executeVerification(fileOrDemo) {
    const isDemo = fileOrDemo === "demo" || !fileOrDemo;
    const runBtn = $("runBtn");
    const demoBtn = $("demoBtn");
    const progress = $("progress");
    const progBar = $("progBar");
    const status = $("status");
    const errEl = $("err");

    const loadingBox = $("loadingBox");
    const loadingMsg = $("loadingMsg");

    if (runBtn) runBtn.disabled = true;
    if (demoBtn) demoBtn.disabled = true;
    errEl.textContent = "";

    progress.classList.add("show");
    loadingBox?.classList.add("show");
    progBar.style.width = "15%";
    const initialMsg = isDemo
      ? "Loading benchmark demonstration paper..."
      : "Extracting citations and document text in-memory...";
    status.textContent = initialMsg;
    if (loadingMsg) loadingMsg.textContent = initialMsg;

    // Progress animation steps
    let progressVal = 15;
    const progressTimer = setInterval(() => {
      if (progressVal < 45) {
        progressVal += 10;
        progBar.style.width = progressVal + "%";
        const m = "Querying scholarly databases (Semantic Scholar, CrossRef)...";
        status.textContent = m;
        if (loadingMsg) loadingMsg.textContent = m;
      } else if (progressVal < 85) {
        progressVal += 8;
        progBar.style.width = progressVal + "%";
        const m = "Analyzing entailment against retrieved sources...";
        status.textContent = m;
        if (loadingMsg) loadingMsg.textContent = m;
      }
    }, 400);

    try {
      let response;
      if (isDemo) {
        response = await fetch("/api/demo");
      } else {
        const formData = new FormData();
        formData.append("file", fileOrDemo);
        response = await fetch("/api/verify", {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(errPayload.error || `Server returned status ${response.status}`);
      }

      const payload = await response.json();
      clearInterval(progressTimer);
      progBar.style.width = "100%";
      status.textContent = "Verification synthesis complete";

      // Parse confidence to numeric integer percentage
      function parseConfidence(rawConf, sim) {
        if (typeof rawConf === "number" && !isNaN(rawConf)) {
          return rawConf <= 1.0 ? Math.round(rawConf * 100) : Math.round(rawConf);
        }
        const s = String(rawConf || "").trim().toLowerCase();
        if (s === "high") return 92;
        if (s === "medium") return 74;
        if (s === "low") return 48;
        const num = parseFloat(s);
        if (!isNaN(num)) {
          return num <= 1.0 ? Math.round(num * 100) : Math.round(num);
        }
        const numSim = parseFloat(sim);
        if (!isNaN(numSim) && numSim > 0) {
          return Math.min(95, Math.max(45, Math.round(numSim * 100)));
        }
        return 75;
      }

      // Normalize records
      const rawRecords = payload.results || [];
      verificationData = rawRecords.map((r, idx) => {
        const confVal = parseConfidence(r.Confidence || r.confidence, r.Similarity || r.similarity);
        return {
          id: idx + 1,
          claim: r.Claim || r.claim || "Unspecified scientific claim",
          source: r.Citation || r["Cited Source"] || r.source || "Document Assertion",
          verdict: r.Verdict || r.verdict || "Review Required",
          confidence: confVal,
          confidenceLabel: (typeof r.Confidence === "string" && ["high", "medium", "low"].includes(r.Confidence.toLowerCase())) ? r.Confidence : "",
          reviewStatus: r["Review Status"] || r.reviewStatus || "Standard",
          reasoning: r.Details || r["NLI Reasoning"] || r.reasoning || "Evaluation based on retrieved reference abstract.",
          evidence: r["Top Evidence Passage"] || r["Source Passage"] || r.evidence || "No external evidence passage found.",
          doi: r["Source Link"] || r.DOI || r.doi || "",
        };
      });

      activeDiagnostics = payload.diagnostics || {};

      updateSummary(payload);
      renderTable();

      if ($("exportBtn")) $("exportBtn").disabled = false;
      toast(`Verification complete: ${verificationData.length} claims audited.`);

      setTimeout(() => {
        $("results")?.scrollIntoView({ behavior: "smooth" });
      }, 350);

    } catch (error) {
      clearInterval(progressTimer);
      console.error("CiteVerify execution error:", error);
      errEl.textContent = "Verification encountered an issue: " + error.message;
      status.textContent = "Verification interrupted";
      toast("Error running verification.");
    } finally {
      if (runBtn) {
        if (selectedFile) {
          runBtn.disabled = false;
          runBtn.textContent = "Re-run Review (" + (selectedFile.name.length > 18 ? selectedFile.name.substring(0, 16) + "..." : selectedFile.name) + ")";
        } else {
          runBtn.disabled = true;
          runBtn.textContent = "Run Review";
        }
      }
      if (demoBtn) {
        demoBtn.disabled = false;
        demoBtn.textContent = "Run Benchmark Demo";
      }
      if (isDemo) {
        status.textContent = "Benchmark test suite audit complete";
      } else if (selectedFile) {
        status.textContent = "Audit complete for " + selectedFile.name;
      }
      loadingBox?.classList.remove("show");
      setTimeout(() => {
        progress.classList.remove("show");
        progBar.style.width = "0%";
      }, 700);
    }
  }

  // Run Button triggers
  if ($("runBtn")) {
    $("runBtn").onclick = () => {
      if (selectedFile) {
        executeVerification(selectedFile);
      }
    };
  }

  if ($("demoBtn")) {
    $("demoBtn").onclick = () => {
      executeVerification("demo");
    };
  }

  // Any button with .demo-trigger runs the demo immediately
  document.querySelectorAll(".demo-trigger").forEach((el) => {
    el.onclick = (e) => {
      e.preventDefault();
      $("workspace")?.scrollIntoView({ behavior: "smooth" });
      executeVerification("demo");
    };
  });

  // Update Summary Metrics
  function updateSummary(payload) {
    const total = payload.total_claims || verificationData.length;
    const sup = payload.supported_count ?? verificationData.filter((r) => r.verdict.toLowerCase().includes("support") && !r.verdict.toLowerCase().includes("partial")).length;
    const con = payload.contradicted_count ?? verificationData.filter((r) => r.verdict.toLowerCase().includes("contradict")).length;
    const rev = (payload.needs_review_count || 0) + (payload.partial_count || 0) + (payload.outdated_count || 0);

    if ($("mTotal")) $("mTotal").textContent = total;
    if ($("mSup")) $("mSup").textContent = sup;
    if ($("mCon")) $("mCon").textContent = con;
    if ($("mRev")) $("mRev").textContent = rev;

    if (total > 0) {
      const bS = $("bS");
      const bC = $("bC");
      const bR = $("bR");
      if (bS) bS.style.width = (sup / total) * 100 + "%";
      if (bC) bC.style.width = (con / total) * 100 + "%";
      if (bR) bR.style.width = (rev / total) * 100 + "%";
    }
  }

  // Render Results Table
  function renderTable() {
    const rowsEl = $("rows");
    if (!rowsEl) return;

    if (!verificationData.length) {
      rowsEl.innerHTML = `<tr><td colspan="4" class="empty">Upload a paper or click "Run Benchmark Demo" to view results.</td></tr>`;
      return;
    }

    const query = ($("q")?.value || "").toLowerCase().trim();

    const filtered = verificationData.filter((item) => {
      const matchesFilter =
        currentFilter === "all" ||
        (currentFilter === "Supported" && item.verdict.toLowerCase().includes("support") && !item.verdict.toLowerCase().includes("partial")) ||
        (currentFilter === "Contradicted" && item.verdict.toLowerCase().includes("contradict")) ||
        (currentFilter === "Needs review" && (item.verdict.toLowerCase().includes("review") || item.verdict.toLowerCase().includes("partial") || item.verdict.toLowerCase().includes("outdated")));

      const matchesSearch =
        !query ||
        item.claim.toLowerCase().includes(query) ||
        item.source.toLowerCase().includes(query) ||
        item.verdict.toLowerCase().includes(query);

      return matchesFilter && matchesSearch;
    });

    if (!filtered.length) {
      rowsEl.innerHTML = `<tr><td colspan="4" class="empty">No claims match the filter criteria.</td></tr>`;
      return;
    }

    const getBadgeClass = (verdict) => {
      const v = verdict.toLowerCase();
      if (v.includes("contradict")) return "b-con";
      if (v.includes("partial")) return "b-par";
      if (v.includes("support")) return "b-sup";
      if (v.includes("outdated")) return "b-out";
      return "b-rev";
    };

    const getVerdictSymbol = (verdict) => {
      const v = verdict.toLowerCase();
      if (v.includes("contradict")) return "[✕]";
      if (v.includes("partial")) return "[~]";
      if (v.includes("support")) return "[✓]";
      if (v.includes("outdated")) return "[◷]";
      return "[!]";
    };

    rowsEl.innerHTML = filtered
      .map(
        (item) => `
        <tr class="clickable-row" data-id="${item.id}" title="Click to inspect evidence">
          <td data-l="Claim" class="claim">
            ${esc(item.claim)}
            <small>Click row to inspect evidence details</small>
          </td>
          <td data-l="Cited source" class="src">
            ${esc(item.source)}
            ${item.doi ? `<br><small style="font-family:var(--mono);color:var(--muted)">DOI: ${esc(item.doi)}</small>` : ""}
          </td>
          <td data-l="Verdict">
            <span class="badge ${getBadgeClass(item.verdict)}">
              ${getVerdictSymbol(item.verdict)} ${esc(item.verdict)}
            </span>
          </td>
          <td data-l="Confidence">
            <span class="conf">
              <i><b style="width:${item.confidence}%"></b></i>
              ${item.confidence}%
            </span>
          </td>
        </tr>
      `
      )
      .join("");

    // Bind row click for inspection modal
    rowsEl.querySelectorAll(".clickable-row").forEach((row) => {
      row.onclick = () => {
        const id = parseInt(row.getAttribute("data-id"), 10);
        const item = verificationData.find((x) => x.id === id);
        if (item) openModal(item);
      };
    });
  }

  // Inspection Modal
  function openModal(item) {
    const modal = $("detailModal");
    if (!modal) return;

    $("mClaimText").textContent = item.claim;
    $("mSourceText").textContent = item.source;
    $("mVerdictBadge").textContent = item.verdict;
    $("mVerdictBadge").className = `badge ${item.verdict.toLowerCase().includes("support") ? "b-sup" : item.verdict.toLowerCase().includes("contradict") ? "b-con" : "b-rev"}`;
    $("mConfidenceText").textContent = item.confidence + "% " + (item.confidenceLabel ? "(" + item.confidenceLabel + ")" : "Rubric Grounding");
    $("mEvidenceText").textContent = item.evidence || "Extracted from verified paper abstract.";
    $("mReasoningText").textContent = item.reasoning || "NLI semantic entailment verification.";

    modal.style.display = "grid";
  }

  function closeModal() {
    const modal = $("detailModal");
    if (modal) modal.style.display = "none";
  }

  if ($("modalClose")) $("modalClose").onclick = closeModal;
  if ($("detailModal")) {
    $("detailModal").onclick = (e) => {
      if (e.target === $("detailModal")) closeModal();
    };
  }

  // Segment Filter Tabs
  const segEl = $("seg");
  if (segEl) {
    segEl.onclick = (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      currentFilter = btn.getAttribute("data-f") || "all";
      segEl.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
      renderTable();
    };
  }

  // Live Search
  const searchInput = $("q");
  if (searchInput) {
    searchInput.oninput = renderTable;
  }

  // CSV Export
  const exportBtn = $("exportBtn");
  if (exportBtn) {
    exportBtn.onclick = () => {
      if (!verificationData.length) return;
      const headers = ["Claim", "Cited Source", "Verdict", "Confidence", "Reasoning", "Evidence"];
      const rows = verificationData.map((r) => [
        `"${String(r.claim).replace(/"/g, '""')}"`,
        `"${String(r.source).replace(/"/g, '""')}"`,
        `"${String(r.verdict).replace(/"/g, '""')}"`,
        r.confidence,
        `"${String(r.reasoning).replace(/"/g, '""')}"`,
        `"${String(r.evidence).replace(/"/g, '""')}"`,
      ]);

      const csvContent = [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `citeverify_audit_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      toast("Verification report exported (CSV)");
    };
  }

  // Mobile menu toggle
  const menuBtn = $("menuBtn");
  const mobileMenu = $("mobileMenu");
  if (menuBtn && mobileMenu) {
    menuBtn.onclick = () => {
      const open = mobileMenu.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    };

    mobileMenu.onclick = (e) => {
      if (e.target.tagName === "A") {
        mobileMenu.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
      }
    };
  }

  // Keyboard escape to close modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
})();
