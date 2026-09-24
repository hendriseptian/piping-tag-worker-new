import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";

const API_BASE = "https://piping-tag-worker-new.side-gs78.workers.dev";
const EXTRACT_URL = `${API_BASE}/api/extract`;

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const $ = (id) => document.getElementById(id);

const els = {
  pdfInput: $("pdfInput"),
  dropzone: $("dropzone"),
  dropTitle: $("dropTitle"),
  dropHint: $("dropHint"),
  fileRow: $("fileRow"),
  fileName: $("fileName"),
  fileMeta: $("fileMeta"),
  removeFileBtn: $("removeFileBtn"),
  scaleSelect: $("scaleSelect"),
  analyzeBtn: $("analyzeBtn"),
  progressCard: $("progressCard"),
  progressBar: $("progressBar"),
  progressPercent: $("progressPercent"),
  progressMessage: $("progressMessage"),
  progressTiles: $("progressTiles"),
  progressCandidates: $("progressCandidates"),
  progressFinal: $("progressFinal"),
  previewCard: $("previewCard"),
  pidNo: $("pidNo"),
  candidateCount: $("candidateCount"),
  tagCount: $("tagCount"),
  resultsBody: $("resultsBody"),
  emptyState: $("emptyState"),
  addRowBtn: $("addRowBtn"),
  exportBtn: $("exportBtn"),
  logCard: $("logCard"),
  logOutput: $("logOutput"),
  connectionStatus: $("connectionStatus"),
};

let selectedFile = null;
let currentRows = [];
let currentPid = "";
let currentMeta = {};

function setStatus(text, ok = true) {
  els.connectionStatus.innerHTML = `
    <span class="status-dot" style="${ok ? "" : "background:#ff6b6b;box-shadow:none"}"></span>
    <span>${escapeHtml(text)}</span>
  `;
}

function setProgress(percent, message) {
  const safe = Math.max(0, Math.min(100, Math.round(percent)));
  els.progressBar.style.width = `${safe}%`;
  els.progressPercent.textContent = `${safe}%`;
  els.progressMessage.textContent = message;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function log(message, data = null) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  els.logOutput.textContent += `${line}\n`;
  if (data !== null) {
    els.logOutput.textContent += `${JSON.stringify(data, null, 2)}\n`;
  }
  els.logCard.classList.remove("hidden");
}

function resetResults() {
  currentRows = [];
  currentPid = "";
  currentMeta = {};
  els.resultsBody.innerHTML = "";
  els.previewCard.classList.add("hidden");
  els.emptyState.classList.add("hidden");
}

function selectFile(file) {
  if (!file) return;

  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    alert("Please select a PDF file.");
    return;
  }

  selectedFile = file;
  resetResults();

  els.fileRow.classList.remove("hidden");
  els.fileName.textContent = file.name;
  els.fileMeta.textContent = `${formatBytes(file.size)} · PDF`;
  els.dropTitle.textContent = "PDF selected";
  els.dropHint.textContent = "Ready for high-resolution AI extraction.";
  els.analyzeBtn.disabled = false;
  setStatus("PDF ready");
}

function clearFile() {
  selectedFile = null;
  els.pdfInput.value = "";
  els.fileRow.classList.add("hidden");
  els.dropTitle.textContent = "Drop PDF here or click to browse";
  els.dropHint.textContent = "The first page will be processed.";
  els.analyzeBtn.disabled = true;
  resetResults();
  setStatus("Ready");
}

els.pdfInput.addEventListener("change", (event) => {
  selectFile(event.target.files?.[0]);
});

els.removeFileBtn.addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  selectFile(event.dataTransfer.files?.[0]);
});

async function loadFirstPage(file) {
  const buffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;

  if (pdf.numPages < 1) throw new Error("PDF has no pages.");

  const page = await pdf.getPage(1);
  return { pdf, page };
}

async function canvasToBase64(canvas, quality = 0.72) {
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", quality)
  );

  if (!blob) throw new Error("Unable to encode rendered image.");

  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }

  return {
    data: btoa(binary),
    mime_type: "image/jpeg",
    byteLength: blob.size,
  };
}

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

async function renderPage(page, scale) {
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(viewport.width, viewport.height);
  const context = canvas.getContext("2d", { alpha: false });

  await page.render({
    canvasContext: context,
    viewport,
    background: "white",
  }).promise;

  return { canvas, viewport };
}

function tileRects(width, height) {
  // 4 x 2 tiles with overlap. Overlap is intentional so tags near
  // tile boundaries are visible in a neighboring tile.
  const cols = 4;
  const rows = 2;
  const overlapX = 0.12;
  const overlapY = 0.18;

  const rects = [];

  for (let row = 0; row < rows; row++) {
    const y0 = row / rows;
    const y1 = (row + 1) / rows;

    for (let col = 0; col < cols; col++) {
      const x0 = col / cols;
      const x1 = (col + 1) / cols;

      const left = Math.max(0, x0 - overlapX / 2);
      const right = Math.min(1, x1 + overlapX / 2);
      const top = Math.max(0, y0 - overlapY / 2);
      const bottom = Math.min(1, y1 + overlapY / 2);

      rects.push({
        id: `tile-${row * cols + col + 1}`,
        left: Math.round(left * width),
        top: Math.round(top * height),
        right: Math.round(right * width),
        bottom: Math.round(bottom * height),
      });
    }
  }

  return rects;
}

async function makeTile(renderedCanvas, rect, index) {
  const width = rect.right - rect.left;
  const height = rect.bottom - rect.top;
  const canvas = createCanvas(width, height);
  const context = canvas.getContext("2d", { alpha: false });

  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);

  context.drawImage(
    renderedCanvas,
    rect.left,
    rect.top,
    width,
    height,
    0,
    0,
    width,
    height
  );

  // Keep payload safely below the Worker limits.
  let quality = 0.72;
  let result = await canvasToBase64(canvas, quality);

  while (result.data.length > 4_900_000 && quality > 0.45) {
    quality -= 0.07;
    result = await canvasToBase64(canvas, quality);
  }

  if (result.data.length > 5_000_000) {
    throw new Error(
      `Tile ${index + 1} is too large (${formatBytes(result.byteLength)}).`
    );
  }

  return {
    id: `tile-${index + 1}`,
    image: result.data,
    mime_type: result.mime_type,
  };
}

async function buildTiles(page, scale) {
  setProgress(12, "Rendering page at high resolution…");

  const rendered = await renderPage(page, scale);
  const { canvas } = rendered;

  log("Rendered page", {
    width: canvas.width,
    height: canvas.height,
    scale,
  });

  const rects = tileRects(canvas.width, canvas.height);

  setProgress(25, "Creating overlapping vision tiles…");

  const tiles = [];
  for (let i = 0; i < rects.length; i++) {
    setProgress(25 + (i / rects.length) * 18, `Encoding tile ${i + 1} / ${rects.length}…`);
    tiles.push(await makeTile(canvas, rects[i], i));
  }

  // Overview is the full page at a lower payload size, used only for P&ID number.
  const overviewCanvas = createCanvas(
    Math.min(canvas.width, 2600),
    Math.round(Math.min(canvas.width, 2600) * canvas.height / canvas.width)
  );
  const overviewCtx = overviewCanvas.getContext("2d", { alpha: false });
  overviewCtx.fillStyle = "#fff";
  overviewCtx.fillRect(0, 0, overviewCanvas.width, overviewCanvas.height);
  overviewCtx.drawImage(canvas, 0, 0, overviewCanvas.width, overviewCanvas.height);

  const overview = await canvasToBase64(overviewCanvas, 0.65);

  return {
    tiles,
    overview: {
      image: overview.data,
      mime_type: overview.mime_type,
    },
  };
}

async function analyze() {
  if (!selectedFile) return;

  els.analyzeBtn.disabled = true;
  els.progressCard.classList.remove("hidden");
  els.previewCard.classList.add("hidden");
  els.logOutput.textContent = "";
  setStatus("Analyzing…");
  setProgress(2, "Loading PDF…");

  try {
    const { page } = await loadFirstPage(selectedFile);
    const scale = Number(els.scaleSelect.value);

    const { tiles, overview } = await buildTiles(page, scale);

    els.progressTiles.textContent = `0 / ${tiles.length}`;
    setProgress(45, "Sending tiles to Gemini vision…");
    log("Prepared request", {
      tileCount: tiles.length,
      totalBase64Chars: tiles.reduce((sum, tile) => sum + tile.image.length, 0),
    });

    const response = await fetch(EXTRACT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ tiles, overview }),
    });

    const text = await response.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Worker returned non-JSON response (${response.status}).`);
    }

    if (!response.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail || data)
      );
    }

    setProgress(90, "Processing AI results…");
    log("Worker response", data);

    currentPid = data.pid_no || "";
    currentMeta = data.meta || {};
    currentRows = Array.isArray(data.tags) ? data.tags.map((row) => ({
      tag_no: row.tag_no || "",
      pid_no: row.pid_no || currentPid,
      from: "",
      to: "",
      size_nps_in: row.size_nps_in || "",
      confidence: row.confidence || "",
      evidence: row.evidence || "",
      tile_id: row.tile_id || "",
    })) : [];

    renderResults();

    els.progressTiles.textContent =
      `${data.meta?.tiles_processed ?? tiles.length} / ${tiles.length}`;
    els.progressCandidates.textContent =
      String(data.meta?.raw_candidates ?? currentRows.length);
    els.progressFinal.textContent = String(currentRows.length);

    setProgress(100, "Extraction complete.");
    setStatus("Extraction complete");
  } catch (error) {
    console.error(error);
    log("ERROR", { message: error?.message || String(error) });
    setProgress(100, "Extraction failed.");
    setStatus("Error", false);
    alert(error?.message || "Extraction failed.");
  } finally {
    els.analyzeBtn.disabled = !selectedFile;
  }
}

function renderResults() {
  els.previewCard.classList.remove("hidden");
  els.pidNo.textContent = currentPid || "—";
  els.candidateCount.textContent =
    String(currentMeta.raw_candidates ?? currentRows.length);
  els.tagCount.textContent = String(currentRows.length);

  els.resultsBody.innerHTML = "";

  if (!currentRows.length) {
    els.emptyState.classList.remove("hidden");
    return;
  }

  els.emptyState.classList.add("hidden");

  currentRows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td><input data-field="tag_no" data-index="${index}" value="${escapeHtml(row.tag_no)}"></td>
      <td><input data-field="pid_no" data-index="${index}" value="${escapeHtml(row.pid_no)}"></td>
      <td><input data-field="from" data-index="${index}" value=""></td>
      <td><input data-field="to" data-index="${index}" value=""></td>
      <td><input data-field="size_nps_in" data-index="${index}" value="${escapeHtml(row.size_nps_in)}"></td>
      <td class="confidence">${escapeHtml(row.confidence || "—")}</td>
      <td><button class="delete-btn" data-delete="${index}" type="button" title="Delete">×</button></td>
    `;
    els.resultsBody.appendChild(tr);
  });
}

els.resultsBody.addEventListener("input", (event) => {
  const target = event.target;
  const index = Number(target.dataset.index);
  const field = target.dataset.field;

  if (!Number.isInteger(index) || !field || !currentRows[index]) return;

  currentRows[index][field] = target.value;
});

els.resultsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete]");
  if (!button) return;

  const index = Number(button.dataset.delete);
  if (!Number.isInteger(index)) return;

  currentRows.splice(index, 1);
  renderResults();
});

els.addRowBtn.addEventListener("click", () => {
  currentRows.push({
    tag_no: "",
    pid_no: currentPid,
    from: "",
    to: "",
    size_nps_in: "",
    confidence: "manual",
    evidence: "",
    tile_id: "",
  });
  renderResults();
});

function exportExcel() {
  if (!currentRows.length) {
    alert("There are no rows to export.");
    return;
  }

  if (!window.XLSX) {
    alert("Excel library is not available. Check your internet connection.");
    return;
  }

  const rows = currentRows.map((row) => ({
    "Tag No.": row.tag_no,
    "P&ID No.": row.pid_no || currentPid,
    "From": "",
    "To": "",
    "NPS (in)": row.size_nps_in,
  }));

  const worksheet = XLSX.utils.json_to_sheet(rows);
  worksheet["!cols"] = [
    { wch: 24 },
    { wch: 18 },
    { wch: 18 },
    { wch: 18 },
    { wch: 12 },
  ];

  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Piping Tags");

  const base = selectedFile?.name?.replace(/\.pdf$/i, "") || "piping-tags";
  XLSX.writeFile(workbook, `${base}_piping_tags.xlsx`);
}

els.analyzeBtn.addEventListener("click", analyze);
els.exportBtn.addEventListener("click", exportExcel);

// Load SheetJS without blocking the module startup.
const xlsxScript = document.createElement("script");
xlsxScript.src = "https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js";
xlsxScript.async = true;
document.head.appendChild(xlsxScript);

setStatus("Ready");
