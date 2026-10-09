"use strict";

const state = {
  dashboard: null,
  dataset: null,
  analysisFeature: null,
  analysisLoaded: false,
  risks: null,
  riskMapLoaded: false,
  riskMap: null,
  riskMapInstance: null,
  chsEvaluationLoaded: false,
  chsEvaluation: null
};

const postureColors = { High: "#159b82", Medium: "#e69a22", Low: "#d94a42" };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  $("#toast-container").append(toast);
  window.setTimeout(() => toast.remove(), 4400);
}

function setMessage(selector, message, type = "info") {
  const element = $(selector);
  element.textContent = message;
  element.className = `inline-message ${type}`;
}

function clearMessage(selector) {
  const element = $(selector);
  element.textContent = "";
  element.className = "inline-message hidden";
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}

function setTopLevelError(message) {
  const error = $("#app-error");
  error.textContent = message;
  error.classList.remove("hidden");
}

function clearTopLevelError() {
  $("#app-error").classList.add("hidden");
}

function showTab(name) {
  const target = $(`#tab-${name}`);
  if (!target) return;
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel === target));
  $$(".nav-tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  history.replaceState(null, "", `#${name}`);
  if (window.Plotly) {
    window.setTimeout(() => $$(".chart-area").forEach((chart) => {
      if (chart.data) Plotly.Plots.resize(chart);
    }), 40);
  }
  if (name === "analysis" && !state.analysisLoaded) loadAnalysis();
  if (name === "risks" && !state.risks) loadRisks();
  if (name === "risk-map" && !state.riskMapLoaded) loadRiskMap();
  if (name === "risk-map" && state.riskMapInstance) {
    window.setTimeout(() => state.riskMapInstance.invalidateSize(), 80);
  }
  if (name === "chs-evaluation" && !state.chsEvaluationLoaded) loadCHSEvaluation();
}

function plotOrFallback(id, data, layout, config = {}) {
  const element = $(`#${id}`);
  if (!window.Plotly) {
    element.innerHTML = '<p class="chart-fallback">Interactive charts require an internet connection to load Plotly.</p>';
    return;
  }
  Plotly.react(element, data, {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, Segoe UI, sans-serif", size: 11, color: "#526176" },
    margin: { t: 18, r: 20, b: 42, l: 44 },
    ...layout
  }, { displayModeBar: false, responsive: true, ...config });
}

function metricRows(comparison, selectedName) {
  if (!comparison?.length) return '<tr><td colspan="6" class="table-empty">No model evaluation results are available. Train the models to populate this table.</td></tr>';
  return comparison.map((model) => `
    <tr>
      <td class="strong-cell">${escapeHtml(model.name)}</td>
      <td>${formatMetric(model.accuracy)}</td><td>${formatMetric(model.precision)}</td>
      <td>${formatMetric(model.recall)}</td><td>${formatMetric(model.f1_score)}</td>
      <td>${model.name === selectedName
        ? '<span class="status-chip status-selected"><i class="fa-solid fa-check"></i>Selected</span>'
        : '<span class="status-chip">Evaluated</span>'}</td>
    </tr>`).join("");
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

function updateComparisonViews(model) {
  const comparison = model?.comparison || [];
  const selectedName = model?.name || null;
  const rowHtml = metricRows(comparison, selectedName);
  $("#dashboard-model-table").innerHTML = rowHtml;
  $("#training-table").innerHTML = rowHtml;
  $("#management-table").innerHTML = rowHtml;
  $("#training-selected").textContent = selectedName
    ? `${selectedName} selected · evaluated on ${model.test_samples ?? "held-out"} hold-out records${model.is_current === false ? " · results are from the previous dataset; retrain to refresh" : ""}`
    : "No training run recorded yet.";

  if (!comparison.length) {
    $("#chart-comparison").innerHTML = '<p class="chart-fallback">Train a model to see comparison metrics.</p>';
    return;
  }
  const metricNames = ["accuracy", "precision", "recall", "f1_score"];
  const labels = ["Accuracy", "Precision", "Recall", "F1 score"];
  const colors = ["#3177dc", "#24a994", "#e5a12a", "#8c72d9"];
  const traces = comparison.map((entry) => ({
    type: "bar",
    name: entry.name,
    x: labels,
    y: metricNames.map((key) => Number(entry[key] || 0)),
    marker: { color: entry.name === selectedName ? "#159b82" : colors[comparison.indexOf(entry) % colors.length] },
    hovertemplate: `${escapeHtml(entry.name)}<br>%{x}: %{y:.1%}<extra></extra>`
  }));
  plotOrFallback("chart-comparison", traces, {
    barmode: "group",
    yaxis: { range: [0, 1], tickformat: ".0%", gridcolor: "#edf1f5", zeroline: false },
    xaxis: { gridcolor: "rgba(0,0,0,0)" },
    legend: { orientation: "h", y: -0.25, x: 0 }
  });
}

function renderDashboard(dashboard) {
  state.dashboard = dashboard;
  clearTopLevelError();
  $("#header-dataset").textContent = dashboard.dataset;
  $("#header-model").textContent = dashboard.model.available
    ? (dashboard.model.is_current ? dashboard.model.name : "Model needs retraining")
    : "Model unavailable";
  $("#demo-notice").classList.toggle("hidden", !dashboard.is_demo);
  $("#stat-records").textContent = Number(dashboard.rows).toLocaleString();
  $("#stat-score").textContent = `${Number(dashboard.mean_score).toFixed(1)} / 100`;
  $("#stat-high").textContent = Number(dashboard.posture_counts.High || 0).toLocaleString();
  $("#stat-low").textContent = Number(dashboard.posture_counts.Low || 0).toLocaleString();
  $("#risk-high-count").textContent = Number(dashboard.risk_counts.high || 0).toLocaleString();
  $("#risk-medium-count").textContent = Number(dashboard.risk_counts.medium || 0).toLocaleString();
  $("#risk-low-count").textContent = Number(dashboard.risk_counts.low || 0).toLocaleString();
  $("#dashboard-model-name").textContent = dashboard.model.name
    ? `${dashboard.model.name}${dashboard.model.is_current ? "" : " · previous dataset"}`
    : (dashboard.model.error || "No trained model");
  $("#dashboard-model-f1").textContent = dashboard.model.metrics?.f1_score == null
    ? "—"
    : `${(Number(dashboard.model.metrics.f1_score) * 100).toFixed(1)}%`;
  $("#chart-metrics").classList.toggle("chart-missing", !dashboard.model.metrics);
  $("#management-model-summary").innerHTML = dashboard.model.available
    ? `<strong class="selected-model-title">${escapeHtml(dashboard.model.name)}</strong>
       <div class="selected-model-meta"><span class="status-chip ${dashboard.model.is_current ? "status-selected" : "posture-medium"}"><i class="fa-solid ${dashboard.model.is_current ? "fa-check" : "fa-rotate"}"></i>${dashboard.model.is_current ? "Current dataset" : "Retraining recommended"}</span><span class="status-chip">${Number(dashboard.model.training_rows || 0).toLocaleString()} training records</span><span class="status-chip">Test split: ${Number(dashboard.model.test_samples || 0).toLocaleString()}</span></div>
       <span class="model-name">Trained from ${escapeHtml(dashboard.model.trained_on || "dataset not recorded")}</span>
       <span class="model-name">Saved locally at models/securemailscope.joblib</span>`
    : `<span class="model-state-error">${escapeHtml(dashboard.model.error || "No trained model is available.")}</span>
       <span class="model-name">Train the models when a valid labeled dataset is available.</span>`;

  const postures = ["High", "Medium", "Low"];
  const counts = postures.map((level) => Number(dashboard.posture_counts[level] || 0));
  const postureTrace = {
    type: "pie",
    hole: 0.62,
    labels: postures,
    values: counts,
    sort: false,
    marker: { colors: postures.map((level) => postureColors[level]) },
    textinfo: "label+percent",
    hovertemplate: "%{label}: %{value} records (%{percent})<extra></extra>"
  };
  plotOrFallback("chart-posture", [postureTrace], {
    showlegend: true,
    legend: { orientation: "h", y: -0.1, x: 0.1 },
    margin: { t: 5, r: 10, b: 30, l: 10 }
  });

  const metrics = dashboard.model.metrics;
  if (metrics) {
    plotOrFallback("chart-metrics", [{
      type: "bar",
      x: ["Accuracy", "Precision", "Recall", "F1 score"],
      y: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score].map(Number),
      marker: { color: ["#3177dc", "#159b82", "#4e9ba8", "#84bfa7"], line: { width: 0 } },
      text: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score].map((value) => `${(Number(value) * 100).toFixed(1)}%`),
      textposition: "outside",
      hovertemplate: "%{x}: %{y:.1%}<extra></extra>"
    }], {
      yaxis: { range: [0, 1.15], tickformat: ".0%", gridcolor: "#edf1f5", zeroline: false },
      xaxis: { gridcolor: "rgba(0,0,0,0)" },
      margin: { t: 24, r: 10, b: 42, l: 38 }
    });
  } else {
    $("#chart-metrics").innerHTML = `<p class="chart-fallback">${escapeHtml(dashboard.model.error || "Train a classifier to view metrics.")}</p>`;
  }
  updateComparisonViews(dashboard.model);
  renderLandscape(dashboard.landscape || []);
  renderDatasetSummary(dashboard);
}

function renderLandscape(records) {
  const container = $("#security-landscape");
  if (!records.length) {
    container.innerHTML = '<p class="chart-fallback">No configuration records are available.</p>';
    return;
  }
  container.innerHTML = records.map((record) => {
    const label = `${record.posture} posture · record ${record.record} · rule-based score ${record.score}/100`;
    return `<span class="landscape-marker marker-${escapeHtml(record.posture.toLowerCase())}" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}"></span>`;
  }).join("");
  container.style.setProperty("--marker-count", Math.min(12, records.length));
}

function renderDatasetSummary(dashboard) {
  $("#dataset-name").textContent = dashboard.dataset;
  $("#dataset-rows").textContent = Number(dashboard.rows).toLocaleString();
  $("#dataset-columns").textContent = dashboard.columns;
}

function renderDataset(dataset) {
  state.dataset = dataset;
  $("#dataset-name").textContent = dataset.dataset;
  $("#dataset-rows").textContent = Number(dataset.rows).toLocaleString();
  $("#dataset-columns").textContent = dataset.columns;
  const missingTotal = dataset.missing.reduce((total, item) => total + Number(item.count), 0);
  $("#missing-total").textContent = missingTotal.toLocaleString();
  $("#missing-table").innerHTML = dataset.missing.map((item) => `
    <tr><td>${escapeHtml(item.feature.replaceAll("_", " "))}</td><td>${Number(item.count).toLocaleString()}</td><td>${Number(item.percent).toFixed(1)}%</td></tr>
  `).join("");
  $("#statistics-table").innerHTML = dataset.statistics.map((item) => {
    const summary = item.type === "numeric"
      ? `Mean ${item.mean ?? "—"} · median ${item["50%"] ?? "—"} · range ${item.min ?? "—"}–${item.max ?? "—"}`
      : `Top: ${item.top || "—"} (${item.top_frequency ?? 0}) · ${item.unique ?? 0} unique`;
    return `<tr><td>${escapeHtml(item.feature.replaceAll("_", " "))}</td><td>${escapeHtml(item.type)}</td><td>${Number(item.count).toLocaleString()}</td><td>${escapeHtml(summary)}</td></tr>`;
  }).join("");

  if (!dataset.preview.length) {
    $("#preview-table").innerHTML = '<p class="table-empty">No preview rows are available.</p>';
  } else {
    const columns = Object.keys(dataset.preview[0]);
    $("#preview-table").innerHTML = `<table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column.replaceAll("_", " ").toUpperCase())}</th>`).join("")}</tr></thead><tbody>${
      dataset.preview.map((record) => `<tr>${columns.map((column) => `<td>${escapeHtml(record[column] ?? "—")}</td>`).join("")}</tr>`).join("")
    }</tbody></table>`;
  }

  const selector = $("#analysis-feature");
  const selected = state.analysisFeature || selector.value;
  selector.innerHTML = dataset.features.map((feature) => `<option value="${escapeHtml(feature)}">${escapeHtml(feature.replaceAll("_", " "))}</option>`).join("");
  if (dataset.features.includes(selected)) selector.value = selected;
  state.analysisFeature = selector.value;
}

function renderAnalysis(data) {
  state.analysisLoaded = true;
  state.analysisFeature = data.feature;
  $("#analysis-feature").value = data.feature;
  plotOrFallback("chart-feature", [{
    type: data.numeric_description && Object.keys(data.numeric_description).length ? "histogram" : "bar",
    x: data.numeric_description && Object.keys(data.numeric_description).length ? data.raw_values : data.labels,
    y: data.numeric_description && Object.keys(data.numeric_description).length ? undefined : data.counts,
    marker: { color: "#3177dc" },
    hovertemplate: data.numeric_description && Object.keys(data.numeric_description).length
      ? "Value: %{x}<br>Records: %{y}<extra></extra>"
      : "%{x}: %{y} records<extra></extra>"
  }], {
    xaxis: { title: data.feature.replaceAll("_", " "), gridcolor: "rgba(0,0,0,0)" },
    yaxis: { title: "Records", gridcolor: "#edf1f5", zeroline: false },
    margin: { t: 8, r: 16, b: 55, l: 48 }
  });
  plotOrFallback("chart-correlation", [{
    type: "heatmap",
    x: data.correlation_labels.map((item) => item.replaceAll("_", " ")),
    y: data.correlation_labels.map((item) => item.replaceAll("_", " ")),
    z: data.correlation,
    zmin: -1,
    zmax: 1,
    colorscale: [[0, "#d94a42"], [0.5, "#f4d477"], [1, "#159b82"]],
    colorbar: { title: "r" },
    hovertemplate: "%{y} × %{x}<br>Encoded correlation: %{z:.2f}<extra></extra>"
  }], {
    margin: { t: 8, r: 14, b: 110, l: 145 },
    xaxis: { tickangle: -35, automargin: true },
    yaxis: { automargin: true, autorange: "reversed" }
  });
}

async function loadAnalysis() {
  const feature = $("#analysis-feature").value || "tls_version";
  try {
    const result = await api(`/api/analysis?feature=${encodeURIComponent(feature)}`);
    renderAnalysis(result);
  } catch (error) {
    setTopLevelError(error.message);
  }
}

function renderRisks(risks) {
  state.risks = risks;
  const issueRows = risks.issue_counts.length
    ? risks.issue_counts.map((item) => `<tr><td>${escapeHtml(item.issue)}</td><td><span class="count-badge">${Number(item.count).toLocaleString()}</span></td></tr>`).join("")
    : '<tr><td colspan="2" class="table-empty">No rule-based findings were detected in the current dataset.</td></tr>';
  $("#findings-table").innerHTML = issueRows;
  $("#risk-records").innerHTML = risks.records.length
    ? risks.records.map((item) => `<tr><td>#${item.record}</td><td><span class="posture-chip posture-${escapeHtml(item.posture.toLowerCase())}">${escapeHtml(item.posture)}</span></td><td>${item.score}/100</td><td><span class="severity-chip severity-${escapeHtml(item.severity)}">${escapeHtml(item.severity)}</span></td><td>${escapeHtml(item.issues)}</td></tr>`).join("")
    : '<tr><td colspan="5" class="table-empty">No records with rule-based findings are available.</td></tr>';
}

async function loadRisks() {
  try {
    renderRisks(await api("/api/risks"));
  } catch (error) {
    setTopLevelError(error.message);
  }
}

function mapText(value, missingLabel = "Not provided") {
  return value == null || value === "" ? missingLabel : String(value);
}

function assetPopupHtml(asset) {
  const riskName = { high: "High Risk", medium: "Medium Risk", low: "Low Risk" }[asset.risk_level] || "Risk not classified";
  const confidence = asset.model_probability == null
    ? '<p><strong>Model probability:</strong> Not available</p>'
    : `<p><strong>Model probability:</strong> ${(Number(asset.model_probability) * 100).toFixed(1)}% <small>(${escapeHtml(asset.model_probability_label)})</small></p>`;
  const deprecated = Array.isArray(asset.deprecated_algorithms) && asset.deprecated_algorithms.length
    ? asset.deprecated_algorithms.map(escapeHtml).join(", ")
    : "None identified in the supplied algorithm fields";
  const config = asset.configuration || {};
  return `<div class="risk-map-popup">
    <h3>Asset #${escapeHtml(asset.asset_id)}: ${escapeHtml(asset.asset_name)}</h3>
    <p><strong>Site coordinates:</strong> ${Number(asset.latitude).toFixed(5)}, ${Number(asset.longitude).toFixed(5)} <span class="synthetic-coordinate">(synthetic)</span></p>
    <p><strong>Predicted Posture Risk:</strong> ${riskName}${asset.posture_prediction ? ` <small>(predicted posture: ${escapeHtml(asset.posture_prediction)})</small>` : ""}</p>
    ${confidence}
    <p><strong>Risk source:</strong> ${escapeHtml(asset.risk_source)}</p>
    <p><strong>Compliance Status:</strong> ${escapeHtml(asset.configuration_checks)}</p>
    <p><strong>Deprecated Algorithms:</strong> ${deprecated}</p>
    <p><strong>Certificate Status / expiry:</strong> ${escapeHtml(asset.certificate_status)}</p>
    <p><strong>PQC Readiness:</strong> ${escapeHtml(asset.pqc_readiness)}</p>
    <p><strong>Key Usage:</strong> ${escapeHtml(asset.key_usage)}</p>
    <p><strong>RNG Status:</strong> ${escapeHtml(asset.rng_status)}</p>
    <details><summary>Supplied configuration</summary><p>TLS: ${escapeHtml(mapText(config.tls_version))}; encryption: ${escapeHtml(mapText(config.encryption_mechanism))}; algorithm: ${escapeHtml(mapText(config.cryptographic_algorithm))}; key strength: ${escapeHtml(mapText(config.key_strength))} bits; authentication: ${escapeHtml(mapText(config.authentication_method))}; cipher suite: ${escapeHtml(mapText(config.cipher_suite))}</p></details>
    <small class="risk-map-popup-footnote">Demo visualization only. Coordinates are generated and are not real infrastructure locations.</small>
  </div>`;
}

function addRiskMapLegend(map) {
  const legend = window.L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const container = window.L.DomUtil.create("div", "leaflet-control risk-map-map-legend");
    container.innerHTML = `<strong>Reference indicators</strong>
      <span><i class="risk-map-dot risk-dot-high"></i>FIPS / NIST-PQC controls not assessed</span>
      <span><i class="risk-map-dot risk-dot-medium"></i>NIST/PQC references only</span>
      <span><i class="risk-map-dot risk-dot-low"></i>PQC fields not present in dataset</span>`;
    window.L.DomEvent.disableClickPropagation(container);
    return container;
  };
  legend.addTo(map);
}

function renderRiskMap(result) {
  state.riskMap = result;
  $("#risk-map-source-note").textContent = result.source_note;
  $("#risk-map-count").textContent = `${Number(result.asset_count).toLocaleString()} mapped assets · synthetic coordinates`;
  $("#risk-map-model-note").textContent = result.model_message;
  const container = $("#cryptographic-risk-map");
  if (!window.L) {
    container.innerHTML = '<div class="map-error">The map library could not load. Check your internet connection and reload this page.</div>';
    return;
  }
  if (!state.riskMapInstance) {
    state.riskMapInstance = window.L.map(container, {
      scrollWheelZoom: true
    }).setView([20.8, 79.5], 7);
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
    }).addTo(state.riskMapInstance);
    addRiskMapLegend(state.riskMapInstance);
  }

  if (state.riskMapMarkers) state.riskMapMarkers.clearLayers();
  else state.riskMapMarkers = window.L.featureGroup().addTo(state.riskMapInstance);
  const colors = { high: "#eb7541", medium: "#f2b544", low: "#31a993" };
  result.assets.forEach((asset) => {
    const latitude = Number(asset.latitude);
    const longitude = Number(asset.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)
        || latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return;
    const level = ["high", "medium", "low"].includes(asset.risk_level) ? asset.risk_level : "medium";
    window.L.circleMarker([latitude, longitude], {
      radius: 7,
      color: "#354c56",
      weight: 1,
      opacity: 0.9,
      fillColor: colors[level],
      fillOpacity: 0.82
    }).bindPopup(assetPopupHtml(asset), { maxWidth: 370, minWidth: 265 }).addTo(state.riskMapMarkers);
  });
  if (result.assets.length) {
    state.riskMapInstance.fitBounds(state.riskMapMarkers.getBounds().pad(0.16), { maxZoom: 9 });
  }
  window.setTimeout(() => state.riskMapInstance.invalidateSize(), 80);
  state.riskMapLoaded = true;
}

async function loadRiskMap() {
  clearMessage("#risk-map-error");
  try {
    renderRiskMap(await api("/api/risk-map"));
  } catch (error) {
    setMessage("#risk-map-error", error.message, "error");
    $("#cryptographic-risk-map").innerHTML = `<div class="map-error">${escapeHtml(error.message)}</div>`;
  }
}

function renderCHSEvaluation(evaluation) {
  state.chsEvaluation = evaluation;
  const { metrics, points } = evaluation;
  $("#chs-mae").textContent = Number(metrics.mae).toFixed(2);
  $("#chs-mse").textContent = Number(metrics.mse).toFixed(2);
  $("#chs-rmse").textContent = Number(metrics.rmse).toFixed(2);
  $("#chs-r2").textContent = Number(metrics.r2_score).toFixed(3);
  $("#chs-data-status span").textContent = evaluation.metric_note;
  $("#chs-plot-note").textContent = evaluation.plot_note;

  const actual = points.map((point) => Number(point.actual));
  const predicted = points.map((point) => Number(point.predicted));
  const isMobile = window.matchMedia("(max-width: 760px)").matches;
  plotOrFallback("chart-chs", [
    {
      type: "scatter",
      mode: "markers",
      name: "Predictions",
      x: actual,
      y: predicted,
      marker: { color: "#84c8e3", size: 8, opacity: 0.9 },
      hovertemplate: "Actual CHS: %{x}<br>Predicted CHS: %{y}<extra>Predictions · reference/demo</extra>"
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Ideal 1:1 Fit",
      x: [12, 60],
      y: [12, 60],
      line: { color: "#168d7a", width: 2.5, dash: "dash" },
      hovertemplate: "Ideal fit: %{x}<extra></extra>"
    }
  ], {
    xaxis: {
      title: { text: "Actual Ground Truth CHS", standoff: 12 },
      range: [10, 61],
      dtick: 10,
      gridcolor: "#e6ebf0",
      zeroline: false,
      automargin: true
    },
    yaxis: {
      title: { text: "Model Predicted CHS", standoff: 10 },
      range: [10, 61],
      dtick: 10,
      gridcolor: "#e6ebf0",
      zeroline: false,
      automargin: true
    },
    legend: isMobile
      ? { orientation: "h", x: 0, y: -0.32 }
      : { orientation: "v", x: 1.02, y: 1, xanchor: "left" },
    margin: isMobile
      ? { t: 18, r: 12, b: 88, l: 56 }
      : { t: 18, r: 155, b: 64, l: 66 },
    showlegend: true
  });
  state.chsEvaluationLoaded = true;
}

async function loadCHSEvaluation() {
  clearMessage("#chs-evaluation-error");
  try {
    const evaluation = await api("/api/chs-evaluation");
    renderCHSEvaluation(evaluation);
  } catch (error) {
    setMessage("#chs-evaluation-error", error.message, "error");
  }
}

async function loadEverything() {
  try {
    const [dashboard, dataset] = await Promise.all([api("/api/dashboard"), api("/api/dataset")]);
    renderDashboard(dashboard);
    renderDataset(dataset);
    if ($("#tab-risks").classList.contains("active")) {
      state.risks = null;
      await loadRisks();
    }
    if ($("#tab-analysis").classList.contains("active")) {
      state.analysisLoaded = false;
      await loadAnalysis();
    }
    clearTopLevelError();
  } catch (error) {
    setTopLevelError(error.message);
  }
}

function selectUpload(file) {
  if (!file) return;
  $("#csv-file").files = new DataTransfer().files;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $("#csv-file").files = transfer.files;
  $("#selected-file-name").textContent = file.name;
  $("#upload-button").disabled = false;
  clearMessage("#upload-message");
}

async function uploadDataset(event) {
  event.preventDefault();
  const file = $("#csv-file").files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    setMessage("#upload-message", "Choose a file with a .csv extension.", "error");
    return;
  }
  const button = $("#upload-button");
  button.disabled = true;
  button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>Validating…';
  const body = new FormData();
  body.append("file", file);
  try {
    const result = await api("/api/upload", { method: "POST", body });
    setMessage("#upload-message", `${result.message} ${result.rows.toLocaleString()} rows and ${result.columns} columns loaded.`, "success");
    showToast("Dataset uploaded successfully.");
    state.analysisLoaded = false;
    state.risks = null;
    state.riskMapLoaded = false;
    await loadEverything();
    if ($("#tab-risk-map").classList.contains("active")) await loadRiskMap();
  } catch (error) {
    setMessage("#upload-message", error.message, "error");
  } finally {
    button.disabled = false;
    button.innerHTML = '<i class="fa-solid fa-upload"></i>Upload and validate';
  }
}

async function trainModels() {
  const activeMessage = $("#tab-management").classList.contains("active")
    ? "#management-message"
    : "#training-message";
  setMessage(activeMessage, "Training and comparing classifiers on the active dataset…", "info");
  const buttons = $$('[data-action="train"]');
  buttons.forEach((button) => {
    button.disabled = true;
    button.dataset.oldText = button.innerHTML;
    button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>Training…';
  });
  try {
    const result = await api("/api/train", { method: "POST" });
    showToast(`Saved ${result.selected_model} as the selected model.`);
    setMessage(activeMessage, result.message, "success");
    state.analysisLoaded = false;
    state.risks = null;
    state.riskMapLoaded = false;
    await loadEverything();
    if ($("#tab-risk-map").classList.contains("active")) await loadRiskMap();
  } catch (error) {
    showToast(error.message, "error");
    setMessage(activeMessage, error.message, "error");
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
      button.innerHTML = button.dataset.oldText || '<i class="fa-solid fa-rotate"></i>Retrain models';
    });
  }
}

function renderAssessment(result) {
  const rating = result.prediction.toLowerCase();
  const findings = result.findings.length
    ? result.findings.map((item) => `<li class="finding-item"><span class="severity-chip severity-${escapeHtml(item.severity)}">${escapeHtml(item.severity)}</span><div><strong>${escapeHtml(item.issue)}</strong><p>${escapeHtml(item.detail)}</p><small><b>Recommendation:</b> ${escapeHtml(item.recommendation)}</small></div></li>`).join("")
    : '<li class="no-findings"><i class="fa-solid fa-circle-check"></i><span>No rule-based weaknesses were identified for the submitted values. This is not a guarantee of security.</span></li>';
  const probabilities = Object.entries(result.class_probabilities || {}).map(([label, value]) =>
    `<span class="probability-chip">${escapeHtml(label)} <strong>${(Number(value) * 100).toFixed(1)}%</strong></span>`
  ).join("");
  $("#assessment-result").innerHTML = `
    <div class="result-heading"><div><p class="eyebrow">ASSESSMENT RESULT</p><h3>Configuration assessment</h3></div><span class="posture-chip posture-${rating}">${escapeHtml(result.prediction)} posture</span></div>
    <div class="score-row"><div class="score-circle score-${rating}"><strong>${result.score}</strong><small>/ 100</small></div><div><span class="stat-label">RULE-BASED SECURITY SCORE</span><p class="score-explainer">Rule-based score indicates <strong>${escapeHtml(result.score_posture)}</strong> posture</p><span class="model-name">Predicted by ${escapeHtml(result.model_name)}</span></div></div>
    ${result.confidence == null ? "" : `<div class="confidence-row"><span>Model confidence</span><strong>${(Number(result.confidence) * 100).toFixed(1)}%</strong></div>`}
    ${probabilities ? `<div class="probability-row">${probabilities}</div>` : ""}
    <div class="result-note"><i class="fa-solid fa-circle-info"></i>${escapeHtml(result.model_note)}</div>
    <div class="result-findings"><h4>Findings &amp; recommendations <span>${result.findings.length}</span></h4><ul>${findings}</ul></div>`;
}

async function submitAssessment(event) {
  event.preventDefault();
  clearMessage("#assessment-error");
  const form = event.currentTarget;
  const submit = $('button[type="submit"]', form);
  if (!form.reportValidity()) return;
  const config = Object.fromEntries(new FormData(form).entries());
  submit.disabled = true;
  submit.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>Assessing…';
  try {
    renderAssessment(await api("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    }));
    showToast("Configuration assessment complete.");
  } catch (error) {
    setMessage("#assessment-error", error.message, "error");
  } finally {
    submit.disabled = false;
    submit.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i>Assess configuration';
  }
}

function download(path) {
  window.location.assign(path);
}

function bindEvents() {
  $$(".nav-tab").forEach((button) => button.addEventListener("click", () => showTab(button.dataset.tab)));
  $$("[data-open-tab]").forEach((button) => button.addEventListener("click", () => showTab(button.dataset.openTab)));
  $$('[data-action="download"]').forEach((button) => button.addEventListener("click", () => download("/download/dataset")));
  $$('[data-action="download-demo"]').forEach((button) => button.addEventListener("click", () => download("/download/demo")));
  $$('[data-action="train"]').forEach((button) => button.addEventListener("click", trainModels));

  const dropzone = $("#dropzone");
  dropzone.addEventListener("click", () => $("#csv-file").click());
  $("#csv-file").addEventListener("change", (event) => selectUpload(event.target.files[0]));
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    selectUpload(event.dataTransfer.files[0]);
  });
  $("#upload-form").addEventListener("submit", uploadDataset);
  $("#analysis-feature").addEventListener("change", () => {
    state.analysisFeature = $("#analysis-feature").value;
    loadAnalysis();
  });
  $("#assessment-form").addEventListener("submit", submitAssessment);
  $("#assessment-form").addEventListener("reset", () => {
    clearMessage("#assessment-error");
    $("#assessment-result").innerHTML = '<div class="empty-result"><span><i class="fa-solid fa-shield-halved"></i></span><h3>Assessment results</h3><p>Submit a configuration to see the model prediction, score, and specific recommendations.</p></div>';
  });
  let chsResizeTimer;
  window.addEventListener("resize", () => {
    if (state.riskMapInstance && $("#tab-risk-map").classList.contains("active")) {
      window.setTimeout(() => state.riskMapInstance.invalidateSize(), 80);
    }
    if (!state.chsEvaluation || !$("#tab-chs-evaluation").classList.contains("active")) return;
    window.clearTimeout(chsResizeTimer);
    chsResizeTimer = window.setTimeout(() => renderCHSEvaluation(state.chsEvaluation), 140);
  });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  const initial = location.hash.slice(1);
  if (initial && $(`#tab-${initial}`)) showTab(initial);
  loadEverything();
});
