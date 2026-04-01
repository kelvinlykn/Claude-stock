// StockPredict AI v8 - Full-featured frontend

const MODEL_COLORS = {
    "Ensemble": "#f59e0b",
    "LSTM": "#3b82f6",
    "XGBoost": "#10b981",
    "LightGBM": "#8b5cf6",
    "Random Forest": "#ef4444",
    "Gradient Boosting": "#06b6d4",
    "SVR": "#ec4899",
    "Ridge Regression": "#f97316",
    "ARIMA/SARIMAX": "#14b8a6",
    "Prophet": "#a855f7"
};

const PLOTLY_LAYOUT = {
    paper_bgcolor: "#1a2234",
    plot_bgcolor: "#111827",
    font: { color: "#e2e8f0", family: "-apple-system, BlinkMacSystemFont, sans-serif", size: 12 },
    xaxis: { gridcolor: "#2a3a52", linecolor: "#2a3a52" },
    yaxis: { gridcolor: "#2a3a52", linecolor: "#2a3a52" },
    margin: { t: 30, r: 30, b: 50, l: 70 },
    legend: { bgcolor: "rgba(26, 34, 52, 0.9)", bordercolor: "#2a3a52", font: { size: 11 } },
    hovermode: "x unified"
};

let currentData = null;
let visibleModels = new Set();

// ==================== TAB MANAGEMENT ====================

function switchTab(tabName) {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
    document.getElementById(`tab-${tabName}`).classList.add("active");
}

// ==================== UTILITIES ====================

function setLoading(id, show) {
    document.getElementById(id).classList.toggle("hidden", !show);
}

function showError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = msg;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 8000);
}

function hideAllPredictionSections() {
    ["stockInfo", "reliabilityWarning", "confidenceSection", "predictionSection", "monteCarloSection",
     "accuracySection", "ensembleSection", "backtestSection", "technicalSection",
     "summarySection"].forEach(id => document.getElementById(id).classList.add("hidden"));
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// ==================== PREDICTION ====================

async function runPrediction() {
    const ticker = document.getElementById("ticker").value.trim();
    const days = parseInt(document.getElementById("days").value);
    if (!ticker) { showError("error", "Please enter a stock ticker."); return; }

    setLoading("loading", true);
    hideAllPredictionSections();

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker, days })
        });
        const data = await res.json();
        if (data.error) { showError("error", data.error); return; }
        currentData = data;
        renderPrediction(data);
    } catch (e) {
        showError("error", "Connection failed: " + e.message);
    } finally {
        setLoading("loading", false);
    }
}

function renderPrediction(data) {
    // Stock info
    document.getElementById("stockInfo").classList.remove("hidden");
    document.getElementById("infoTicker").textContent = data.ticker;
    document.getElementById("infoPrice").textContent = "$" + data.current_price.toFixed(2);
    const changeEl = document.getElementById("infoChange");
    changeEl.textContent = (data.price_change_1d >= 0 ? "+" : "") + data.price_change_1d.toFixed(2) + "%";
    changeEl.style.color = data.price_change_1d >= 0 ? "#10b981" : "#ef4444";
    document.getElementById("infoUpdated").textContent = data.last_updated;

    const score = data.confidence_score || 0;
    document.getElementById("infoConfidence").textContent = score + "/100";
    document.getElementById("infoConfidence").style.color =
        score >= 60 ? "#10b981" : score >= 40 ? "#f59e0b" : "#ef4444";

    // Reliability warning banner
    if (data.reliability_warning) {
        const warningEl = document.getElementById("reliabilityWarning");
        warningEl.textContent = data.reliability_warning;
        warningEl.className = "reliability-warning";
        const days = data.prediction_dates ? data.prediction_dates.length : 0;
        if (days <= 7) warningEl.classList.add("reliability-high");
        else if (days <= 14) warningEl.classList.add("reliability-moderate");
        else if (days <= 30) warningEl.classList.add("reliability-low");
        else warningEl.classList.add("reliability-very-low");
        warningEl.classList.remove("hidden");
    }

    renderConfidenceGauge(data);
    visibleModels = new Set(["Ensemble"]);
    renderModelToggles(data);
    renderPredictionChart(data);
    renderMonteCarloChart(data);
    renderAccuracyChart(data);
    renderEnsembleChart(data);
    renderSummaryTable(data);

    ["confidenceSection", "predictionSection", "monteCarloSection",
     "accuracySection", "ensembleSection", "summarySection"].forEach(id =>
        document.getElementById(id).classList.remove("hidden"));
}

function renderConfidenceGauge(data) {
    const score = data.confidence_score || 0;
    const gauge = document.getElementById("confidenceGauge");
    const color = score >= 60 ? "#10b981" : score >= 40 ? "#f59e0b" : "#ef4444";
    gauge.style.background = `conic-gradient(${color} ${score * 3.6}deg, #1f2937 ${score * 3.6}deg)`;
    document.getElementById("gaugeText").textContent = Math.round(score);
    document.getElementById("gaugeText").style.color = color;

    let desc = score >= 70 ? "High confidence. Models agree on direction with low error." :
               score >= 50 ? "Moderate confidence. Reasonable model agreement." :
               score >= 30 ? "Low confidence. Significant model disagreement." :
               "Very low confidence. Highly uncertain predictions.";
    document.getElementById("confidenceDesc").textContent = desc;

    const mc = data.monte_carlo;
    if (mc) {
        document.getElementById("mcProb").innerHTML =
            `<strong>Monte Carlo (${mc.n_simulations} simulations):</strong> ` +
            `${mc.prob_up}% probability of increase | Expected: ${mc.expected_return >= 0 ? "+" : ""}${mc.expected_return}%`;
    }
}

function renderModelToggles(data) {
    const container = document.getElementById("modelToggles");
    container.innerHTML = "";
    Object.keys(data.predictions).forEach(name => {
        const btn = document.createElement("button");
        const isActive = visibleModels.has(name);
        btn.className = "model-toggle" + (isActive ? " active" : "");
        btn.textContent = name;
        const color = MODEL_COLORS[name] || "#3b82f6";
        if (isActive) btn.style.background = color;
        btn.onclick = () => {
            visibleModels.has(name) ? visibleModels.delete(name) : visibleModels.add(name);
            btn.classList.toggle("active");
            btn.style.background = visibleModels.has(name) ? color : "";
            renderPredictionChart(data);
        };
        container.appendChild(btn);
    });
}

function renderPredictionChart(data) {
    const traces = [];
    const showCI = document.getElementById("showCI").checked;

    traces.push({
        x: data.historical.dates, y: data.historical.prices,
        name: "Historical", type: "scatter", mode: "lines",
        line: { color: "#64748b", width: 2 }
    });

    const lastDate = data.historical.dates[data.historical.dates.length - 1];
    const lastPrice = data.historical.prices[data.historical.prices.length - 1];

    Object.keys(data.predictions).forEach(name => {
        if (!visibleModels.has(name)) return;
        const preds = data.predictions[name];
        const dates = [lastDate, ...data.prediction_dates.slice(0, preds.length)];
        const prices = [lastPrice, ...preds];
        const color = MODEL_COLORS[name] || "#3b82f6";

        if (showCI && data.confidence_intervals && data.confidence_intervals[name]) {
            const ci = data.confidence_intervals[name];
            const ciDates = [lastDate, ...data.prediction_dates.slice(0, ci.upper.length)];
            const upper = [lastPrice, ...ci.upper];
            const lower = [lastPrice, ...ci.lower];
            traces.push({
                x: ciDates.concat([...ciDates].reverse()),
                y: upper.concat([...lower].reverse()),
                fill: "toself", fillcolor: hexToRgba(color, 0.1),
                line: { color: "transparent" }, name: name + " CI",
                showlegend: false, hoverinfo: "skip"
            });
        }

        traces.push({
            x: dates, y: prices, name, type: "scatter", mode: "lines",
            line: { color, width: name === "Ensemble" ? 3 : 1.5, dash: name === "Ensemble" ? "solid" : "dot" }
        });
    });

    // Confidence decay shading: overlay rectangles with increasing red opacity
    const decayShapes = [{ type: "line", x0: lastDate, x1: lastDate, y0: 0, y1: 1, yref: "paper",
                           line: { color: "#f59e0b", width: 1, dash: "dash" } }];
    if (data.confidence_decay && data.prediction_dates) {
        const decay = data.confidence_decay;
        const pDates = data.prediction_dates;
        // Group days into bands to avoid too many shapes
        const bandSize = Math.max(1, Math.floor(pDates.length / 20));
        for (let i = 0; i < pDates.length; i += bandSize) {
            const endIdx = Math.min(i + bandSize, pDates.length) - 1;
            const avgDecay = decay[Math.floor((i + endIdx) / 2)] || decay[i];
            const opacity = Math.max(0, (1.0 - avgDecay) * 0.15);
            if (opacity > 0.005) {
                decayShapes.push({
                    type: "rect", x0: pDates[i], x1: pDates[endIdx],
                    y0: 0, y1: 1, yref: "paper",
                    fillcolor: `rgba(239, 68, 68, ${opacity.toFixed(4)})`,
                    line: { width: 0 }, layer: "below"
                });
            }
        }
    }

    Plotly.newPlot("predictionChart", traces, {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: "Date" },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Price ($)" },
        shapes: decayShapes,
        annotations: [{ x: lastDate, y: 1, yref: "paper", text: "Prediction Start",
                       showarrow: false, font: { color: "#f59e0b", size: 10 }, yshift: 10 }]
    }, { responsive: true });
}

function renderMonteCarloChart(data) {
    if (!data.monte_carlo) return;
    const mc = data.monte_carlo;
    const dates = data.prediction_dates.slice(0, mc.mean.length);
    const lastDate = data.historical.dates[data.historical.dates.length - 1];
    const lastPrice = data.current_price;
    const allDates = [lastDate, ...dates];

    document.getElementById("mcDesc").textContent =
        `${mc.n_simulations} simulations. Shaded: 10th-90th and 25th-75th percentiles.`;

    Plotly.newPlot("monteCarloChart", [
        { x: allDates.concat([...allDates].reverse()), y: [lastPrice, ...mc.p90].concat([lastPrice, ...mc.p10].reverse()),
          fill: "toself", fillcolor: "rgba(59,130,246,0.08)", line: { color: "transparent" }, name: "P10-P90" },
        { x: allDates.concat([...allDates].reverse()), y: [lastPrice, ...mc.p75].concat([lastPrice, ...mc.p25].reverse()),
          fill: "toself", fillcolor: "rgba(59,130,246,0.15)", line: { color: "transparent" }, name: "P25-P75" },
        { x: allDates, y: [lastPrice, ...mc.mean], name: "Mean", type: "scatter", mode: "lines", line: { color: "#3b82f6", width: 2 } },
        { x: allDates, y: [lastPrice, ...mc.median], name: "Median", type: "scatter", mode: "lines", line: { color: "#10b981", width: 2, dash: "dash" } },
        { x: allDates, y: Array(allDates.length).fill(lastPrice), name: "Current", type: "scatter", mode: "lines", line: { color: "#64748b", width: 1, dash: "dot" } }
    ], { ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: "Date" }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Price ($)" } }, { responsive: true });
}

function renderAccuracyChart(data) {
    const items = [];
    Object.entries(data.model_results).forEach(([name, info]) => {
        if (info.available && info.mae && typeof info.mae === "number") {
            items.push({ name, mae: info.mae, wfMae: info.wf_mae || null, color: MODEL_COLORS[name] || "#3b82f6" });
        }
    });
    items.sort((a, b) => a.mae - b.mae);

    const traces = [{
        x: items.map(i => i.name), y: items.map(i => i.mae), name: "Test MAE", type: "bar",
        marker: { color: items.map(i => i.color) },
        text: items.map(i => "$" + i.mae.toFixed(2)), textposition: "outside", textfont: { color: "#e2e8f0" }
    }];

    if (items.some(i => i.wfMae)) {
        traces.push({
            x: items.map(i => i.name), y: items.map(i => i.wfMae || 0), name: "Walk-Forward MAE", type: "bar",
            marker: { color: items.map(i => hexToRgba(i.color, 0.5)) },
            text: items.map(i => i.wfMae ? "$" + i.wfMae.toFixed(2) : ""), textposition: "outside", textfont: { color: "#94a3b8" }
        });
    }

    Plotly.newPlot("accuracyChart", traces, { ...PLOTLY_LAYOUT, barmode: "group", yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "MAE ($)" } }, { responsive: true });

    const container = document.getElementById("modelCards");
    container.innerHTML = items.map(i => `
        <div class="model-card">
            <h3 style="color: ${i.color}">${i.name}</h3>
            <div class="mae">$${i.mae.toFixed(2)}</div>
            ${i.wfMae ? `<div class="wf-mae">Walk-Forward: $${i.wfMae.toFixed(2)}</div>` : ""}
        </div>
    `).join("");
}

function renderEnsembleChart(data) {
    const ensemble = data.model_results.Ensemble;
    if (!ensemble || !ensemble.weights) return;
    const names = Object.keys(ensemble.weights);
    Plotly.newPlot("ensembleChart", [{
        labels: names, values: Object.values(ensemble.weights), type: "pie",
        marker: { colors: names.map(n => MODEL_COLORS[n] || "#3b82f6") },
        textinfo: "label+percent", textfont: { color: "#e2e8f0" }, hole: 0.4
    }], { ...PLOTLY_LAYOUT, showlegend: true }, { responsive: true });
}

function renderSummaryTable(data) {
    const tbody = document.getElementById("summaryBody");
    tbody.innerHTML = "";

    // Add reliability context row at the top if available
    if (data.reliability_warning) {
        const lastDecay = data.confidence_decay ? data.confidence_decay[data.confidence_decay.length - 1] : null;
        const decayStr = lastDecay !== null ? (lastDecay * 100).toFixed(0) + "%" : "N/A";
        tbody.innerHTML += `
            <tr style="background: rgba(59, 130, 246, 0.05);">
                <td colspan="5" style="font-style: italic; color: var(--text-secondary);">${data.reliability_warning}</td>
                <td colspan="2" style="font-style: italic; color: var(--text-secondary);">End-of-range confidence: ${decayStr}</td>
            </tr>
        `;
    }

    Object.entries(data.predictions).forEach(([name, preds]) => {
        const avg = (arr, n) => arr.slice(0, Math.min(n, arr.length)).reduce((a, b) => a + b, 0) / Math.min(n, arr.length);
        const direction = preds[preds.length - 1] > data.current_price ? "UP" : "DOWN";
        const mae = data.model_results[name]?.mae;
        const ci = data.confidence_intervals?.[name];
        const ciStr = ci ? `$${ci.lower[ci.lower.length - 1].toFixed(2)} - $${ci.upper[ci.upper.length - 1].toFixed(2)}` : "N/A";

        tbody.innerHTML += `
            <tr>
                <td style="color: ${MODEL_COLORS[name] || '#3b82f6'}; font-weight: 600">${name}</td>
                <td>$${avg(preds, 7).toFixed(2)}</td>
                <td>$${avg(preds, 14).toFixed(2)}</td>
                <td>$${avg(preds, 30).toFixed(2)}</td>
                <td class="${direction === "UP" ? "direction-up" : "direction-down"}">${direction === "UP" ? "&#9650;" : "&#9660;"} ${direction}</td>
                <td>${typeof mae === "number" ? "$" + mae.toFixed(2) : (mae || "N/A")}</td>
                <td>${ciStr}</td>
            </tr>
        `;
    });
}

// ==================== BACKTEST ====================

async function runBacktest() {
    const ticker = document.getElementById("ticker").value.trim();
    if (!ticker) return;
    setLoading("loading", true);
    hideAllPredictionSections();
    try {
        const res = await fetch("/api/backtest", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker })
        });
        const data = await res.json();
        if (data.error) { showError("error", data.error); return; }
        renderBacktest(data);
    } catch (e) { showError("error", e.message); }
    finally { setLoading("loading", false); }
}

function renderBacktest(data) {
    document.getElementById("backtestSection").classList.remove("hidden");
    const traces = [{
        x: data.dates, y: data.actual_prices, name: "Actual",
        type: "scatter", mode: "lines", line: { color: "#64748b", width: 2.5 }
    }];
    const sorted = Object.entries(data.results).sort((a, b) => a[1].mae - b[1].mae);
    sorted.forEach(([name, r]) => {
        traces.push({
            x: data.dates.slice(0, r.predictions.length), y: r.predictions,
            name, type: "scatter", mode: "lines",
            line: { color: MODEL_COLORS[name] || "#3b82f6", width: 1.5 }
        });
    });
    Plotly.newPlot("backtestChart", traces, {
        ...PLOTLY_LAYOUT,
        title: { text: `Backtest: ${data.test_period}`, font: { color: "#e2e8f0" } },
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: "Date" },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Price ($)" }
    }, { responsive: true });

    document.getElementById("backtestMetrics").innerHTML = sorted.map(([name, r]) => `
        <div class="metric-card">
            <h3 style="color: ${MODEL_COLORS[name] || '#3b82f6'}">${name}</h3>
            <div class="metric-row"><span class="metric-label">MAE</span><span class="metric-value">$${r.mae.toFixed(2)}</span></div>
            <div class="metric-row"><span class="metric-label">RMSE</span><span class="metric-value">$${r.rmse.toFixed(2)}</span></div>
            <div class="metric-row"><span class="metric-label">MAPE</span><span class="metric-value">${r.mape.toFixed(2)}%</span></div>
            <div class="metric-row"><span class="metric-label">R²</span><span class="metric-value">${r.r2.toFixed(4)}</span></div>
            <div class="metric-row"><span class="metric-label">Direction Acc.</span><span class="metric-value">${r.direction_accuracy.toFixed(1)}%</span></div>
        </div>
    `).join("");
}

// ==================== TECHNICAL ====================

async function runTechnical() {
    const ticker = document.getElementById("ticker").value.trim();
    if (!ticker) return;
    setLoading("loading", true);
    hideAllPredictionSections();
    try {
        const res = await fetch("/api/technical", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker })
        });
        const data = await res.json();
        if (data.error) { showError("error", data.error); return; }
        renderTechnical(data);
    } catch (e) { showError("error", e.message); }
    finally { setLoading("loading", false); }
}

function renderTechnical(data) {
    document.getElementById("technicalSection").classList.remove("hidden");
    document.getElementById("stockInfo").classList.remove("hidden");
    document.getElementById("infoTicker").textContent = data.ticker;
    document.getElementById("infoPrice").textContent = "$" + data.current_price.toFixed(2);

    const signalEl = document.getElementById("overallSignal");
    let cls = data.overall_signal.includes("BUY") ? "signal-buy" : data.overall_signal.includes("SELL") ? "signal-sell" : "signal-hold";
    signalEl.className = "overall-signal " + cls;
    signalEl.innerHTML = `<div>${data.overall_signal}</div>
        <div style="font-size:0.9rem;margin-top:6px;opacity:0.8">${data.ticker} @ $${data.current_price} |
        Buy: ${data.summary.buy_signals} | Sell: ${data.summary.sell_signals} | Neutral: ${data.summary.neutral_signals}</div>`;

    document.getElementById("signalsList").innerHTML = data.signals.map(s => {
        const badge = s.signal.includes("BUY") ? "buy" : s.signal.includes("SELL") ? "sell" : "neutral";
        return `<div class="signal-item"><div><div class="signal-name">${s.indicator}</div><div class="signal-value">${s.value}</div></div>
                <span class="signal-badge ${badge}">${s.signal}</span></div>`;
    }).join("");

    // Pivot Points
    if (data.pivot_points) {
        const pp = data.pivot_points;
        document.getElementById("pivotPointsSection").innerHTML = `
            <h3 class="ta-sub-heading">Pivot Points (Standard)</h3>
            <div class="pivot-table">
                <div class="pivot-row pivot-header">
                    <span>S3</span><span>S2</span><span>S1</span><span class="pivot-center">Pivot</span><span>R1</span><span>R2</span><span>R3</span>
                </div>
                <div class="pivot-row">
                    <span class="pivot-support">$${pp.s3}</span>
                    <span class="pivot-support">$${pp.s2}</span>
                    <span class="pivot-support">$${pp.s1}</span>
                    <span class="pivot-center-val">$${pp.pivot}</span>
                    <span class="pivot-resistance">$${pp.r1}</span>
                    <span class="pivot-resistance">$${pp.r2}</span>
                    <span class="pivot-resistance">$${pp.r3}</span>
                </div>
            </div>`;
    }

    // Support / Resistance
    if (data.support_resistance) {
        const sr = data.support_resistance;
        const supHtml = (sr.support || []).length > 0
            ? sr.support.map(l => `<span class="sr-level sr-support">$${l}</span>`).join("")
            : '<span class="sr-empty">No levels detected</span>';
        const resHtml = (sr.resistance || []).length > 0
            ? sr.resistance.map(l => `<span class="sr-level sr-resistance">$${l}</span>`).join("")
            : '<span class="sr-empty">No levels detected</span>';
        document.getElementById("supportResistanceSection").innerHTML = `
            <h3 class="ta-sub-heading">Support &amp; Resistance (60-Day)</h3>
            <div class="sr-grid">
                <div class="sr-column">
                    <h4 class="sr-label sr-label-support">Support Levels</h4>
                    <div class="sr-levels">${supHtml}</div>
                </div>
                <div class="sr-column">
                    <h4 class="sr-label sr-label-resistance">Resistance Levels</h4>
                    <div class="sr-levels">${resHtml}</div>
                </div>
            </div>`;
    }

    // Fibonacci Retracement
    if (data.fibonacci_levels) {
        const fb = data.fibonacci_levels;
        const curPrice = data.current_price;
        const fibRows = [
            { label: "0% (52w High)", value: fb.high_52w },
            { label: "23.6%", value: fb.level_236 },
            { label: "38.2%", value: fb.level_382 },
            { label: "50.0%", value: fb.level_500 },
            { label: "61.8%", value: fb.level_618 },
            { label: "78.6%", value: fb.level_786 },
            { label: "100% (52w Low)", value: fb.low_52w },
        ];
        document.getElementById("fibonacciSection").innerHTML = `
            <h3 class="ta-sub-heading">Fibonacci Retracement (52-Week)</h3>
            <div class="fib-table">
                ${fibRows.map(r => {
                    const isNearest = Math.abs(r.value - curPrice) / curPrice < 0.02;
                    return `<div class="fib-row${isNearest ? ' fib-nearest' : ''}">
                        <span class="fib-label">${r.label}</span>
                        <span class="fib-value">$${r.value.toFixed(2)}</span>
                    </div>`;
                }).join("")}
            </div>`;
    }
}

// ==================== RISK ANALYSIS ====================

async function runRiskAnalysis() {
    const ticker = document.getElementById("riskTicker").value.trim();
    if (!ticker) return;
    setLoading("riskLoading", true);
    document.getElementById("riskResults").classList.add("hidden");

    try {
        const res = await fetch("/api/risk", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker })
        });
        const data = await res.json();
        if (data.error) { showError("riskError", data.error); return; }
        renderRisk(data);
    } catch (e) { showError("riskError", e.message); }
    finally { setLoading("riskLoading", false); }
}

function renderRisk(data) {
    document.getElementById("riskResults").classList.remove("hidden");

    const statColor = (val, good) => good ? (val >= 0 ? "#10b981" : "#ef4444") : (val <= 0 ? "#10b981" : "#ef4444");

    document.getElementById("riskSummary").innerHTML = [
        { label: "Sharpe Ratio", value: data.sharpe_ratio, color: statColor(data.sharpe_ratio, true) },
        { label: "Annual Return", value: data.annual_return + "%", color: statColor(data.annual_return, true) },
        { label: "Annual Volatility", value: data.annual_volatility + "%", color: "#f59e0b" },
        { label: "Max Drawdown", value: data.max_drawdown + "%", color: "#ef4444" },
        { label: "Beta", value: data.beta, color: "#3b82f6" },
        { label: "Win Rate", value: data.win_rate + "%", color: data.win_rate >= 50 ? "#10b981" : "#ef4444" }
    ].map(s => `<div class="risk-stat"><div class="stat-value" style="color:${s.color}">${s.value}</div><div class="stat-label">${s.label}</div></div>`).join("");

    document.getElementById("riskMetrics").innerHTML = `
        <div class="metric-card">
            <h3>Value at Risk (Daily)</h3>
            <div class="metric-row"><span class="metric-label">VaR 95% (Historical)</span><span class="metric-value" style="color:#ef4444">${data.var_95_historical}%</span></div>
            <div class="metric-row"><span class="metric-label">VaR 99% (Historical)</span><span class="metric-value" style="color:#ef4444">${data.var_99_historical}%</span></div>
            <div class="metric-row"><span class="metric-label">VaR 95% (Parametric)</span><span class="metric-value" style="color:#ef4444">${data.var_95_parametric}%</span></div>
            <div class="metric-row"><span class="metric-label">CVaR 95%</span><span class="metric-value" style="color:#ef4444">${data.cvar_95}%</span></div>
        </div>
        <div class="metric-card">
            <h3>Risk Ratios</h3>
            <div class="metric-row"><span class="metric-label">Sharpe Ratio</span><span class="metric-value">${data.sharpe_ratio}</span></div>
            <div class="metric-row"><span class="metric-label">Sortino Ratio</span><span class="metric-value">${data.sortino_ratio}</span></div>
            <div class="metric-row"><span class="metric-label">Calmar Ratio</span><span class="metric-value">${data.calmar_ratio}</span></div>
            <div class="metric-row"><span class="metric-label">Alpha</span><span class="metric-value">${data.alpha}%</span></div>
        </div>
        <div class="metric-card">
            <h3>Distribution</h3>
            <div class="metric-row"><span class="metric-label">Skewness</span><span class="metric-value">${data.skewness}</span></div>
            <div class="metric-row"><span class="metric-label">Kurtosis</span><span class="metric-value">${data.kurtosis}</span></div>
            <div class="metric-row"><span class="metric-label">Best Day</span><span class="metric-value" style="color:#10b981">${data.best_day}%</span></div>
            <div class="metric-row"><span class="metric-label">Worst Day</span><span class="metric-value" style="color:#ef4444">${data.worst_day}%</span></div>
        </div>
    `;

    // Drawdown chart
    if (data.drawdown_series && data.drawdown_dates) {
        Plotly.newPlot("drawdownChart", [{
            x: data.drawdown_dates,
            y: data.drawdown_series.map(d => d * 100),
            type: "scatter", mode: "lines", fill: "tozeroy",
            line: { color: "#ef4444", width: 1 },
            fillcolor: "rgba(239, 68, 68, 0.2)",
            name: "Drawdown"
        }], {
            ...PLOTLY_LAYOUT,
            title: { text: "Drawdown (Last Year)", font: { color: "#e2e8f0", size: 14 } },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Drawdown (%)" }
        }, { responsive: true });
    }
}

// ==================== COMPARE ====================

async function runCompare() {
    const tickers = document.getElementById("compareTickers").value.split(",").map(t => t.trim()).filter(Boolean);
    if (tickers.length < 2) { showError("compareError", "Enter at least 2 tickers."); return; }

    setLoading("compareLoading", true);
    document.getElementById("compareResults").classList.add("hidden");

    try {
        const res = await fetch("/api/compare", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tickers })
        });
        const data = await res.json();
        if (data.error) { showError("compareError", data.error); return; }
        renderCompare(data);
    } catch (e) { showError("compareError", e.message); }
    finally { setLoading("compareLoading", false); }
}

function renderCompare(data) {
    document.getElementById("compareResults").classList.remove("hidden");
    const traces = [];
    const colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"];

    Object.entries(data).forEach(([ticker, info], i) => {
        if (info.error || !info.prices) return;
        // Normalize to percentage change
        const basePrice = info.prices[0];
        const normalized = info.prices.map(p => ((p / basePrice) - 1) * 100);
        traces.push({
            x: info.dates, y: normalized, name: ticker,
            type: "scatter", mode: "lines",
            line: { color: colors[i % colors.length], width: 2 }
        });
    });

    Plotly.newPlot("compareChart", traces, {
        ...PLOTLY_LAYOUT,
        title: { text: "Normalized Price Comparison (%)", font: { color: "#e2e8f0", size: 14 } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Change (%)" }
    }, { responsive: true });

    // Comparison table
    let tableHTML = `<table><thead><tr><th>Ticker</th><th>Price</th><th>1Y Change</th><th>Sharpe</th><th>Volatility</th><th>Max DD</th><th>Win Rate</th></tr></thead><tbody>`;
    Object.entries(data).forEach(([ticker, info]) => {
        if (info.error) {
            tableHTML += `<tr><td>${ticker}</td><td colspan="6">Data unavailable</td></tr>`;
            return;
        }
        const chgColor = info.change_1y >= 0 ? "#10b981" : "#ef4444";
        tableHTML += `<tr>
            <td style="font-weight:700;color:#3b82f6">${ticker}</td>
            <td>$${info.current_price.toFixed(2)}</td>
            <td style="color:${chgColor}">${info.change_1y >= 0 ? "+" : ""}${info.change_1y}%</td>
            <td>${info.sharpe.toFixed(2)}</td>
            <td>${info.volatility.toFixed(1)}%</td>
            <td style="color:#ef4444">${info.max_drawdown.toFixed(1)}%</td>
            <td>${info.win_rate.toFixed(1)}%</td>
        </tr>`;
    });
    tableHTML += "</tbody></table>";
    document.getElementById("compareTable").innerHTML = tableHTML;
}

async function runCorrelation() {
    const tickers = document.getElementById("compareTickers").value.split(",").map(t => t.trim()).filter(Boolean);
    if (tickers.length < 2) { showError("compareError", "Enter at least 2 tickers."); return; }

    setLoading("compareLoading", true);
    document.getElementById("correlationSection").classList.add("hidden");

    try {
        const res = await fetch("/api/correlation", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tickers })
        });
        const data = await res.json();
        if (data.error) { showError("compareError", data.error); return; }
        renderCorrelation(data);
    } catch (e) { showError("compareError", e.message); }
    finally { setLoading("compareLoading", false); }
}

function renderCorrelation(data) {
    document.getElementById("correlationSection").classList.remove("hidden");
    Plotly.newPlot("correlationChart", [{
        z: data.matrix, x: data.tickers, y: data.tickers,
        type: "heatmap", colorscale: [[0, "#ef4444"], [0.5, "#1f2937"], [1, "#10b981"]],
        zmin: -1, zmax: 1, text: data.matrix.map(row => row.map(v => v.toFixed(3))),
        texttemplate: "%{text}", textfont: { color: "#e2e8f0" }
    }], {
        ...PLOTLY_LAYOUT,
        title: { text: "Return Correlation Matrix", font: { color: "#e2e8f0", size: 14 } }
    }, { responsive: true });
}

// ==================== WATCHLIST ====================

async function addToWatchlist() {
    const ticker = document.getElementById("watchTicker").value.trim().toUpperCase();
    if (!ticker) return;
    await fetch("/api/watchlist", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker })
    });
    document.getElementById("watchTicker").value = "";
    refreshWatchlist();
}

async function removeFromWatchlist(ticker) {
    await fetch("/api/watchlist", {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker })
    });
    refreshWatchlist();
}

async function refreshWatchlist() {
    setLoading("watchlistLoading", true);
    try {
        const res = await fetch("/api/watchlist");
        const data = await res.json();
        const container = document.getElementById("watchlistContent");
        if (!data.watchlist || data.watchlist.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 20px;">No stocks in watchlist. Add a ticker above.</p>';
            return;
        }
        container.innerHTML = data.watchlist.map(s => {
            const chgColor = s.change >= 0 ? "#10b981" : "#ef4444";
            const arrow = s.change >= 0 ? "&#9650;" : "&#9660;";
            return `<div class="watchlist-card" onclick="document.getElementById('ticker').value='${s.ticker}';switchTab('predict');runPrediction()">
                <div>
                    <div class="wl-ticker">${s.ticker}</div>
                    <div class="wl-price">$${s.price.toFixed(2)}</div>
                </div>
                <div class="wl-actions">
                    <span class="wl-change" style="color:${chgColor}">${arrow} ${s.change >= 0 ? "+" : ""}${s.change.toFixed(2)}%</span>
                    <button class="danger" onclick="event.stopPropagation();removeFromWatchlist('${s.ticker}')">Remove</button>
                </div>
            </div>`;
        }).join("");
    } catch (e) { console.error(e); }
    finally { setLoading("watchlistLoading", false); }
}

// ==================== PRICE ALERTS ====================

let alerts = JSON.parse(localStorage.getItem("stockAlerts") || "[]");

function openAlertModal() {
    document.getElementById("alertModal").classList.remove("hidden");
    const ticker = document.getElementById("ticker").value.trim();
    if (ticker) document.getElementById("alertTicker").value = ticker;
}

function closeAlertModal() {
    document.getElementById("alertModal").classList.add("hidden");
}

function saveAlert() {
    const ticker = document.getElementById("alertTicker").value.trim().toUpperCase();
    const price = parseFloat(document.getElementById("alertPrice").value);
    const direction = document.getElementById("alertDirection").value;
    if (!ticker || isNaN(price)) return;

    alerts.push({ ticker, price, direction, active: true, created: new Date().toISOString() });
    localStorage.setItem("stockAlerts", JSON.stringify(alerts));
    renderActiveAlerts();
    updateAlertBadge();
    document.getElementById("alertPrice").value = "";
}

function removeAlert(index) {
    alerts.splice(index, 1);
    localStorage.setItem("stockAlerts", JSON.stringify(alerts));
    renderActiveAlerts();
    updateAlertBadge();
}

function renderActiveAlerts() {
    const container = document.getElementById("activeAlertsContent");
    const activeAlerts = alerts.filter(a => a.active);
    if (activeAlerts.length === 0) {
        container.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem;">No active alerts.</p>';
        return;
    }
    container.innerHTML = alerts.map((a, i) => {
        if (!a.active) return "";
        const dirLabel = a.direction === "above" ? "Above" : "Below";
        const dirColor = a.direction === "above" ? "#10b981" : "#ef4444";
        return `<div class="active-alert-item">
            <div class="alert-info">
                <span class="alert-ticker">${a.ticker}</span>
                <span class="alert-detail" style="color:${dirColor}">${dirLabel} $${a.price.toFixed(2)}</span>
            </div>
            <button class="alert-remove" onclick="removeAlert(${i})">Remove</button>
        </div>`;
    }).join("");
}

function updateAlertBadge() {
    const badge = document.getElementById("alertBadge");
    const count = alerts.filter(a => a.active).length;
    if (count > 0) {
        badge.textContent = count;
        badge.classList.remove("hidden");
    } else {
        badge.classList.add("hidden");
    }
}

async function checkAlerts() {
    const activeAlerts = alerts.filter(a => a.active);
    if (activeAlerts.length === 0) return;

    try {
        const res = await fetch("/api/check_alerts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ alerts: activeAlerts })
        });
        const data = await res.json();
        if (data.error) { console.error("Alert check error:", data.error); return; }

        if (data.triggered && data.triggered.length > 0) {
            const banner = document.getElementById("alertNotifications");
            data.triggered.forEach(t => {
                // Find the actual index in the full alerts array
                let matchCount = 0;
                let realIndex = -1;
                for (let i = 0; i < alerts.length; i++) {
                    if (alerts[i].active) {
                        if (matchCount === t.index) { realIndex = i; break; }
                        matchCount++;
                    }
                }

                const cssClass = t.direction === "above" ? "alert-above" : "alert-below";
                const arrow = t.direction === "above" ? "\u25B2" : "\u25BC";
                const dirText = t.direction === "above" ? "rose above" : "dropped below";

                const notification = document.createElement("div");
                notification.className = `alert-notification ${cssClass}`;
                notification.innerHTML = `
                    <span>${arrow} <strong>${t.ticker}</strong> ${dirText} $${t.target_price.toFixed(2)} &mdash; now at <strong>$${t.current_price.toFixed(2)}</strong></span>
                    <button class="alert-dismiss" onclick="this.parentElement.remove()">&times;</button>
                `;
                banner.appendChild(notification);

                // Deactivate the triggered alert
                if (realIndex >= 0) {
                    alerts[realIndex].active = false;
                }
            });

            localStorage.setItem("stockAlerts", JSON.stringify(alerts));
            renderActiveAlerts();
            updateAlertBadge();
        }
    } catch (e) {
        console.error("Failed to check alerts:", e);
    }
}

// ==================== EXPORT CSV ====================

function exportCSV() {
    if (!currentData || !currentData.predictions) {
        alert("No prediction data to export. Run a prediction first.");
        return;
    }

    const d = currentData;
    let csv = "Date," + Object.keys(d.predictions).join(",") + "\n";

    const maxLen = Math.max(...Object.values(d.predictions).map(p => p.length));
    for (let i = 0; i < maxLen; i++) {
        const date = d.prediction_dates[i] || "";
        const values = Object.values(d.predictions).map(p => p[i] !== undefined ? p[i].toFixed(2) : "");
        csv += date + "," + values.join(",") + "\n";
    }

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${d.ticker}_predictions_${d.prediction_dates[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// ==================== KEYBOARD SHORTCUTS ====================

document.getElementById("ticker").addEventListener("keyup", e => { if (e.key === "Enter") runPrediction(); });
document.getElementById("riskTicker")?.addEventListener("keyup", e => { if (e.key === "Enter") runRiskAnalysis(); });
document.getElementById("watchTicker")?.addEventListener("keyup", e => { if (e.key === "Enter") addToWatchlist(); });

document.addEventListener("keydown", e => {
    // Don't trigger in input fields
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.key === "1") switchTab("predict");
    else if (e.key === "2") switchTab("risk");
    else if (e.key === "3") switchTab("compare");
    else if (e.key === "4") switchTab("watchlist");
    else if (e.key === "?") document.getElementById("shortcutsModal").classList.remove("hidden");
    else if (e.key === "Escape") {
        document.getElementById("alertModal").classList.add("hidden");
        document.getElementById("shortcutsModal").classList.add("hidden");
    }
});

// ==================== INIT ALERTS ====================

renderActiveAlerts();
updateAlertBadge();
checkAlerts();
setInterval(checkAlerts, 60000);
