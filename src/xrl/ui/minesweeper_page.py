"""Browser page for the local Minesweeper viewer."""

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xrl Minesweeper</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      --bg: #f2f4f5;
      --surface: #ffffff;
      --surface-soft: #f7f8f8;
      --border: #d7dcde;
      --ink: #182023;
      --muted: #667176;
      --accent: #1769aa;
      --accent-dark: #10578e;
      --accent-soft: #e8f2fa;
      --safe: #187654;
      --danger: #b42318;
      --danger-soft: #fff0ee;
      --board-size: min(66vw, calc(100vh - 250px), 680px);
      --cell-gap: clamp(4px, 0.6vw, 7px);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink); }
    button, input, select { font: inherit; }
    button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {
      outline: 3px solid #8cc5ed;
      outline-offset: 2px;
    }
    main { width: min(96vw, 1240px); margin: 0 auto; padding: 22px 0 30px; }
    h1, h2, p { margin: 0; letter-spacing: 0; }
    h1 { font-size: 24px; line-height: 1.15; }
    h2 { font-size: 15px; line-height: 1.3; }
    .app-header { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin-bottom: 16px; }
    .brand { display: flex; align-items: center; gap: 11px; }
    .brand-mark { width: 30px; height: 30px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; }
    .brand-mark span { background: var(--ink); border-radius: 2px; }
    .brand-mark span:nth-child(2), .brand-mark span:nth-child(6) { background: var(--accent); }
    .connection { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 13px; }
    .connection-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--safe); }
    .connection.offline .connection-dot { background: var(--danger); }
    .error-banner {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
      padding: 11px 13px;
      border: 1px solid #efaaa4;
      border-radius: 6px;
      background: var(--danger-soft);
      color: #7a1d16;
      font-size: 14px;
    }
    .error-banner.visible { display: flex; }
    .workspace { display: grid; grid-template-columns: minmax(0, var(--board-size)) minmax(270px, 1fr); gap: 18px; align-items: start; }
    .game, .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
    .game { padding: clamp(12px, 1.5vw, 18px); }
    .game-status { display: grid; grid-template-columns: minmax(90px, 1fr) minmax(150px, 2fr) minmax(70px, 1fr); gap: 14px; align-items: end; margin-bottom: 13px; }
    .eyebrow { color: var(--muted); font-size: 11px; line-height: 1.2; text-transform: uppercase; }
    .status-value, .metric-value { margin-top: 3px; font-weight: 750; font-variant-numeric: tabular-nums; }
    .status-value.win { color: var(--safe); }
    .status-value.loss { color: var(--danger); }
    .progress-track { height: 7px; margin-top: 7px; overflow: hidden; border-radius: 4px; background: #e4e8e9; }
    .progress-fill { width: 0; height: 100%; border-radius: inherit; background: var(--safe); transition: width 160ms ease; }
    .board { display: grid; gap: var(--cell-gap); width: 100%; }
    .board.locked { pointer-events: none; opacity: 0.78; }
    .board-cell {
      aspect-ratio: 1;
      min-width: 0;
      position: relative;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid #bcc5c9;
      border-radius: 6px;
      background: #e3e8ea;
      color: var(--ink);
      box-shadow: inset 0 -3px 0 #c9d1d4;
      cursor: pointer;
      font-weight: 800;
      font-size: clamp(17px, 2.2vw, 32px);
    }
    .board-cell.hidden:hover { border-color: var(--accent); background: #dceaf4; }
    .board-cell.revealed { border-color: #dce1e3; background: var(--surface-soft); box-shadow: none; cursor: default; }
    .board-cell.zero { color: transparent; }
    .board-cell.mine { background: var(--danger-soft); color: var(--danger); }
    .board-cell.exploded { border-color: var(--danger); background: var(--danger); color: #fff; }
    .board-cell.best { border-color: var(--accent); box-shadow: inset 0 0 0 2px var(--accent); }
    .board-cell:disabled { cursor: default; }
    .q-tag { position: absolute; right: 4px; bottom: 3px; color: var(--muted); font-size: clamp(9px, 0.8vw, 11px); font-weight: 650; font-variant-numeric: tabular-nums; }
    .clue-1 { color: #1769aa; }
    .clue-2 { color: #187654; }
    .clue-3 { color: #b42318; }
    .clue-4, .clue-5, .clue-6, .clue-7, .clue-8 { color: #70529c; }
    .controls { display: grid; grid-template-columns: minmax(130px, 1.4fr) minmax(80px, 0.8fr) minmax(110px, 1fr) minmax(100px, 0.8fr); gap: 8px; margin-top: 12px; }
    .control, select, input {
      min-width: 0;
      min-height: 40px;
      border: 1px solid #c6cdd0;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
    }
    button.control { cursor: pointer; font-weight: 680; }
    button.control:hover:not(:disabled) { border-color: #8e9ba0; background: var(--surface-soft); }
    button.control.primary { border-color: var(--accent); background: var(--accent); color: #fff; }
    button.control.primary:hover:not(:disabled) { border-color: var(--accent-dark); background: var(--accent-dark); }
    button.control:disabled, select:disabled, input:disabled { cursor: default; opacity: 0.5; }
    .seed-field { position: relative; }
    .seed-field label { position: absolute; top: 4px; left: 10px; color: var(--muted); font-size: 10px; line-height: 1; }
    .seed-field input { width: 100%; padding-top: 16px; }
    .options { display: flex; justify-content: flex-end; margin-top: 10px; }
    .check { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); font-size: 13px; cursor: pointer; }
    .check input { width: 16px; min-height: 16px; margin: 0; accent-color: var(--accent); }
    .sidebar { display: grid; gap: 12px; }
    .panel { padding: 14px; }
    .panel-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 11px; }
    .icon-button { width: 34px; min-height: 34px; padding: 0; font-size: 20px; line-height: 1; }
    .agent-picker { display: grid; grid-template-columns: minmax(0, 1fr); gap: 7px; }
    .agent-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .metric { min-width: 0; }
    .metric-value { overflow-wrap: anywhere; }
    .run-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
    .run-tags[hidden] { display: none; }
    .tag { padding: 3px 9px; border: 1px solid var(--accent); border-radius: 999px; background: var(--accent-soft); color: var(--accent-dark); font-size: 11px; font-weight: 680; line-height: 1.3; }
    .model-card { margin-top: 12px; border-top: 1px solid #edf0f1; }
    .model-card[hidden] { display: none; }
    .model-card-row { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 8px; align-items: baseline; padding-top: 7px; font-size: 13px; }
    .model-card-row .card-value { overflow-wrap: anywhere; }
    .history { max-height: 255px; overflow: auto; }
    .history-head, .history-row { display: grid; grid-template-columns: 42px minmax(65px, 1fr) 70px 68px; gap: 7px; align-items: center; }
    .history-head { padding: 0 5px 7px; color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .history-row { min-height: 34px; padding: 7px 5px; border-top: 1px solid #edf0f1; font-size: 13px; font-variant-numeric: tabular-nums; }
    .actor { font-weight: 680; text-transform: capitalize; }
    details { border-top: 1px solid #e6eaeb; margin-top: 12px; padding-top: 10px; }
    summary { color: var(--muted); cursor: pointer; font-size: 13px; }
    .q-grid { display: grid; grid-template-columns: var(--q-columns, repeat(5, 1fr)); gap: 4px; margin-top: 10px; }
    .q-cell { min-width: 0; min-height: 30px; padding: 0; display: grid; place-items: center; border: 1px solid #dde2e4; border-radius: 4px; background: var(--surface-soft); color: var(--muted); cursor: pointer; font-size: 10px; font-variant-numeric: tabular-nums; }
    .q-cell:hover:not(:disabled) { border-color: #8e9ba0; }
    .q-cell.best { border-color: var(--accent); background: var(--accent-soft); color: var(--accent-dark); font-weight: 800; }
    .q-cell.selected { box-shadow: inset 0 0 0 2px var(--ink); }
    .q-cell.illegal { cursor: default; opacity: 0.35; }
    .q-distribution { margin-top: 13px; padding-top: 11px; border-top: 1px solid #edf0f1; }
    .q-distribution[hidden] { display: none; }
    .distribution-heading { min-height: 34px; display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
    .distribution-heading strong { display: block; color: var(--ink); font-size: 12px; }
    .distribution-stats { text-align: right; }
    .distribution-chart { height: 116px; display: flex; align-items: flex-end; gap: 1px; padding-top: 8px; border-bottom: 1px solid #aeb8bc; }
    .distribution-bar { min-width: 1px; flex: 1 1 0; background: var(--safe); }
    .distribution-bar.negative { background: #c34f46; }
    .distribution-bar.zero { background: #738086; }
    .distribution-axis { display: flex; justify-content: space-between; margin-top: 4px; color: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
    .empty { padding: 12px 0; color: var(--muted); font-size: 13px; }
    .explain-controls { display: flex; gap: 8px; margin-top: 8px; }
    .explain-controls select { flex: 1 1 auto; min-width: 0; }
    .explain-status { margin-top: 8px; color: var(--muted); font-size: 12px; }
    .explain-summary { margin-top: 10px; font-size: 12px; color: var(--ink); }
    .explain-summary .assumption { margin-top: 6px; color: var(--muted); font-size: 11px; }
    .attr-cell { position: relative; display: grid; place-items: center; min-height: 34px; border: 1px solid #dde2e4; border-radius: 4px; font-size: 10px; font-variant-numeric: tabular-nums; color: var(--ink); }
    .attr-cell .glyph { position: absolute; top: 2px; left: 4px; font-size: 9px; color: var(--muted); }
    .attr-cell.best { box-shadow: inset 0 0 0 2px var(--ink); font-weight: 800; }
    .contrib-row { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; font-variant-numeric: tabular-nums; margin-top: 3px; }
    .contrib-row .positive { color: #1c7c46; }
    .contrib-row .negative { color: #c34f46; }
    .explain-history { margin-top: 10px; }
    .explain-history select { width: 100%; }
    @media (max-width: 900px) {
      :root { --board-size: min(94vw, 650px); }
      .workspace { grid-template-columns: 1fr; }
      .sidebar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .sidebar .panel:last-child { grid-column: 1 / -1; }
    }
    @media (max-width: 580px) {
      main { width: min(94vw, 480px); padding-top: 14px; }
      .app-header { align-items: flex-start; }
      .connection span:last-child { display: none; }
      .game-status { grid-template-columns: 1fr 1.5fr 0.7fr; gap: 8px; }
      .controls { grid-template-columns: 1fr 1fr; }
      .sidebar { grid-template-columns: 1fr; }
      .sidebar .panel:last-child { grid-column: auto; }
      .history-head, .history-row { grid-template-columns: 36px minmax(58px, 1fr) 60px 58px; gap: 4px; }
    }
  </style>
</head>
<body>
  <main>
    <header class="app-header">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
        </div>
        <h1>xrl Minesweeper</h1>
      </div>
      <div class="connection" id="connection"><span class="connection-dot"></span><span id="connection-text">Connecting</span></div>
    </header>
    <div class="error-banner" id="error-banner" role="alert">
      <span id="error-message"></span>
      <button class="control" id="retry" type="button">Retry</button>
    </div>
    <section class="workspace">
      <section class="game" aria-label="Minesweeper game">
        <div class="game-status">
          <div><div class="eyebrow">Game</div><div class="status-value" id="status">Loading</div></div>
          <div>
            <div class="eyebrow">Safe cells <span id="safe-count"></span></div>
            <div class="progress-track" aria-hidden="true"><div class="progress-fill" id="progress-fill"></div></div>
          </div>
          <div><div class="eyebrow">Moves</div><div class="status-value" id="move-count">0</div></div>
        </div>
        <div class="board locked" id="board"></div>
        <div class="controls">
          <button class="control primary" id="auto-agent" type="button">▶ Run agent</button>
          <button class="control" id="step-agent" type="button">Step</button>
          <button class="control" id="reset" type="button">New board</button>
          <div class="seed-field"><label for="seed">Seed</label><input id="seed" type="number" step="1" value="0"></div>
        </div>
        <div class="options"><label class="check"><input id="show-scores" type="checkbox">Show model scores</label></div>
      </section>
      <aside class="sidebar">
        <section class="panel">
          <div class="panel-heading">
            <h2>Agent</h2>
            <button class="control icon-button" id="refresh-runs" type="button" title="Refresh runs" aria-label="Refresh runs">↻</button>
          </div>
          <div class="agent-picker"><select id="run" aria-label="Saved agent"></select></div>
          <div class="run-tags" id="run-tags" hidden></div>
          <div class="agent-meta" id="agent-meta"></div>
          <div class="model-card" id="model-card" hidden></div>
          <details>
            <summary>Model scores <span id="q-context"></span></summary>
            <div class="q-grid" id="qvalues"></div>
            <div class="q-distribution" id="q-distribution" hidden>
              <div class="distribution-heading">
                <strong id="distribution-action">Return distribution</strong>
                <span class="distribution-stats" id="distribution-stats"></span>
              </div>
              <div class="distribution-chart" id="distribution-chart" role="img"></div>
              <div class="distribution-axis" id="distribution-axis"></div>
            </div>
          </details>
        </section>
        <section class="panel">
          <div class="panel-heading"><h2>Session</h2></div>
          <div class="agent-meta" id="session-metrics"></div>
        </section>
        <section class="panel">
          <div class="panel-heading"><h2>Explanations</h2></div>
          <div class="explain-controls">
            <select id="explain-method" aria-label="Explanation method"></select>
            <button class="control" id="explain-run" type="button">Explain current state</button>
          </div>
          <div class="explain-status" id="explain-status"></div>
          <div class="explain-summary" id="explain-summary" hidden></div>
          <div class="q-grid" id="explain-grid"></div>
          <details class="explain-history">
            <summary>Saved explanations</summary>
            <select id="explain-history" aria-label="Saved explanations" size="1"></select>
          </details>
        </section>
        <section class="panel">
          <div class="panel-heading"><h2>Move log</h2></div>
          <div class="history-head"><span>#</span><span>Player</span><span>Cell</span><span>Reward</span></div>
          <div class="history" id="history"></div>
        </section>
      </aside>
    </section>
  </main>
  <script>
    const boardEl = document.getElementById("board");
    const statusEl = document.getElementById("status");
    const safeCountEl = document.getElementById("safe-count");
    const progressFillEl = document.getElementById("progress-fill");
    const moveCountEl = document.getElementById("move-count");
    const runEl = document.getElementById("run");
    const runTagsEl = document.getElementById("run-tags");
    const agentMetaEl = document.getElementById("agent-meta");
    const modelCardEl = document.getElementById("model-card");
    const sessionMetricsEl = document.getElementById("session-metrics");
    const historyEl = document.getElementById("history");
    const qvaluesEl = document.getElementById("qvalues");
    const qContextEl = document.getElementById("q-context");
    const qDistributionEl = document.getElementById("q-distribution");
    const distributionActionEl = document.getElementById("distribution-action");
    const distributionStatsEl = document.getElementById("distribution-stats");
    const distributionChartEl = document.getElementById("distribution-chart");
    const distributionAxisEl = document.getElementById("distribution-axis");
    const stepAgentEl = document.getElementById("step-agent");
    const autoAgentEl = document.getElementById("auto-agent");
    const resetEl = document.getElementById("reset");
    const refreshRunsEl = document.getElementById("refresh-runs");
    const seedEl = document.getElementById("seed");
    const showScoresEl = document.getElementById("show-scores");
    const connectionEl = document.getElementById("connection");
    const connectionTextEl = document.getElementById("connection-text");
    const errorBannerEl = document.getElementById("error-banner");
    const errorMessageEl = document.getElementById("error-message");
    const explainMethodEl = document.getElementById("explain-method");
    const explainRunEl = document.getElementById("explain-run");
    const explainStatusEl = document.getElementById("explain-status");
    const explainSummaryEl = document.getElementById("explain-summary");
    const explainGridEl = document.getElementById("explain-grid");
    const explainHistoryEl = document.getElementById("explain-history");
    let current = null;
    let autoTimer = null;
    let busy = false;
    let selectedQAction = null;
    let explainBusy = false;
    let explainVersion = 0;
    let savedExplanations = [];
    let explainHistoryRunId;

    async function request(path, body, timeoutMs) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), timeoutMs || 10000);
      try {
        const response = await fetch(path, {
          method: body === undefined ? "GET" : "POST",
          headers: body === undefined ? {} : {"content-type": "application/json"},
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
        return payload;
      } catch (error) {
        if (error.name === "AbortError") throw new Error("The server did not respond.");
        if (error instanceof TypeError) throw new Error("Cannot reach the local game server.");
        throw error;
      } finally {
        window.clearTimeout(timeout);
      }
    }

    function render(data) {
      current = data;
      hideError();
      setConnected(true);
      seedEl.value = data.seed;
      statusEl.textContent = statusText(data);
      statusEl.className = `status-value ${data.outcome || ""}`;
      safeCountEl.textContent = `${data.safe_cells_revealed} / ${data.total_safe_cells}`;
      progressFillEl.style.width = `${Math.round(data.completion_ratio * 100)}%`;
      moveCountEl.textContent = data.step_count;
      renderBoard(data);
      renderRunPicker(data);
      renderMetrics(data);
      renderQValues(data);
      renderHistory(data);
      renderExplainStub(data);
      syncControls();
      if (data.terminal) stopAuto();
    }

    function statusText(data) {
      if (data.outcome === "win") return "Won";
      if (data.outcome === "loss") return "Mine hit";
      if (autoTimer) return "Agent running";
      return data.step_count === 0 ? "Ready" : "In progress";
    }

    function renderBoard(data) {
      boardEl.innerHTML = "";
      boardEl.style.gridTemplateColumns = `repeat(${data.columns}, minmax(0, 1fr))`;
      const mines = new Set(data.mine_positions || []);
      data.cells.forEach((value, index) => {
        const row = Math.floor(index / data.columns) + 1;
        const column = index % data.columns + 1;
        const isHidden = value === data.hidden_value;
        const isExploded = data.exploded_action === index;
        const isMine = mines.has(index);
        const q = data.legal_q_values ? data.legal_q_values[index] : null;
        const button = document.createElement("button");
        button.className = "board-cell";
        button.classList.add(isHidden ? "hidden" : "revealed");
        if (value === 0) button.classList.add("zero");
        if (value > 0 && value < data.exploded_value) button.classList.add(`clue-${value}`);
        if (isMine) button.classList.add("mine");
        if (isExploded) button.classList.add("exploded");
        if (showScoresEl.checked && data.q_action === index && isHidden) button.classList.add("best");
        button.textContent = cellText(value, data, index, mines);
        button.disabled = data.terminal || !isHidden;
        button.setAttribute("aria-label", cellLabel(value, row, column, isHidden, isMine, isExploded, q));
        button.addEventListener("click", () => runAction("/reveal", {action: index}));
        if (showScoresEl.checked && q !== null && q !== undefined && isHidden) {
          const tag = document.createElement("span");
          tag.className = "q-tag";
          tag.textContent = Number(q).toFixed(2);
          button.appendChild(tag);
        }
        boardEl.appendChild(button);
      });
    }

    function cellText(value, data, index, mines) {
      if (data.exploded_action === index) return "×";
      if (mines.has(index)) return "●";
      if (value === data.hidden_value || value === 0) return "";
      return String(value);
    }

    function cellLabel(value, row, column, isHidden, isMine, isExploded, q) {
      const place = `row ${row}, column ${column}`;
      if (isExploded) return `Exploded mine, ${place}`;
      if (isMine) return `Mine, ${place}`;
      if (!isHidden) return value === 0 ? `Empty cell, ${place}` : `${value} adjacent mines, ${place}`;
      const score = q === null || q === undefined ? "" : `, model score ${Number(q).toFixed(2)}`;
      return `Covered cell, ${place}${score}`;
    }

    function renderRunTags(run) {
      runTagsEl.innerHTML = "";
      const tags = (run && run.tags) || [];
      runTagsEl.hidden = !tags.length;
      tags.forEach(tag => {
        const chip = document.createElement("span");
        chip.className = "tag";
        chip.textContent = tag;
        runTagsEl.appendChild(chip);
      });
    }

    function renderModelCard(run) {
      modelCardEl.innerHTML = "";
      const rows = (run && run.model_card) || [];
      modelCardEl.hidden = !rows.length;
      rows.forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "model-card-row";
        row.innerHTML = `<div class="eyebrow">${escapeHtml(key)}</div><div class="card-value">${escapeHtml(String(value))}</div>`;
        modelCardEl.appendChild(row);
      });
    }

    function renderRunPicker(data) {
      renderRunTags(data.selected_run);
      renderModelCard(data.selected_run);
      runEl.innerHTML = "";
      if (!data.runs.length) {
        const option = document.createElement("option");
        option.textContent = "No saved agents found";
        option.disabled = true;
        option.selected = true;
        runEl.appendChild(option);
        runEl.disabled = true;
        agentMetaEl.innerHTML = '<div class="empty">No Minesweeper DQN runs found.</div>';
        return;
      }
      runEl.disabled = busy;
      data.runs.forEach(run => {
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = run.label;
        option.disabled = !run.loadable;
        option.selected = Boolean(data.selected_run && run.id === data.selected_run.id);
        runEl.appendChild(option);
      });
      const run = data.selected_run;
      if (!run) {
        agentMetaEl.innerHTML = '<div class="empty">No loadable agent selected.</div>';
        return;
      }
      const summary = run.summary || {};
      const evaluation = run.latest_evaluation || {};
      renderMetricItems(agentMetaEl, [
        ["Training games", run.config.episodes],
        ["Train win rate", optionalPercent(summary.win_rate)],
        ["Eval win rate", optionalPercent(evaluation.win_rate)],
        ["Model", run.config.value_head || "scalar"]
      ]);
    }

    function renderMetrics(data) {
      renderMetricItems(sessionMetricsEl, [
        ["Board", `${data.rows} × ${data.columns}`],
        ["Mines", data.mines],
        ["Completion", percent(data.completion_ratio)],
        ["Return", format(data.total_reward)]
      ]);
    }

    function renderQValues(data) {
      qvaluesEl.innerHTML = "";
      qvaluesEl.style.setProperty("--q-columns", `repeat(${data.columns}, minmax(0, 1fr))`);
      qContextEl.textContent = data.q_error
        ? "— error"
        : data.q_values_available
          ? `— ${data.q_policy_label || "greedy"}`
          : "— unavailable";
      const distributions = data.legal_q_distributions || [];
      if (!distributions[selectedQAction]) {
        selectedQAction = distributions[data.q_action] ? data.q_action : distributions.findIndex(Boolean);
      }
      data.cells.forEach((value, index) => {
        const cell = document.createElement("button");
        const q = data.legal_q_values ? data.legal_q_values[index] : null;
        cell.className = "q-cell";
        cell.type = "button";
        if (q === null || q === undefined) cell.classList.add("illegal");
        if (data.q_action === index) cell.classList.add("best");
        if (selectedQAction === index && distributions[index]) cell.classList.add("selected");
        cell.textContent = q === null || q === undefined ? "–" : Number(q).toFixed(2);
        cell.disabled = q === null || q === undefined;
        cell.setAttribute("aria-label", qCellLabel(data, index, q));
        cell.setAttribute("aria-pressed", String(selectedQAction === index));
        if (distributions[index]) {
          cell.addEventListener("click", () => {
            selectedQAction = index;
            renderQValues(current);
          });
        }
        qvaluesEl.appendChild(cell);
      });
      renderQDistribution(data, selectedQAction);
    }

    function qCellLabel(data, index, q) {
      const row = Math.floor(index / data.columns) + 1;
      const column = index % data.columns + 1;
      if (q === null || q === undefined) return `Cell R${row} C${column}, unavailable`;
      return `Cell R${row} C${column}, expected Q ${Number(q).toFixed(3)}`;
    }

    function renderQDistribution(data, action) {
      const support = data.q_support || [];
      const distributions = data.legal_q_distributions || [];
      const probabilities = action === null || action < 0 ? null : distributions[action];
      const available = data.q_distribution_available && support.length && probabilities;
      qDistributionEl.hidden = !available;
      distributionChartEl.innerHTML = "";
      distributionAxisEl.innerHTML = "";
      if (!available) return;

      const row = Math.floor(action / data.columns) + 1;
      const column = action % data.columns + 1;
      const expected = data.legal_q_values[action];
      const peak = Math.max(...probabilities);
      distributionActionEl.textContent = `Return distribution · R${row} C${column}`;
      distributionStatsEl.textContent = `E[Q] ${Number(expected).toFixed(3)} · peak ${percent(peak)}`;
      distributionChartEl.setAttribute(
        "aria-label",
        `Categorical Q distribution for row ${row}, column ${column}, expected value ${Number(expected).toFixed(3)}`
      );

      probabilities.forEach((probability, index) => {
        const value = Number(support[index]);
        const bar = document.createElement("span");
        bar.className = `distribution-bar ${value < 0 ? "negative" : value > 0 ? "positive" : "zero"}`;
        bar.style.height = `${peak > 0 ? Math.max(1, Number(probability) / peak * 100) : 0}%`;
        bar.title = `${value.toFixed(3)}: ${(Number(probability) * 100).toFixed(2)}%`;
        distributionChartEl.appendChild(bar);
      });

      const middle = support[Math.floor((support.length - 1) / 2)];
      [support[0], middle, support[support.length - 1]].forEach(value => {
        const label = document.createElement("span");
        label.textContent = Number(value).toFixed(2);
        distributionAxisEl.appendChild(label);
      });
    }

    function renderHistory(data) {
      historyEl.innerHTML = "";
      const rows = (data.history || []).filter(row => row.action !== null && row.action !== undefined).reverse();
      if (!rows.length) {
        historyEl.innerHTML = '<div class="empty">No moves yet.</div>';
        return;
      }
      rows.forEach(row => {
        const item = document.createElement("div");
        const boardRow = Math.floor(row.action / data.columns) + 1;
        const boardColumn = row.action % data.columns + 1;
        item.className = "history-row";
        item.innerHTML = `<span>${row.step}</span><span class="actor">${escapeHtml(row.actor)}</span><span>R${boardRow} C${boardColumn}</span><span>${format(row.reward)}</span>`;
        historyEl.appendChild(item);
      });
    }

    function attributionColor(value, maxAbs) {
      if (!maxAbs) return "transparent";
      const strength = Math.min(1, Math.abs(value) / maxAbs);
      const alpha = 0.12 + strength * 0.55;
      return value >= 0 ? `rgba(28, 124, 70, ${alpha})` : `rgba(195, 79, 70, ${alpha})`;
    }

    function renderExplainStub(data) {
      explainVersion += 1;  // the board changed; any in-flight result is stale
      const stub = data.explain || {available: false, reason: "Explanations unavailable.", methods: []};
      const previous = explainMethodEl.value;
      explainMethodEl.innerHTML = "";
      (stub.methods || []).forEach(method => {
        const option = document.createElement("option");
        option.value = method.id;
        option.textContent = method.label;
        option.disabled = !method.available;
        if (method.reason) option.title = method.reason;
        explainMethodEl.appendChild(option);
      });
      const usable = (stub.methods || []).filter(method => method.available);
      if (usable.some(method => method.id === previous)) {
        explainMethodEl.value = previous;
      } else if (usable.length) {
        explainMethodEl.value = usable[0].id;
      }
      explainStatusEl.textContent = stub.available
        ? "Explains the greedy action's Q-value for the current board."
        : (stub.reason || "Explanations unavailable.");
      syncExplainControls(stub);
      // The saved list only changes when a new result is computed or the run
      // switches; reloading it on every board move would re-read every saved
      // file from disk each step (and each auto-play tick).
      const runId = data.selected_run ? data.selected_run.id : null;
      if (runId !== explainHistoryRunId) {
        explainHistoryRunId = runId;
        loadExplanations();
      }
    }

    function syncExplainControls(stub) {
      const state = stub || (current && current.explain) || {available: false, methods: []};
      const usable = (state.methods || []).some(method => method.available);
      explainRunEl.disabled = explainBusy || !state.available || !usable;
      explainMethodEl.disabled = explainBusy || !(state.methods || []).length;
    }

    async function runExplain() {
      if (explainBusy || !explainMethodEl.value) return;
      const version = explainVersion;
      explainBusy = true;
      explainRunEl.textContent = "Explaining…";
      explainStatusEl.textContent = "Computing attribution (first run also collects a rollout background)…";
      syncExplainControls();
      try {
        const payload = await request("/explain", {method: explainMethodEl.value}, 120000);
        if (version !== explainVersion) return;  // board moved on; drop stale result
        if (!payload.available) {
          explainStatusEl.textContent = payload.reason || payload.error || "Explanation unavailable.";
          explainSummaryEl.hidden = true;
          explainGridEl.innerHTML = "";
          return;
        }
        explainStatusEl.textContent = payload.save_error
          ? `Computed, but saving failed: ${payload.save_error}`
          : "";
        renderExplanation(payload.document);
        loadExplanations();
      } catch (error) {
        if (version === explainVersion) explainStatusEl.textContent = error.message;
      } finally {
        explainBusy = false;
        explainRunEl.textContent = "Explain current state";
        syncExplainControls();
      }
    }

    function renderExplanation(doc) {
      const layout = doc.layout || {rows: current ? current.rows : 5, columns: current ? current.columns : 5};
      const row = Math.floor(doc.action / layout.columns) + 1;
      const column = doc.action % layout.columns + 1;
      const delta = doc.full_value - doc.baseline_value;
      const duration = doc.duration_seconds ? ` · ${Number(doc.duration_seconds).toFixed(2)}s` : "";
      explainSummaryEl.hidden = false;
      explainSummaryEl.innerHTML = "";
      const headline = document.createElement("div");
      headline.innerHTML = `<strong>${escapeHtml(doc.method)}</strong> · R${row} C${column} · `
        + `baseline ${format(doc.baseline_value)} → full ${format(doc.full_value)} · Δ ${format(delta)}${duration}`;
      explainSummaryEl.appendChild(headline);
      renderTopContributions(doc);
      const assumption = document.createElement("div");
      assumption.className = "assumption";
      assumption.textContent = doc.characteristic_assumption || "";
      explainSummaryEl.appendChild(assumption);
      renderExplanationGrid(doc, layout);
    }

    function renderExplanationGrid(doc, layout) {
      explainGridEl.innerHTML = "";
      explainGridEl.style.setProperty("--q-columns", `repeat(${layout.columns}, minmax(0, 1fr))`);
      const names = Object.keys(doc.attributions);
      const values = names.map(name => doc.attributions[name]);
      const maxAbs = Math.max(...values.map(Math.abs), 0);
      const observation = (doc.state && doc.state.observation) || [];
      values.forEach((value, index) => {
        const cell = document.createElement("div");
        cell.className = "attr-cell";
        if (index === doc.action) cell.classList.add("best");
        cell.style.background = attributionColor(value, maxAbs);
        const glyph = document.createElement("span");
        glyph.className = "glyph";
        glyph.textContent = cellGlyph(Math.round(observation[index]));
        cell.appendChild(glyph);
        cell.appendChild(document.createTextNode(Number(value).toFixed(3)));
        const cellRow = Math.floor(index / layout.columns) + 1;
        const cellColumn = index % layout.columns + 1;
        cell.title = `${names[index]} (R${cellRow} C${cellColumn}): ${Number(value).toFixed(4)}`;
        explainGridEl.appendChild(cell);
      });
    }

    function cellGlyph(value) {
      if (current && value === current.hidden_value) return "■";
      if (current && value === current.exploded_value) return "✸";
      return value === 0 ? "·" : String(value);
    }

    function renderTopContributions(doc) {
      const entries = Object.entries(doc.attributions).sort((left, right) => right[1] - left[1]);
      const top = entries.slice(0, 3).filter(([, value]) => value > 0);
      const bottom = entries.slice(-3).filter(([, value]) => value < 0).reverse();
      [[top, "positive"], [bottom, "negative"]].forEach(([items, kind]) => {
        items.forEach(([name, value]) => {
          const rowEl = document.createElement("div");
          rowEl.className = "contrib-row";
          rowEl.innerHTML = `<span>${escapeHtml(name)}</span><span class="${kind}">${Number(value).toFixed(4)}</span>`;
          explainSummaryEl.appendChild(rowEl);
        });
      });
    }

    async function loadExplanations() {
      try {
        const payload = await request("/explanations");
        savedExplanations = payload.explanations || [];
      } catch (error) {
        savedExplanations = [];
      }
      explainHistoryEl.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = savedExplanations.length
        ? `${savedExplanations.length} saved — select to view`
        : "No saved explanations yet.";
      explainHistoryEl.appendChild(placeholder);
      savedExplanations.forEach(doc => {
        const option = document.createElement("option");
        const created = doc.created_at ? new Date(doc.created_at).toLocaleTimeString() : "?";
        const delta = doc.full_value - doc.baseline_value;
        option.value = doc.id || "";
        option.textContent = `${created} · ${doc.method} · a${doc.action} · Δ${Number(delta).toFixed(3)}`;
        explainHistoryEl.appendChild(option);
      });
      explainHistoryEl.disabled = !savedExplanations.length;
    }

    function onExplainHistoryChange() {
      const doc = savedExplanations.find(item => item.id === explainHistoryEl.value);
      if (doc) {
        explainStatusEl.textContent = `Saved result from ${doc.state && doc.state.source ? doc.state.source : "earlier"} (board snapshot below).`;
        renderExplanation(doc);
      }
    }

    function renderMetricItems(container, items) {
      container.innerHTML = "";
      items.filter(([, value]) => value !== undefined && value !== "").forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "metric";
        item.innerHTML = `<div class="eyebrow">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(String(value))}</div>`;
        container.appendChild(item);
      });
    }

    async function runAction(path, body) {
      stopAuto();
      if (busy) return;
      busy = true;
      syncControls();
      try {
        render(await request(path, body));
      } catch (error) {
        showError(error.message);
      } finally {
        busy = false;
        syncControls();
      }
    }

    function startAuto() {
      if (autoTimer || busy || !agentCanMove()) return;
      autoAgentEl.textContent = "Ⅱ Pause";
      autoTimer = window.setTimeout(autoStep, 0);
      if (current) statusEl.textContent = "Agent running";
    }

    function stopAuto() {
      if (autoTimer) window.clearTimeout(autoTimer);
      autoTimer = null;
      autoAgentEl.textContent = "▶ Run agent";
      if (current) statusEl.textContent = statusText(current);
    }

    async function autoStep() {
      autoTimer = null;
      if (!agentCanMove()) {
        stopAuto();
        return;
      }
      busy = true;
      syncControls();
      try {
        render(await request("/agent-step", {}));
      } catch (error) {
        showError(error.message);
        stopAuto();
      } finally {
        busy = false;
        syncControls();
      }
      if (current && !current.terminal && !errorBannerEl.classList.contains("visible")) {
        autoAgentEl.textContent = "Ⅱ Pause";
        autoTimer = window.setTimeout(autoStep, 360);
      }
    }

    function agentCanMove() {
      return Boolean(current && !current.terminal && current.selected_run && current.selected_run.loadable);
    }

    function syncControls() {
      const canMove = agentCanMove();
      boardEl.classList.toggle("locked", busy || !current);
      stepAgentEl.disabled = busy || !canMove;
      autoAgentEl.disabled = busy || !canMove;
      resetEl.disabled = busy || !current;
      seedEl.disabled = busy || !current;
      refreshRunsEl.disabled = busy;
      runEl.disabled = busy || !current || !current.runs.length;
    }

    function setConnected(connected) {
      connectionEl.classList.toggle("offline", !connected);
      connectionTextEl.textContent = connected ? "Server connected" : "Server unavailable";
    }

    function showError(message) {
      stopAuto();
      setConnected(false);
      errorMessageEl.textContent = message || "Unable to reach the game server.";
      errorBannerEl.classList.add("visible");
    }

    function hideError() {
      errorBannerEl.classList.remove("visible");
    }

    function format(value) {
      return Number(value || 0).toFixed(3);
    }

    function percent(value) {
      return `${Math.round(Number(value || 0) * 100)}%`;
    }

    function optionalPercent(value) {
      return value === undefined ? "" : percent(value);
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
    }

    async function loadState() {
      if (busy) return;
      busy = true;
      syncControls();
      try {
        render(await request("/state"));
      } catch (error) {
        showError(error.message);
      } finally {
        busy = false;
        syncControls();
      }
    }

    stepAgentEl.addEventListener("click", () => runAction("/agent-step", {}));
    autoAgentEl.addEventListener("click", () => autoTimer ? stopAuto() : startAuto());
    resetEl.addEventListener("click", () => runAction("/new", {}));
    seedEl.addEventListener("change", () => runAction("/reset", {seed: seedEl.value}));
    refreshRunsEl.addEventListener("click", () => runAction("/refresh-runs", {}));
    runEl.addEventListener("change", () => runAction("/run", {run_id: runEl.value}));
    showScoresEl.addEventListener("change", () => current && renderBoard(current));
    explainRunEl.addEventListener("click", runExplain);
    explainHistoryEl.addEventListener("change", onExplainHistoryChange);
    document.getElementById("retry").addEventListener("click", loadState);
    loadState();
  </script>
</body>
</html>
"""
