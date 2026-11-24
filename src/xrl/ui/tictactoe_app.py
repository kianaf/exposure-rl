"""Small local browser UI for playing Tic Tac Toe."""

from __future__ import annotations

import argparse
from pathlib import Path

from xrl.ui.server import make_handler as _make_handler
from xrl.ui.server import (
    resolve_runs_dir,  # noqa: F401 - re-exported for xrl.ui.tictactoe_app callers
    serve_ui,
)
from xrl.ui.session import TicTacToeSession

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xrl Tic Tac Toe</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, system-ui, sans-serif;
      --bg: #f5f6f8;
      --surface: #ffffff;
      --border: #d8dde6;
      --muted: #5e6878;
      --ink: #15191f;
      --accent: #3267d6;
      --accent-soft: #e7eefc;
      --play-grid-size: min(46vw, calc(100vh - 190px), 680px);
      --play-gap: clamp(7px, 0.8vw, 12px);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink); }
    main { width: min(98vw, 1500px); margin: 14px auto 24px; display: grid; gap: 12px; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; line-height: 1.1; }
    h2 { margin: 0; font-size: 16px; letter-spacing: 0; line-height: 1.2; }
    .topbar { display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, 360px); gap: 14px; align-items: end; }
    .status { margin-top: 7px; min-height: 22px; font-weight: 650; }
    .agent-chip { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px; min-width: 0; }
    .chip-label, .label { color: var(--muted); font-size: 12px; line-height: 1.2; }
    .chip-value, .value { margin-top: 2px; font-weight: 650; overflow-wrap: anywhere; }
    .game-stage { display: grid; grid-template-columns: minmax(0, var(--play-grid-size)) minmax(0, var(--play-grid-size)); gap: clamp(12px, 1.6vw, 22px); align-items: start; justify-content: center; }
    .surface, .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: clamp(10px, 1vw, 14px); }
    .surface-heading, .panel-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .board, .q-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--play-gap); width: 100%; max-width: var(--play-grid-size); margin: 0 auto; }
    button.cell { aspect-ratio: 1; border: 1px solid #c8ced8; border-radius: 8px; background: white; font-size: clamp(46px, 7vw, 92px); font-weight: 700; cursor: pointer; }
    button.cell:disabled { cursor: default; color: var(--ink); background: #edf0f4; }
    .bar { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; margin-top: 10px; }
    select, button.control { border: 1px solid #c8ced8; border-radius: 6px; background: white; padding: 8px 10px; font: inherit; max-width: 100%; }
    button.control { cursor: pointer; }
    button.control:disabled { cursor: default; opacity: 0.55; }
    button.control.primary:not(:disabled) { border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 700; }
    select { width: 100%; }
    .details { display: grid; grid-template-columns: minmax(320px, 1.25fr) minmax(230px, 0.9fr) minmax(230px, 0.9fr); gap: 12px; align-items: start; }
    .replay-panel { grid-column: 1 / -1; }
    pre { margin: 0; min-height: 42px; max-height: 120px; padding: 10px 12px; border-radius: 8px; background: #20242b; color: #eef2f7; overflow: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 7px 12px; }
    .picker { display: grid; gap: 8px; }
    .picker-row { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; align-items: center; }
    .picker-row select { flex: 1; min-width: 0; }
    .metric { min-width: 0; }
    .runs { display: grid; gap: 7px; max-height: 190px; overflow: auto; }
    .run { width: 100%; text-align: left; border: 1px solid #d8dde6; border-radius: 8px; background: #fff; padding: 10px 12px; cursor: pointer; font: inherit; }
    .run[aria-selected="true"] { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .run:disabled { cursor: default; opacity: 0.58; }
    .run-title { font-weight: 700; }
    .run-subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .q-context { color: var(--muted); font-size: clamp(12px, 1vw, 14px); font-variant-numeric: tabular-nums; }
    .q-cell { aspect-ratio: 1; min-height: 0; border: 1px solid transparent; border-radius: 8px; background: #eef2f7; display: grid; gap: clamp(5px, 0.6vw, 9px); align-content: center; justify-items: center; padding: clamp(7px, 0.9vw, 12px); font-variant-numeric: tabular-nums; font-size: clamp(13px, 1.2vw, 18px); overflow: hidden; }
    .q-cell.illegal { opacity: 0.45; }
    .q-cell.pending { border-color: var(--accent); background: var(--accent-soft); box-shadow: inset 0 0 0 2px var(--accent); }
    .q-cell.greedy:not(.pending) { border-color: var(--accent); border-style: dashed; }
    .q-greedy-tag { font-size: 10px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent); line-height: 1; }
    .q-mean { font-weight: 700; }
    .histogram { width: 100%; height: clamp(24px, 3.2vw, 42px); display: flex; align-items: end; gap: 1px; }
    .bin { flex: 1; min-width: 1px; background: var(--accent); opacity: 0.78; border-radius: 2px 2px 0 0; }
    .sverl-panel { display: grid; gap: 10px; }
    .target-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
    .target-tabs .control[aria-selected="true"] { border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 700; }
    .sverl-content { display: grid; grid-template-columns: minmax(180px, 330px) minmax(220px, 1fr); gap: 14px; align-items: start; }
    .sverl-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; width: 100%; max-width: 330px; }
    .sverl-feature { aspect-ratio: 1; border: 1px solid #c8ced8; border-radius: 8px; display: grid; align-content: center; justify-items: center; gap: 4px; padding: 8px; font-variant-numeric: tabular-nums; }
    .sverl-mark { font-size: clamp(22px, 3vw, 42px); line-height: 1; font-weight: 750; }
    .sverl-value { font-size: 13px; font-weight: 700; }
    .sverl-summary { display: grid; gap: 10px; }
    .sverl-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 7px 12px; }
    .sverl-interpretation { color: #243043; font-size: 14px; line-height: 1.42; }
    .sverl-equation { border: 1px solid #d9dee8; border-radius: 8px; padding: 9px 10px; background: #f8fafc; font-variant-numeric: tabular-nums; font-size: 13px; line-height: 1.4; }
    .sverl-equation strong { display: block; margin-bottom: 3px; color: #243043; }
    .sverl-legend { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; color: var(--muted); font-size: 13px; line-height: 1.35; }
    .legend-swatch { width: 18px; height: 12px; border-radius: 3px; border: 1px solid #c8ced8; display: inline-block; }
    .legend-positive { background: rgba(19, 121, 91, 0.55); }
    .legend-negative { background: rgba(180, 35, 24, 0.55); }
    .legend-neutral { background: #eef2f7; }
    .sverl-bars { display: grid; gap: 5px; }
    .sverl-bars-title { font-size: 13px; font-weight: 750; color: #243043; }
    .sverl-bar-row { display: grid; grid-template-columns: minmax(74px, auto) minmax(0, 1fr) minmax(52px, auto); gap: 8px; align-items: center; font-size: 13px; }
    .sverl-bar-label { color: #243043; white-space: nowrap; }
    .sverl-bar-track { position: relative; height: 14px; background: #eef2f7; border-radius: 4px; }
    .sverl-bar-track::before { content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #c8ced8; }
    .sverl-bar-fill { position: absolute; top: 2px; bottom: 2px; max-width: 50%; border-radius: 3px; }
    .sverl-bar-fill.positive { left: 50%; background: rgba(19, 121, 91, 0.62); }
    .sverl-bar-fill.negative { right: 50%; background: rgba(180, 35, 24, 0.62); }
    .sverl-bar-value { font-variant-numeric: tabular-nums; font-weight: 750; text-align: right; white-space: nowrap; }
    .sverl-assumption { color: var(--muted); font-size: 13px; line-height: 1.35; }
    .empty { color: var(--muted); }
    @media (max-width: 940px) {
      :root { --play-grid-size: min(94vw, calc(50vh - 72px), 560px); }
      .topbar, .game-stage, .details { grid-template-columns: 1fr; }
      .sverl-content { grid-template-columns: 1fr; }
      .replay-panel { grid-column: auto; }
    }
    @media (max-width: 520px) {
      :root { --play-grid-size: min(94vw, 430px); --play-gap: 7px; }
      main { width: min(94vw, 430px); margin-top: 12px; gap: 10px; }
      .surface, .panel { padding: 10px; }
      button.cell { font-size: clamp(40px, 16vw, 70px); }
      .bar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div>
        <h1>xrl Tic Tac Toe</h1>
        <div class="status" id="status"></div>
      </div>
      <div class="agent-chip" id="selected-agent"></div>
    </header>
    <section class="game-stage">
      <section class="surface">
        <div class="board" id="board"></div>
        <div class="bar">
          <button class="control" id="step">Step Agent</button>
          <button class="control" id="reset">Reset</button>
        </div>
      </section>
      <section class="surface">
        <div class="surface-heading">
          <h2>Current Q-values</h2>
          <div class="q-context" id="q-context"></div>
        </div>
        <div class="q-grid" id="qvalues"></div>
      </section>
    </section>
    <section class="panel sverl-panel">
      <div class="panel-heading">
        <h2>SVERL Explanation</h2>
        <div class="target-tabs" id="sverl-targets"></div>
      </div>
      <div id="sverl"></div>
    </section>
    <section class="details">
      <section class="panel">
        <div class="panel-heading">
          <h2>Select Agent</h2>
          <button class="control" id="refresh-runs" type="button">Refresh</button>
        </div>
        <div class="picker">
          <div class="picker-row">
            <select id="run"></select>
          </div>
          <div class="runs" id="runs"></div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-heading"><h2>Selected Agent Results</h2></div>
        <div class="grid" id="results"></div>
      </section>
      <section class="panel">
        <div class="panel-heading"><h2>Training Configuration</h2></div>
        <div class="grid" id="metadata"></div>
      </section>
      <section class="panel replay-panel">
        <div class="panel-heading"><h2>Replay</h2></div>
        <pre id="replay"></pre>
      </section>
    </section>
  </main>
  <script>
    const boardEl = document.getElementById("board");
    const statusEl = document.getElementById("status");
    const replayEl = document.getElementById("replay");
    const runEl = document.getElementById("run");
    const runsEl = document.getElementById("runs");
    const resultsEl = document.getElementById("results");
    const metadataEl = document.getElementById("metadata");
    const qvaluesEl = document.getElementById("qvalues");
    const qContextEl = document.getElementById("q-context");
    const sverlEl = document.getElementById("sverl");
    const sverlTargetsEl = document.getElementById("sverl-targets");
    const selectedAgentEl = document.getElementById("selected-agent");
    const stepEl = document.getElementById("step");
    const refreshRunsEl = document.getElementById("refresh-runs");
    let selectedSverlTarget = "behavior";
    let renderVersion = 0;

    async function request(path, body) {
      const response = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: body ? {"content-type": "application/json"} : {},
        body: body ? JSON.stringify(body) : undefined
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function mark(value) {
      if (value === 1) return "X";
      if (value === -1) return "O";
      return "";
    }

    function render(data) {
      const version = ++renderVersion;
      boardEl.innerHTML = "";
      data.board.forEach((value, index) => {
        const button = document.createElement("button");
        button.className = "cell";
        button.textContent = mark(value);
        button.disabled = !data.selected_run || data.agent_pending || data.terminal || value !== 0 || data.current_player !== 1;
        button.addEventListener("click", async () => render(await request("/move", {action: index})));
        boardEl.appendChild(button);
      });
      statusEl.textContent = data.agent_pending ? `${data.status} - agent paused` : data.status;
      replayEl.textContent = data.history.map(row => `${row.agent}: ${row.action}`).join("\\n");
      stepEl.disabled = !data.agent_pending;
      stepEl.classList.toggle("primary", data.agent_pending);
      renderSelectedAgent(data.selected_run);
      renderRunPicker(data);
      renderResults(data.selected_run);
      renderMetadata(data.selected_run);
      renderQValues(data.legal_q_values, data.q_value_distributions, data.pending_agent_action, data.q_value_support, data.q_action);
      renderSVERL(data.sverl, data.board);
      if (data.agent_pending) {
        request("/sverl")
          .then(sverl => {
            if (version === renderVersion) renderSVERL(sverl, data.board);
          })
          .catch(error => {
            if (version === renderVersion) renderSVERL({available: false, error: error.message}, data.board);
          });
      }
    }

    function renderSelectedAgent(run) {
      if (!run) {
        selectedAgentEl.innerHTML = `<div class="chip-label">O agent</div><div class="chip-value">No trained run selected</div>`;
        return;
      }
      selectedAgentEl.innerHTML = `<div class="chip-label">O agent</div><div class="chip-value">${escapeHtml(run.label)} (${escapeHtml(run.model_type)})</div>`;
    }

    function renderRunPicker(data) {
      runEl.innerHTML = "";
      runsEl.innerHTML = "";
      if (!data.runs.length) {
        const option = document.createElement("option");
        option.textContent = "No trained runs found";
        option.disabled = true;
        option.selected = true;
        runEl.appendChild(option);
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No trained runs found.";
        runsEl.appendChild(empty);
        return;
      }
      data.runs.forEach(run => {
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = `${run.id} - ${run.label}`;
        option.disabled = !run.loadable;
        option.selected = data.selected_run && run.id === data.selected_run.id;
        runEl.appendChild(option);

        const button = document.createElement("button");
        button.className = "run";
        button.disabled = !run.loadable;
        button.setAttribute("aria-selected", data.selected_run && run.id === data.selected_run.id ? "true" : "false");
        button.innerHTML = `<div class="run-title">${escapeHtml(run.label)}</div><div class="run-subtitle">${escapeHtml(run.path)} - ${escapeHtml(run.status)}</div>`;
        button.addEventListener("click", async () => render(await request("/run", {run_id: run.id})));
        runsEl.appendChild(button);
      });
    }

    function renderResults(run) {
      resultsEl.innerHTML = "";
      if (!run) {
        resultsEl.innerHTML = `<div class="empty">Select a trained run below to see training and evaluation results.</div>`;
        return;
      }
      const summary = run.summary || {};
      const policy = run.latest_policy || {};
      const items = [
        ["Train games", summary.matches || ""],
        ["Train record", summary.matches ? `${summary.wins}W ${summary.losses}L ${summary.draws}D` : ""],
        ["Train win/draw", trainRateSummary(summary)],
        ["Mean return", summary.mean_learner_return === undefined ? "" : format(summary.mean_learner_return)],
        ...evalItems(run.latest_evaluations || []),
        ["Minimax action fraction", minimaxFraction(policy)],
        ["Mean minimax regret", policy.mean_minimax_regret === undefined ? "" : format(policy.mean_minimax_regret)],
      ];
      renderMetricItems(resultsEl, items);
    }

    function renderMetadata(run) {
      metadataEl.innerHTML = "";
      if (!run) {
        metadataEl.innerHTML = `<div class="empty">No agent selected.</div>`;
        return;
      }
      const items = [
        ["Type", run.model_type],
        ["Path", run.path],
        ["Opponent", run.training.opponent],
        ["Episodes", run.training.episodes],
        ["Seed", run.training.seed],
        ["Encoder", run.training.encoder],
        ["Gamma", run.training.gamma],
        ["Learning", run.training.alpha ?? run.training.learning_rate],
        ["Replay", run.training.replay_capacity ? `${run.training.replay_capacity} cap, ${run.training.batch_size} batch` : ""],
      ];
      renderMetricItems(metadataEl, items);
    }

    function renderMetricItems(container, items) {
      items.filter(([, value]) => value !== undefined && value !== "").forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "metric";
        item.innerHTML = `<div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div>`;
        container.appendChild(item);
      });
    }

    function evalItems(rows) {
      return rows.map(row => [
        `Eval vs ${row.opponent}`,
        evalSummary(row)
      ]);
    }

    function trainRateSummary(summary) {
      const parts = [];
      if (summary.win_rate !== undefined) parts.push(`${percent(summary.win_rate)} W`);
      if (summary.loss_rate !== undefined) parts.push(`${percent(summary.loss_rate)} L`);
      if (summary.draw_rate !== undefined) parts.push(`${percent(summary.draw_rate)} D`);
      return parts.join(" / ");
    }

    function evalSummary(row) {
      const parts = [];
      if (row.win_rate !== undefined) parts.push(`${percent(row.win_rate)} W`);
      if (row.loss_rate !== undefined) parts.push(`${percent(row.loss_rate)} L`);
      if (row.draw_rate !== undefined) parts.push(`${percent(row.draw_rate)} D`);
      const meanReturn = row.mean_learner_return ?? row.mean_return;
      if (meanReturn !== undefined) parts.push(`return ${format(meanReturn)}`);
      return parts.join(" / ");
    }

    function minimaxFraction(policy) {
      if (policy.minimax_matches !== undefined && policy.states !== undefined) {
        const rate = policy.minimax_action_match_rate === undefined ? Number(policy.minimax_matches) / Number(policy.states) : policy.minimax_action_match_rate;
        return `${policy.minimax_matches}/${policy.states} (${percent(rate)})`;
      }
      return policy.minimax_action_match_rate === undefined ? "" : percent(policy.minimax_action_match_rate);
    }

    function renderQValues(values, distributions, pendingAction, support, greedyAction) {
      qvaluesEl.innerHTML = "";
      qContextEl.textContent = support && support.length ? `support ${format(support[0])} to ${format(support[support.length - 1])}` : values ? "mean Q" : "";
      for (let index = 0; index < 9; index += 1) {
        const cell = document.createElement("div");
        cell.className = "q-cell";
        const value = values ? values[index] : null;
        const legal = !(value === null || value === undefined);
        if (!legal) cell.classList.add("illegal");
        if (pendingAction === index) cell.classList.add("pending");
        if (legal && greedyAction === index) cell.classList.add("greedy");
        if (support && support.length) {
          cell.title = `support ${format(support[0])} to ${format(support[support.length - 1])}`;
        }

        if (legal && greedyAction === index && pendingAction !== index) {
          const tag = document.createElement("div");
          tag.className = "q-greedy-tag";
          tag.textContent = "greedy";
          cell.appendChild(tag);
        }

        const mean = document.createElement("div");
        mean.className = "q-mean";
        mean.textContent = legal ? Number(value).toFixed(3) : "-";
        cell.appendChild(mean);

        const distribution = distributions ? distributions[index] : null;
        if (distribution && distribution.length) {
          cell.appendChild(renderHistogram(distribution, support));
        }
        qvaluesEl.appendChild(cell);
      }
    }

    function renderHistogram(distribution, support) {
      const histogram = document.createElement("div");
      histogram.className = "histogram";
      const useSupport = support && support.length === distribution.length;
      const maxProbability = Math.max(...distribution.map(value => Number(value) || 0), 1e-9);
      distribution.forEach((probability, index) => {
        const bin = document.createElement("span");
        bin.className = "bin";
        bin.style.height = `${Math.max(2, (Number(probability) / maxProbability) * 24)}px`;
        if (useSupport) bin.style.background = supportColor(Number(support[index]));
        histogram.appendChild(bin);
      });
      return histogram;
    }

    function supportColor(value) {
      if (value > 0.05) return "rgba(19, 121, 91, 0.82)";
      if (value < -0.05) return "rgba(180, 35, 24, 0.82)";
      return "rgba(120, 132, 150, 0.72)";
    }

    function renderSVERL(sverl, board) {
      sverlTargetsEl.innerHTML = "";
      sverlEl.innerHTML = "";
      if (!sverl || !sverl.available) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = sverl && (sverl.error || sverl.reason)
          ? (sverl.error || sverl.reason)
          : "Play a move to explain the agent decision.";
        sverlEl.appendChild(empty);
        return;
      }

      const targets = sverl.targets || {};
      if (!targets[selectedSverlTarget]) selectedSverlTarget = Object.keys(targets)[0] || "behavior";
      Object.entries({behavior: "Behavior", prediction: "Prediction", outcome: "Outcome"}).forEach(([key, label]) => {
        const button = document.createElement("button");
        button.className = "control";
        button.type = "button";
        button.textContent = label;
        button.setAttribute("aria-selected", key === selectedSverlTarget ? "true" : "false");
        button.disabled = !targets[key];
        button.addEventListener("click", () => {
          selectedSverlTarget = key;
          renderSVERL(sverl, board);
        });
        sverlTargetsEl.appendChild(button);
      });

      const result = targets[selectedSverlTarget];
      if (!result) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No SVERL target is available for this state.";
        sverlEl.appendChild(empty);
        return;
      }

      const content = document.createElement("div");
      content.className = "sverl-content";
      content.appendChild(renderSVERLGrid(result.attributions || {}, board || []));

      const summary = document.createElement("div");
      summary.className = "sverl-summary";
      const title = selectedSverlTarget === "behavior" ? `Action ${sverl.pending_action}` : targetLabel(selectedSverlTarget);
      const metadata = targetMetadata(selectedSverlTarget, result, sverl.pending_action);
      const items = [
        ["Target", title],
        ["Baseline", format(result.baseline_value)],
        ["Full board", format(result.full_value)],
        ["Change", signedFormat(result.delta)],
        ["Sum of cell effects", signedFormat(sumAttributions(result.attributions || {}))],
      ];
      const metrics = document.createElement("div");
      metrics.className = "sverl-metrics";
      renderMetricItems(metrics, items);
      summary.appendChild(metrics);

      const interpretation = document.createElement("div");
      interpretation.className = "sverl-interpretation";
      interpretation.textContent = metadata.description;
      summary.appendChild(interpretation);

      const equation = document.createElement("div");
      equation.className = "sverl-equation";
      equation.innerHTML = `<strong>How to read it</strong>${format(result.baseline_value)} baseline + ${signedFormat(sumAttributions(result.attributions || {}))} cell effects = ${format(result.full_value)} full board.`;
      summary.appendChild(equation);

      summary.appendChild(renderSVERLLegend(metadata));
      summary.appendChild(renderCellContributions(result.attributions || {}, board || [], metadata));

      const assumption = document.createElement("div");
      assumption.className = "sverl-assumption";
      assumption.textContent = result.exact
        ? `Exact SVERL evaluates ${result.coalition_count || 512} board-cell coalitions; ${result.assumption || ""}.`
        : `Estimated SVERL evaluates ${result.coalition_count || 512} board-cell coalitions; ${result.assumption || ""}.`;
      summary.appendChild(assumption);
      content.appendChild(summary);
      sverlEl.appendChild(content);
    }

    function targetMetadata(target, result, pendingAction) {
      const defaults = {
        behavior: {
          description: `Behavior explains why the agent is about to take action ${pendingAction}. Positive cells increase that action probability; negative cells push against it.`,
          positive_meaning: "increases the pending action probability",
          negative_meaning: "decreases the pending action probability",
          value_unit: "probability",
        },
        prediction: {
          description: "Prediction explains the agent's current state-value estimate. Positive cells increase the predicted value for the agent playing O.",
          positive_meaning: "raises the predicted value for O",
          negative_meaning: "lowers the predicted value for O",
          value_unit: "value",
        },
        outcome: {
          description: "Outcome explains the expected game result if play continues under the selected policies. Positive cells improve the expected result for O.",
          positive_meaning: "improves the expected result for O",
          negative_meaning: "worsens the expected result for O",
          value_unit: "expected return",
        },
      };
      return {...(defaults[target] || defaults.behavior), ...(result || {})};
    }

    function sumAttributions(attributions) {
      return Object.values(attributions).reduce((total, value) => total + (Number(value) || 0), 0);
    }

    function renderSVERLLegend(metadata) {
      const legend = document.createElement("div");
      legend.className = "sverl-legend";
      legend.innerHTML = `
        <span><span class="legend-swatch legend-positive"></span> green ${escapeHtml(metadata.positive_meaning)}</span>
        <span><span class="legend-swatch legend-negative"></span> red ${escapeHtml(metadata.negative_meaning)}</span>
        <span><span class="legend-swatch legend-neutral"></span> neutral is near zero</span>
      `;
      return legend;
    }

    function renderCellContributions(attributions, board, metadata) {
      const wrap = document.createElement("div");
      wrap.className = "sverl-bars";
      const title = document.createElement("div");
      title.className = "sverl-bars-title";
      title.textContent = "Cell contributions (Shapley)";
      wrap.appendChild(title);

      const ranked = Array.from({length: 9}, (_, index) => ({
        index,
        value: Number(attributions[`cell_${index}`]) || 0,
      })).sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
      const maxAbs = Math.max(...ranked.map(({value}) => Math.abs(value)), 1e-9);

      ranked.forEach(({index, value}) => {
        const row = document.createElement("div");
        row.className = "sverl-bar-row";
        const occupant = mark(board[index] || 0) || "empty";
        const direction = value >= 0 ? metadata.positive_meaning : metadata.negative_meaning;
        const width = (Math.abs(value) / maxAbs) * 50;
        const sign = value >= 0 ? "positive" : "negative";
        row.title = `Cell ${index + 1} (${occupant}): ${signedFormat(value)} ${direction}`;
        row.innerHTML = `<span class="sverl-bar-label">Cell ${index + 1} ${escapeHtml(occupant)}</span>`
          + `<span class="sverl-bar-track"><span class="sverl-bar-fill ${sign}" style="width: ${width.toFixed(1)}%"></span></span>`
          + `<span class="sverl-bar-value">${signedFormat(value)}</span>`;
        wrap.appendChild(row);
      });
      return wrap;
    }

    function renderSVERLGrid(attributions, board) {
      const grid = document.createElement("div");
      grid.className = "sverl-grid";
      const values = Array.from({length: 9}, (_, index) => Number(attributions[`cell_${index}`]) || 0);
      const maxAbs = Math.max(...values.map(value => Math.abs(value)), 1e-9);
      values.forEach((value, index) => {
        const cell = document.createElement("div");
        cell.className = "sverl-feature";
        cell.style.backgroundColor = attributionColor(value, maxAbs);
        cell.title = `cell_${index}: ${format(value)}`;
        const markEl = document.createElement("div");
        markEl.className = "sverl-mark";
        markEl.textContent = mark(board[index] || 0);
        const valueEl = document.createElement("div");
        valueEl.className = "sverl-value";
        valueEl.textContent = signedFormat(value);
        cell.appendChild(markEl);
        cell.appendChild(valueEl);
        grid.appendChild(cell);
      });
      return grid;
    }

    function attributionColor(value, maxAbs) {
      const strength = Math.min(1, Math.abs(value) / maxAbs);
      const alpha = 0.10 + strength * 0.55;
      if (value > 0) return `rgba(19, 121, 91, ${alpha})`;
      if (value < 0) return `rgba(180, 35, 24, ${alpha})`;
      return "#eef2f7";
    }

    function targetLabel(target) {
      return target.charAt(0).toUpperCase() + target.slice(1);
    }

    function format(value) {
      return Number(value).toFixed(3);
    }

    function signedFormat(value) {
      const number = Number(value);
      return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
    }

    function percent(value) {
      return value === undefined || value === "" ? "" : `${(Number(value) * 100).toFixed(1)}%`;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
    }

    runEl.addEventListener("change", async () => render(await request("/run", {run_id: runEl.value})));
    stepEl.addEventListener("click", async () => render(await request("/step", {})));
    refreshRunsEl.addEventListener("click", async () => render(await request("/refresh-runs", {})));
    document.getElementById("reset").addEventListener("click", async () => render(await request("/reset", {})));
    request("/state").then(render).catch(error => statusEl.textContent = error.message);
  </script>
</body>
</html>
"""


def make_handler(session: TicTacToeSession):
    return _make_handler(session, html=HTML)


def play(
    host: str = "127.0.0.1",
    port: int = 8765,
    runs_dir: Path = Path("runs"),
    run_id: str | None = None,
    seed: int = 0,
) -> None:
    """Run a local Tic Tac Toe browser UI."""
    serve_ui(
        html=HTML,
        session_factory=TicTacToeSession,
        host=host,
        port=port,
        runs_dir=runs_dir,
        run_id=run_id,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Tic Tac Toe browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    play(host=args.host, port=args.port, runs_dir=args.runs_dir, run_id=args.run_id, seed=args.seed)


if __name__ == "__main__":
    main()
