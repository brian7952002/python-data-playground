/* Python Data Playground — front end.
 *
 * Plain ES modules-free JavaScript on purpose. Adding a bundler, a framework
 * and a node_modules tree to a repo about learning Python would mean you now
 * have two ecosystems to debug instead of one. Everything here is readable
 * top to bottom, and the browser runs it as written.
 */

const state = {
  labs: [],
  results: {},      // labId -> run result
  activeLab: null,
  datasets: [],
  activeDataset: null,
  savedSource: "",  // last known on-disk source, for dirty-checking and Revert
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ fetch */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    // Surface the server's own explanation rather than a generic message --
    // FastAPI puts a useful string in `detail`, including on 422s.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    } catch { /* body was not JSON; keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ------------------------------------------------------------------- view */

function showView(name) {
  for (const id of ["welcome", "lab", "data", "scratch"]) {
    $(`view-${id}`).hidden = id !== name;
  }
  document.querySelectorAll(".tool-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name)
  );
  if (name !== "lab") {
    document.querySelectorAll(".lab-item").forEach((b) => b.classList.remove("active"));
    state.activeLab = null;
  }
}

/* ------------------------------------------------------------------- labs */

function renderLabList() {
  const list = $("lab-list");
  list.innerHTML = "";

  for (const lab of state.labs) {
    const result = state.results[lab.id];
    const li = document.createElement("li");
    const button = document.createElement("button");

    button.className = "lab-item";
    if (state.activeLab === lab.id) button.classList.add("active");
    if (result) button.classList.add(result.ok ? "done" : "failing");

    button.innerHTML = `
      <span class="lab-num">${result ? (result.ok ? "✓" : "!") : lab.number}</span>
      <span class="lab-meta">
        <span class="lab-name">${escapeHtml(lab.title)}</span>
        <span class="lab-track">${escapeHtml(lab.track)}</span>
      </span>`;
    button.addEventListener("click", () => openLab(lab.id));

    li.appendChild(button);
    list.appendChild(li);
  }
  renderProgress();
}

function renderProgress() {
  const run = state.labs.filter((l) => state.results[l.id]);
  if (run.length === 0) { $("overall-progress").hidden = true; return; }

  const done = run.filter((l) => state.results[l.id].ok).length;
  $("overall-progress").hidden = false;
  $("progress-fill").style.width = `${(done / state.labs.length) * 100}%`;
  $("progress-label").textContent = `${done}/${state.labs.length} complete`;
}

async function openLab(labId) {
  const lab = state.labs.find((l) => l.id === labId);
  if (!lab) return;

  state.activeLab = labId;
  showView("lab");
  renderLabList();

  $("lab-track").textContent = lab.track;
  $("lab-title").textContent = `${lab.number}. ${lab.title}`;
  $("lab-concept").textContent = lab.concept;
  $("lab-demo-fn").textContent = `${lab.worked_example}()`;
  $("lab-exercise-fn").textContent = `${lab.exercise}()`;

  $("pane-source").value = "Loading…";
  $("pane-tests").textContent = "";

  const cached = state.results[labId];
  if (cached) renderResults(cached); else $("lab-results").hidden = true;

  try {
    const payload = await api(`/api/labs/${labId}/source`);
    $("pane-source").value = payload.source;
    $("pane-tests").textContent = payload.tests;
    $("editor-path").textContent = payload.module_path;
    state.savedSource = payload.source;
    setSaveState("", "");
  } catch (error) {
    $("pane-source").value = `Could not load source: ${error.message}`;
  }
}

/* ----------------------------------------------------------------- editing */

function setSaveState(text, kind) {
  const element = $("save-state");
  element.textContent = text;
  element.className = `save-state ${kind}`;
}

function markDirty() {
  const dirty = $("pane-source").value !== state.savedSource;
  setSaveState(dirty ? "unsaved changes" : "", dirty ? "dirty" : "");
}

async function saveAndRun() {
  const labId = state.activeLab;
  if (!labId) return;

  const source = $("pane-source").value;
  const button = $("save-source");
  const hint = $("editor-hint");

  button.disabled = true;
  setSaveState("saving…", "");

  try {
    const result = await api(`/api/labs/${labId}/source`, {
      method: "PUT",
      body: JSON.stringify({ source }),
    });
    state.savedSource = source;

    if (result.syntax_error) {
      // The file was still written -- half-finished code is a normal state.
      // But running the tests now would only produce a collection error, so
      // point at the line instead and stop here.
      const { message, line, text } = result.syntax_error;
      $("pane-source").classList.add("has-error");
      hint.classList.add("syntax-error");
      hint.innerHTML = `Saved, but Python cannot parse it — line ${line}: ${escapeHtml(message)}`
        + (text ? `<br><code>${escapeHtml(text)}</code>` : "");
      setSaveState("syntax error", "bad");
      return;
    }

    $("pane-source").classList.remove("has-error");
    hint.classList.remove("syntax-error");
    hint.innerHTML = `Editing <code>${escapeHtml(result.path)}</code> — `
      + `Ctrl/Cmd + S to save and run. Tab inserts four spaces. `
      + `Your safety net is git: <code>git checkout labs/</code> undoes everything.`;
    setSaveState("saved", "ok");

    await runLab(labId, null);
  } catch (error) {
    setSaveState(`save failed: ${error.message}`, "bad");
  } finally {
    button.disabled = false;
  }
}

function handleEditorKeys(event) {
  const editor = $("pane-source");

  // Ctrl/Cmd + S saves, instead of offering to download the page.
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveAndRun();
    return;
  }

  // Tab inserts four spaces rather than moving focus to the next control.
  // In a Python editor this is not a nicety -- indentation is syntax, and a
  // textarea that cannot indent is a textarea you cannot write Python in.
  if (event.key === "Tab") {
    event.preventDefault();
    const { selectionStart, selectionEnd, value } = editor;
    editor.value = `${value.slice(0, selectionStart)}    ${value.slice(selectionEnd)}`;
    editor.selectionStart = editor.selectionEnd = selectionStart + 4;
    markDirty();
    return;
  }

  // Enter keeps the current line's indentation, so you are not re-typing four
  // spaces on every line of a function body.
  if (event.key === "Enter") {
    const { selectionStart, value } = editor;
    const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
    const indent = (value.slice(lineStart, selectionStart).match(/^[ \t]*/) || [""])[0];
    if (indent.length === 0) return;   // nothing to carry; let the browser handle it

    event.preventDefault();
    const insertion = `\n${indent}`;
    editor.value = value.slice(0, selectionStart) + insertion + value.slice(editor.selectionEnd);
    editor.selectionStart = editor.selectionEnd = selectionStart + insertion.length;
    markDirty();
  }
}

async function runLab(labId, button) {
  const previousLabel = button ? button.textContent : null;
  if (button) { button.disabled = true; button.textContent = "Running…"; }

  try {
    const result = await api(`/api/labs/${labId}/run`, { method: "POST" });
    state.results[labId] = result;
    if (state.activeLab === labId) renderResults(result);
    renderLabList();
    return result;
  } catch (error) {
    if (state.activeLab === labId) {
      $("lab-results").hidden = false;
      $("results-summary").innerHTML = `<span class="bad">Could not run tests: ${escapeHtml(error.message)}</span>`;
      $("test-list").innerHTML = "";
      $("raw-output").textContent = "";
    }
    return null;
  } finally {
    if (button) { button.disabled = false; button.textContent = previousLabel; }
  }
}

function renderResults(result) {
  $("lab-results").hidden = false;

  const headline = result.ok
    ? `<span class="ok">All ${result.total} tests pass.</span>`
    : `<span class="bad">${result.failed} failing</span> <span class="muted">· ${result.passed} passing</span>`;

  $("results-summary").innerHTML = `
    ${headline}
    <span class="muted"> — worked example ${result.demo_passed}/${result.demo_total},
    exercise ${result.exercise_passed}/${result.exercise_total}</span>`;

  const list = $("test-list");
  list.innerHTML = "";

  // Failures first: the thing you need is at the top, not buried under
  // thirty green lines you already knew about.
  const ordered = [...result.tests].sort((a, b) => {
    const rank = (t) => (t.outcome === "failed" ? 0 : t.outcome === "skipped" ? 1 : 2);
    return rank(a) - rank(b);
  });

  for (const test of ordered) {
    const li = document.createElement("li");
    li.className = `test-row ${test.outcome}`;
    const mark = test.outcome === "passed" ? "✓" : test.outcome === "failed" ? "✕" : "–";
    li.innerHTML = `
      <span class="mark">${mark}</span>
      <span class="test-name">${escapeHtml(test.name)}</span>
      <span class="test-kind">${escapeHtml(test.kind)}</span>
      ${test.message ? `<span class="test-msg">${escapeHtml(test.message)}</span>` : ""}`;
    list.appendChild(li);
  }

  $("raw-output").textContent = result.raw || "";
}

/* --------------------------------------------------------------- datasets */

async function loadDatasets() {
  state.datasets = await api("/api/datasets");
  const tabs = $("dataset-tabs");
  tabs.innerHTML = "";

  for (const dataset of state.datasets) {
    const button = document.createElement("button");
    button.textContent = `${dataset.name} · ${dataset.rows.toLocaleString()} rows`;
    button.addEventListener("click", () => openDataset(dataset.name));
    tabs.appendChild(button);
  }
  if (state.datasets.length) openDataset(state.datasets[0].name);
}

async function openDataset(name) {
  state.activeDataset = name;
  document.querySelectorAll("#dataset-tabs button").forEach((b) =>
    b.classList.toggle("active", b.textContent.startsWith(name))
  );

  const payload = await api(`/api/datasets/${name}?limit=60`);
  const missing = Object.entries(payload.missing).filter(([, n]) => n > 0);

  $("dataset-meta").innerHTML = `
    <code>${escapeHtml(name)}</code> — ${payload.total_rows.toLocaleString()} rows,
    ${payload.columns.length} columns, showing the first ${payload.rows.length}.
    ${missing.length
      ? `Missing values: ${missing.map(([c, n]) => `<code>${escapeHtml(c)}</code> (${n})`).join(", ")}.`
      : "No missing values."}`;

  const table = $("data-table");
  table.innerHTML = "";

  const head = table.createTHead().insertRow();
  for (const column of payload.columns) {
    const th = document.createElement("th");
    th.innerHTML = `${escapeHtml(column)}<span class="dtype">${escapeHtml(payload.dtypes[column])}</span>`;
    head.appendChild(th);
  }

  const body = table.createTBody();
  for (const row of payload.rows) {
    const tr = body.insertRow();
    for (const column of payload.columns) {
      const td = tr.insertCell();
      const value = row[column];
      if (value === null || value === undefined) {
        td.textContent = "NaN";
        td.className = "null";
      } else {
        td.textContent = String(value);
      }
    }
  }
}

/* -------------------------------------------------------------- scratchpad */

const SCRATCH_EXAMPLE = `from labs import load_clean_orders, load_customers

orders = load_clean_orders()

# Top five products by revenue.
top = (
    orders.groupby("product")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top.round(2))

print()
print("Revenue by channel:")
print(orders.groupby("channel")["revenue"].sum().round(2))
`;

async function runScratch() {
  const button = $("run-scratch");
  const status = $("scratch-status");

  button.disabled = true;
  status.textContent = "Running…";
  status.className = "hint";

  try {
    const result = await api("/api/scratch", {
      method: "POST",
      body: JSON.stringify({ code: $("scratch-code").value, timeout: 20 }),
    });
    $("scratch-output").textContent =
      (result.stdout || "") + (result.stderr ? `\n${result.stderr}` : "");
    status.textContent = result.ok ? "Exit 0" : `Exit ${result.exit_code}`;
    status.className = result.ok ? "hint ok" : "hint bad";
  } catch (error) {
    $("scratch-output").textContent = error.message;
    status.textContent = "Request failed";
    status.className = "hint bad";
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ utils */

function escapeHtml(value) {
  // Everything user- or file-sourced goes through this before touching
  // innerHTML. A lab file containing `<script>` should render as text, not
  // execute -- the same discipline you would apply on a public site.
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

/* -------------------------------------------------------------------- init */

async function init() {
  state.labs = await api("/api/labs");
  renderLabList();

  $("run-lab").addEventListener("click", (event) => {
    if (state.activeLab) runLab(state.activeLab, event.currentTarget);
  });

  $("run-all").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    const original = button.textContent;
    for (const lab of state.labs) {
      button.textContent = `Running ${lab.id}…`;
      await runLab(lab.id, null);   // sequential: parallel pytest runs fight
    }                               // over the same cache and CPU for no gain
    button.textContent = original;
    button.disabled = false;
  });

  document.querySelectorAll(".tool-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      showView(view);
      if (view === "data" && state.datasets.length === 0) loadDatasets();
    });
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const showingSource = tab.dataset.tab === "source";
      $("pane-source").hidden = !showingSource;
      $("editor-hint").hidden = !showingSource;
      $("editor-actions").hidden = !showingSource;
      $("pane-tests").hidden = showingSource;
    });
  });

  $("pane-source").addEventListener("input", markDirty);
  $("pane-source").addEventListener("keydown", handleEditorKeys);
  $("save-source").addEventListener("click", saveAndRun);
  $("revert-source").addEventListener("click", () => {
    $("pane-source").value = state.savedSource;
    $("pane-source").classList.remove("has-error");
    markDirty();
  });

  // Losing an hour of work to a stray Cmd+W is a bad first impression for a
  // tool that is meant to make learning less frustrating.
  window.addEventListener("beforeunload", (event) => {
    if (state.activeLab && $("pane-source").value !== state.savedSource) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  $("scratch-code").value = SCRATCH_EXAMPLE;
  $("run-scratch").addEventListener("click", runScratch);
  $("reset-scratch").addEventListener("click", () => {
    $("scratch-code").value = SCRATCH_EXAMPLE;
  });
  $("scratch-code").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runScratch();
    }
  });
}

init().catch((error) => {
  document.getElementById("content").innerHTML =
    `<h2>Could not start</h2><p class="concept">${escapeHtml(error.message)}</p>`;
});
