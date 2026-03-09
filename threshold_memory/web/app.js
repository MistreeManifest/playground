const state = {
  phases: ["symmetry", "tension", "break", "novelty", "reintegration"],
  domains: [],
};

const statusBand = document.querySelector("#status-band");
const canonGrid = document.querySelector("#canon-grid");
const episodesList = document.querySelector("#episodes-list");
const checkpointsList = document.querySelector("#checkpoints-list");
const collapseList = document.querySelector("#collapse-list");
const glyphsList = document.querySelector("#glyphs-list");
const queryResults = document.querySelector("#query-results");
const kgEntitiesList = document.querySelector("#kg-entities-list");
const kgNeighborsResult = document.querySelector("#kg-neighbors-result");
const demoLog = document.querySelector("#demo-log");
const dbPath = document.querySelector("#db-path");
const emptyTemplate = document.querySelector("#empty-template");

const queryForm = document.querySelector("#query-form");
const queryPhase = document.querySelector("#query-phase");
const queryInput = document.querySelector("#query-input");
const refreshButton = document.querySelector("#refresh-button");

const episodeForm = document.querySelector("#episode-form");
const checkpointForm = document.querySelector("#checkpoint-form");
const collapseForm = document.querySelector("#collapse-form");
const kgNeighborsForm = document.querySelector("#kg-neighbors-form");
const kgNameInput = document.querySelector("#kg-name-input");
const demoButtons = document.querySelectorAll("[data-demo]");
const phaseSelects = document.querySelectorAll(".phase-select");

function renderEmpty(target) {
  target.replaceChildren(emptyTemplate.content.cloneNode(true));
}

function phasePill(phase) {
  if (!phase) {
    return "";
  }
  return `<span class="phase-pill ${escapeHtml(phase)}">${escapeHtml(phase)}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDate(isoString) {
  if (!isoString) {
    return "now";
  }
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    return isoString;
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function populatePhaseSelect(select, allowEmpty = false) {
  const options = (allowEmpty ? ['<option value="">none</option>'] : [])
    .concat(state.phases.map((phase) => `<option value="${phase}">${phase}</option>`))
    .join("");
  select.innerHTML = options;
}

function renderStatus(status) {
  const labels = [
    ["canon_documents", "Canon"],
    ["canon_terms", "Terms"],
    ["episodes", "Episodes"],
    ["checkpoints", "Checkpoints"],
    ["collapse_events", "Collapses"],
    ["glyphs", "Glyphs"],
    ["kg_entities", "Entities"],
    ["kg_relations", "Relations"],
  ];
  statusBand.innerHTML = labels
    .map(
      ([key, label]) => `
        <article class="metric">
          <span class="metric-label">${label}</span>
          <strong>${escapeHtml(status[key] ?? 0)}</strong>
        </article>
      `
    )
    .join("");
}

function renderCanon(documents) {
  if (!documents.length) {
    renderEmpty(canonGrid);
    return;
  }
  canonGrid.innerHTML = documents
    .map(
      (doc) => `
        <article class="doc-card">
          <span class="domain-pill">${escapeHtml(doc.domain)}</span>
          <h3>${escapeHtml(doc.title)}</h3>
          <p>${escapeHtml(doc.summary)}</p>
          <div class="mini-meta">
            <span>${escapeHtml(doc.term_count)} terms</span>
            <span>${escapeHtml(formatDate(doc.created_at))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderEpisodes(episodes) {
  if (!episodes.length) {
    renderEmpty(episodesList);
    return;
  }
  episodesList.innerHTML = episodes
    .map(
      (episode) => `
        <article class="stack-item">
          <div class="meta-strip">
            ${phasePill(episode.phase)}
            <span class="tag">${escapeHtml(episode.source)}</span>
          </div>
          <h3>${escapeHtml(episode.title)}</h3>
          <p>${escapeHtml(episode.content)}</p>
          <div class="mini-meta">
            <span>joy ${escapeHtml(episode.joy ?? "n/a")}</span>
            <span>density ${escapeHtml(episode.density)}</span>
            <span>${escapeHtml(formatDate(episode.created_at))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderCheckpoints(checkpoints) {
  if (!checkpoints.length) {
    renderEmpty(checkpointsList);
    return;
  }
  checkpointsList.innerHTML = checkpoints
    .map(
      (checkpoint) => `
        <article class="stack-item">
          <div class="meta-strip">
            ${phasePill("break")}
            <span class="tag">${escapeHtml(checkpoint.task_key)}</span>
          </div>
          <h3>${escapeHtml(checkpoint.intent)}</h3>
          <p>${escapeHtml(checkpoint.resume_cue)}</p>
          <div class="mini-meta">
            <span>next ${escapeHtml(checkpoint.next_step)}</span>
            <span>${escapeHtml(formatDate(checkpoint.created_at))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderCollapses(collapses) {
  if (!collapses.length) {
    renderEmpty(collapseList);
    return;
  }
  collapseList.innerHTML = collapses
    .map(
      (collapse) => `
        <article class="stack-item">
          <div class="meta-strip">
            ${phasePill("reintegration")}
            <span class="tag">${escapeHtml(collapse.boundary_exceeded)}</span>
          </div>
          <h3>${escapeHtml(collapse.symptoms)}</h3>
          <p>${escapeHtml(collapse.recovery_protocol)}</p>
          <div class="mini-meta">
            <span>${escapeHtml(formatDate(collapse.created_at))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderGlyphs(glyphs) {
  if (!glyphs.length) {
    renderEmpty(glyphsList);
    return;
  }
  glyphsList.innerHTML = glyphs
    .map(
      (glyph) => `
        <article class="stack-item">
          <div class="meta-strip">
            ${phasePill(glyph.phase)}
            ${glyph.keywords.map((keyword) => `<span class="tag">${escapeHtml(keyword)}</span>`).join("")}
          </div>
          <h3>${escapeHtml(glyph.title)}</h3>
          <p>${escapeHtml(glyph.content)}</p>
          <div class="mini-meta">
            <span>joy ${escapeHtml(glyph.joy ?? "n/a")}</span>
            <span>${escapeHtml(formatDate(glyph.created_at))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderKg(entities) {
  if (!entities || !entities.length) {
    renderEmpty(kgEntitiesList);
    return;
  }
  kgEntitiesList.innerHTML = entities
    .map(
      (entity) => `
        <article class="stack-item kg-entity" data-name="${escapeHtml(entity.name)}">
          <div class="meta-strip">
            <span class="domain-pill">${escapeHtml(entity.kind)}</span>
            <span class="tag">${escapeHtml(entity.relation_count)} links</span>
          </div>
          <h3>${escapeHtml(entity.name)}</h3>
          ${entity.description ? `<p>${escapeHtml(entity.description)}</p>` : ""}
        </article>
      `
    )
    .join("");

  // Click an entity card to pre-fill the neighbor search
  kgEntitiesList.querySelectorAll(".kg-entity").forEach((card) => {
    card.addEventListener("click", () => {
      kgNameInput.value = card.dataset.name;
      kgNeighborsForm.dispatchEvent(new Event("submit", { cancelable: true }));
    });
  });
}

function renderKgNeighbors(data) {
  if (!data.found) {
    kgNeighborsResult.innerHTML = `<div class="empty-state"><p>Entity "${escapeHtml(data.entity)}" not found in the graph.</p></div>`;
    return;
  }
  const outbound = data.outbound || [];
  const inbound = data.inbound || [];
  if (!outbound.length && !inbound.length) {
    kgNeighborsResult.innerHTML = `<div class="empty-state"><p>"${escapeHtml(data.entity)}" has no relations yet.</p></div>`;
    return;
  }
  const rows = (items, dir) =>
    items.map((n) => `
      <div class="kg-relation-row">
        <span class="tag">${escapeHtml(dir)}</span>
        <span class="kg-relation-verb">${escapeHtml(n.relation)}</span>
        <strong>${escapeHtml(n.name)}</strong>
        <span class="muted-weight">${escapeHtml(String(n.weight))}</span>
      </div>
    `).join("");
  kgNeighborsResult.innerHTML = `
    <article class="stack-item">
      <div class="meta-strip">
        <span class="domain-pill">${escapeHtml(data.kind || "entity")}</span>
        <strong>${escapeHtml(data.entity)}</strong>
      </div>
      ${data.description ? `<p>${escapeHtml(data.description)}</p>` : ""}
      <div class="kg-relations">
        ${rows(outbound, "→")}
        ${rows(inbound, "←")}
      </div>
    </article>
  `;
}

function renderQuery(results) {
  if (!results.length) {
    renderEmpty(queryResults);
    return;
  }
  queryResults.innerHTML = results
    .map(
      (item) => `
        <article class="stack-item">
          <div class="meta-strip">
            <span class="tag">${escapeHtml(item.kind)}</span>
            ${item.domain ? `<span class="domain-pill">${escapeHtml(item.domain)}</span>` : ""}
            ${phasePill(item.phase)}
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.body)}</p>
          <div class="mini-meta">
            <span>score ${escapeHtml(item.score)}</span>
          </div>
        </article>
      `
    )
    .join("");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || "Request failed.");
  }
  return body;
}

async function loadState() {
  const payload = await fetchJson("/api/state");
  const previousQueryPhase = queryPhase.value;
  state.phases = payload.phases;
  state.domains = payload.domains;
  dbPath.textContent = payload.db_path;
  populatePhaseSelect(queryPhase, true);
  queryPhase.value = previousQueryPhase || "break";
  phaseSelects.forEach((select) => {
    populatePhaseSelect(select, false);
    if (!select.value) {
      select.value = "symmetry";
    }
  });
  renderStatus(payload.status);
  renderCanon(payload.canon_documents);
  renderEpisodes(payload.episodes);
  renderCheckpoints(payload.checkpoints);
  renderCollapses(payload.collapse_events);
  renderGlyphs(payload.glyphs);
  renderKg(payload.kg_entities || []);
}

async function runQuery(event) {
  event?.preventDefault();
  const params = new URLSearchParams();
  params.set("q", queryInput.value.trim());
  if (queryPhase.value) {
    params.set("phase", queryPhase.value);
  }
  const payload = await fetchJson(`/api/query?${params.toString()}`);
  renderQuery(payload.results);
}

function numberOrNull(value) {
  return value === "" ? null : Number(value);
}

function writeLog(value) {
  demoLog.textContent = JSON.stringify(value, null, 2);
}

async function submitForm(event, path, transform) {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  const body = transform ? transform(payload) : payload;
  const response = await fetchJson(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  writeLog(response.result);
  form.reset();
  await loadState();
  await runQuery();
}

queryForm.addEventListener("submit", runQuery);
refreshButton.addEventListener("click", async () => {
  await loadState();
  await runQuery();
});

episodeForm.addEventListener("submit", (event) =>
  submitForm(event, "/api/episodes", (payload) => ({
    ...payload,
    delta_coherence: numberOrNull(payload.delta_coherence),
    effort: numberOrNull(payload.effort),
  }))
);

checkpointForm.addEventListener("submit", (event) =>
  submitForm(event, "/api/checkpoints", (payload) => ({
    ...payload,
    risk_flags: [],
    state_snapshot: {},
  }))
);

collapseForm.addEventListener("submit", (event) =>
  submitForm(event, "/api/collapse", (payload) => ({
    ...payload,
  }))
);

kgNeighborsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = kgNameInput.value.trim();
  if (!name) return;
  const data = await fetchJson(`/api/kg/neighbors?name=${encodeURIComponent(name)}`);
  renderKgNeighbors(data);
});

demoButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const response = await fetchJson("/api/demo", {
      method: "POST",
      body: JSON.stringify({ action: button.dataset.demo }),
    });
    writeLog(response.result);
    await loadState();
    await runQuery();
  });
});

loadState()
  .then(() => runQuery())
  .catch((error) => {
    writeLog({ error: error.message });
  });
