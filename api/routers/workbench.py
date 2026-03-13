"""Internal browser-based workbench for uploading and inspecting documents."""
import asyncio
import logging
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, THREAD_POOL, check_doc_access, get_current_user
from api.models.requests import DocumentResponse
import api.services.entity_editor as entity_svc

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workbench", tags=["workbench"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


async def _require_access(
    tenant_id: str,
    user_id: str,
    doc_id: str,
    sub_tenant: Optional[str] = None,
) -> None:
    if not await check_doc_access(user_id, tenant_id, doc_id, sub_tenant=sub_tenant):
        raise HTTPException(status_code=403, detail="Access denied to this document")


def _to_document_response(doc: dict) -> DocumentResponse:
    return DocumentResponse(
        doc_id=doc["doc_id"],
        filename=doc.get("filename", ""),
        status=doc.get("status", "unknown"),
        error=doc.get("error"),
        created_at=doc.get("created_at"),
        sub_tenant=doc.get("sub_tenant"),
        document_date=doc.get("document_date"),
        document_lang=doc.get("document_lang"),
    )


def _serialize_graph_sync(tenant_id: str, doc_id: str, config_path: str) -> dict:
    graph, _, _ = entity_svc._load_graph_sync(tenant_id, doc_id, config_path)

    nodes = []
    for node_name, node_data in graph.kg.nodes(data=True):
        role_assignments = node_data.get("role_assignments", []) or []
        nodes.append({
            "id": node_name,
            "label": node_data.get("entity_name") or node_name,
            "entity_name": node_data.get("entity_name", ""),
            "entity_type": node_data.get("entity_type", ""),
            "description": node_data.get("description", ""),
            "source_ids": sorted(node_data.get("source_ids", [])),
            "role_count": len(role_assignments),
        })

    edges = []
    for source, target, edge_data in graph.kg.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "relation_name": edge_data.get("relation_name", ""),
            "weight": edge_data.get("weight", 0),
            "description": edge_data.get("description", ""),
            "source_ids": sorted(edge_data.get("source_ids", [])),
        })

    nodes.sort(key=lambda item: item["label"].lower())
    edges.sort(key=lambda item: (item["source"], item["target"], item["relation_name"]))
    return {
        "doc_id": doc_id,
        "summary": {"node_count": len(nodes), "edge_count": len(edges)},
        "nodes": nodes,
        "edges": edges,
    }


async def _load_graph_payload(tenant_id: str, doc_id: str, config_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        THREAD_POOL,
        _serialize_graph_sync,
        tenant_id,
        doc_id,
        config_path,
    )


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def workbench_home() -> HTMLResponse:
    """Serve the internal browser workbench shell."""
    return HTMLResponse(_WORKBENCH_HTML)


@router.get(
    "/api/documents/{doc_id}/bundle",
    summary="Get a workbench document bundle",
    description="Return document metadata plus entities, role assignments, and role statistics for the workbench UI.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Document not found."},
        500: {"description": "Workbench bundle loading failed."},
    },
)
async def get_document_bundle(
    doc_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    sub_tenant: Annotated[
        Optional[str],
        Query(
            min_length=1,
            max_length=128,
            description="Optional sub-tenant scope for document access.",
        ),
    ] = None,
):
    """Return the main inspection bundle for one accessible document."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id, sub_tenant=sub_tenant)

    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    document = _to_document_response(doc)
    if document.status != "ready":
        return {"doc": document, "entities": [], "roles": [], "stats": None}

    try:
        entities, roles, stats = await asyncio.gather(
            entity_svc.list_entities(tenant_id, doc_id, CONFIG_PATH),
            entity_svc.list_roles(tenant_id, doc_id, CONFIG_PATH),
            entity_svc.role_stats(tenant_id, doc_id, CONFIG_PATH),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Graph artifacts not found")
    except Exception as exc:
        log.exception(f"get_document_bundle failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "doc": document,
        "entities": entities,
        "roles": roles,
        "stats": stats,
    }


@router.get(
    "/api/documents/{doc_id}/graph",
    summary="Get a workbench graph view",
    description="Load the persisted document graph and return browser-friendly node and edge JSON for the workbench UI.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Document or graph artifacts not found."},
        409: {"description": "Document indexing is not ready yet."},
        500: {"description": "Graph loading failed."},
    },
)
async def get_document_graph(
    doc_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    sub_tenant: Annotated[
        Optional[str],
        Query(
            min_length=1,
            max_length=128,
            description="Optional sub-tenant scope for document access.",
        ),
    ] = None,
):
    """Return browser-friendly graph JSON for one accessible document."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id, sub_tenant=sub_tenant)

    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready yet")

    try:
        return await _load_graph_payload(tenant_id, doc_id, CONFIG_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Graph artifacts not found")
    except Exception as exc:
        log.exception(f"get_document_graph failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


_WORKBENCH_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BookRAG Workbench</title>
  <script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #0f172a; color: #e2e8f0; }
    main { max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1, h2, h3 { margin: 0 0 12px; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 16px; }
    .panel { background: rgba(15, 23, 42, 0.88); border: 1px solid #334155; border-radius: 12px; padding: 16px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
    input, button, textarea { font: inherit; }
    input { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 8px; border: 1px solid #475569; background: #020617; color: inherit; }
    button { border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; background: #2563eb; color: white; }
    button.secondary { background: #334155; }
    button:disabled { opacity: 0.6; cursor: default; }
    .dropzone { border: 2px dashed #475569; border-radius: 12px; padding: 18px; text-align: center; color: #cbd5e1; }
    .dropzone.dragover { border-color: #60a5fa; background: rgba(37, 99, 235, 0.12); }
    .status { min-height: 20px; margin-top: 10px; color: #93c5fd; }
    .status.error { color: #fca5a5; }
    .status.success { color: #86efac; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid #334155; text-align: left; vertical-align: top; font-size: 13px; }
    .docs-table button { padding: 6px 10px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1e293b; }
    .mono, pre { font-family: ui-monospace, SFMono-Regular, monospace; }
    pre { white-space: pre-wrap; word-break: break-word; background: #020617; padding: 12px; border-radius: 10px; }
    #graph { height: 520px; border: 1px solid #334155; border-radius: 10px; background: #020617; }
    .section { margin-top: 16px; }
    .muted { color: #94a3b8; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>BookRAG Workbench</h1>
  <p class="muted">Internal upload and inspection UI for real-document testing.</p>

  <div class="grid">
    <section class="panel">
      <h2>Authentication</h2>
      <form id="login-form">
        <div class="section"><label for="tenant-id">Tenant ID</label><input id="tenant-id" required placeholder="tenant-a"></div>
        <div class="section"><label for="username">Username</label><input id="username" required placeholder="alice"></div>
        <div class="section"><label for="password">Password</label><input id="password" required type="password"></div>
        <div class="section row">
          <button type="submit">Login</button>
          <button type="button" class="secondary" id="logout-btn">Logout</button>
        </div>
      </form>

      <div class="section">
        <h2>Upload PDFs</h2>
        <form id="upload-form">
          <div id="dropzone" class="dropzone">Drop one or more PDFs here, or pick files below.</div>
          <div class="section"><input id="file-input" type="file" accept="application/pdf,.pdf" multiple></div>
          <div class="section"><label for="document-date">Document date (optional)</label><input id="document-date" placeholder="2025-06-15"></div>
          <div class="section"><label for="document-lang">Document language (optional)</label><input id="document-lang" placeholder="en or auto"></div>
          <div class="row">
            <button type="submit">Upload selected files</button>
            <span id="selected-files" class="muted">No files selected.</span>
          </div>
        </form>
      </div>

      <div class="section row">
        <h2 style="margin-right:auto;">Documents</h2>
        <button id="refresh-docs" type="button" class="secondary">Refresh</button>
      </div>
      <table class="docs-table">
        <thead>
          <tr><th>File</th><th>Status</th><th>Action</th></tr>
        </thead>
        <tbody id="documents-body">
          <tr><td colspan="3" class="muted">Log in to load documents.</td></tr>
        </tbody>
      </table>
      <div id="status" class="status"></div>
    </section>

    <section class="panel">
      <div class="row">
        <h2 id="detail-title" style="margin-right:auto;">Document details</h2>
        <button id="download-raw" type="button" class="secondary">Download raw PDF</button>
      </div>

      <div class="section">
        <h3>Summary</h3>
        <pre id="summary-json">Select a document to inspect.</pre>
      </div>

      <div class="section">
        <h3>Role stats</h3>
        <pre id="stats-json">{}</pre>
      </div>

      <div class="section">
        <h3>Entities</h3>
        <table>
          <thead><tr><th>Name</th><th>Type</th><th>Roles</th><th>Sources</th></tr></thead>
          <tbody id="entities-body"><tr><td colspan="4" class="muted">No data yet.</td></tr></tbody>
        </table>
      </div>

      <div class="section">
        <h3>Roles</h3>
        <table>
          <thead><tr><th>Entity</th><th>Role</th><th>Status</th><th>Tenure</th></tr></thead>
          <tbody id="roles-body"><tr><td colspan="4" class="muted">No data yet.</td></tr></tbody>
        </table>
      </div>

      <div class="section">
        <div class="row"><h3 style="margin-right:auto;">Graph</h3><span id="graph-meta" class="muted"></span></div>
        <div id="graph"></div>
      </div>

      <div class="section">
        <h3>Raw bundle JSON</h3>
        <pre id="bundle-json">{}</pre>
      </div>
    </section>
  </div>
</main>

<script>
const state = {
  token: localStorage.getItem('bookrag.workbench.accessToken') || '',
  selectedDocId: null,
  files: [],
  pollHandle: null,
  cy: null,
};

const elements = {
  loginForm: document.getElementById('login-form'),
  logoutBtn: document.getElementById('logout-btn'),
  refreshDocsBtn: document.getElementById('refresh-docs'),
  uploadForm: document.getElementById('upload-form'),
  fileInput: document.getElementById('file-input'),
  selectedFiles: document.getElementById('selected-files'),
  dropzone: document.getElementById('dropzone'),
  status: document.getElementById('status'),
  documentsBody: document.getElementById('documents-body'),
  detailTitle: document.getElementById('detail-title'),
  summaryJson: document.getElementById('summary-json'),
  statsJson: document.getElementById('stats-json'),
  bundleJson: document.getElementById('bundle-json'),
  entitiesBody: document.getElementById('entities-body'),
  rolesBody: document.getElementById('roles-body'),
  graphMeta: document.getElementById('graph-meta'),
  graph: document.getElementById('graph'),
  downloadRaw: document.getElementById('download-raw'),
};

function setStatus(message, tone = 'info') {
  elements.status.textContent = message || '';
  elements.status.className = `status ${tone === 'info' ? '' : tone}`.trim();
}

function setToken(token) {
  state.token = token || '';
  if (state.token) {
    localStorage.setItem('bookrag.workbench.accessToken', state.token);
  } else {
    localStorage.removeItem('bookrag.workbench.accessToken');
  }
}

async function authorizedFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set('Authorization', `Bearer ${state.token}`);
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    setToken('');
    stopPolling();
    setStatus('Authentication expired. Please log in again.', 'error');
  }
  return response;
}

async function apiJson(path, options = {}) {
  const response = await authorizedFetch(path, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

function setFiles(fileList) {
  state.files = Array.from(fileList || []);
  elements.selectedFiles.textContent = state.files.length
    ? state.files.map(file => file.name).join(', ')
    : 'No files selected.';
}

function clearDetails(message = 'Select a document to inspect.') {
  elements.detailTitle.textContent = 'Document details';
  elements.summaryJson.textContent = message;
  elements.statsJson.textContent = '{}';
  elements.bundleJson.textContent = '{}';
  elements.entitiesBody.innerHTML = '<tr><td colspan="4" class="muted">No data yet.</td></tr>';
  elements.rolesBody.innerHTML = '<tr><td colspan="4" class="muted">No data yet.</td></tr>';
  elements.graphMeta.textContent = '';
  if (state.cy) { state.cy.destroy(); state.cy = null; }
  elements.graph.innerHTML = '';
}

function renderDocuments(documents) {
  if (!documents.length) {
    elements.documentsBody.innerHTML = '<tr><td colspan="3" class="muted">No documents found.</td></tr>';
    return;
  }

  elements.documentsBody.innerHTML = documents.map((doc) => `
    <tr>
      <td><div>${doc.filename}</div><div class="muted mono">${doc.doc_id}</div></td>
      <td><span class="pill">${doc.status}</span></td>
      <td><button type="button" data-doc-id="${doc.doc_id}">Open</button></td>
    </tr>
  `).join('');

  elements.documentsBody.querySelectorAll('button[data-doc-id]').forEach((button) => {
    button.addEventListener('click', () => selectDocument(button.dataset.docId));
  });
}

function renderEntities(entities) {
  if (!entities.length) {
    elements.entitiesBody.innerHTML = '<tr><td colspan="4" class="muted">No entities available.</td></tr>';
    return;
  }
  elements.entitiesBody.innerHTML = entities.map((entity) => `
    <tr>
      <td>${entity.entity_name}</td>
      <td>${entity.entity_type}</td>
      <td>${entity.role_assignments?.length || 0}</td>
      <td>${(entity.source_ids || []).join(', ')}</td>
    </tr>
  `).join('');
}

function renderRoles(roles) {
  if (!roles.length) {
    elements.rolesBody.innerHTML = '<tr><td colspan="4" class="muted">No role assignments available.</td></tr>';
    return;
  }
  elements.rolesBody.innerHTML = roles.map((role) => `
    <tr>
      <td>${role.entity_name} <span class="muted">(${role.entity_type})</span></td>
      <td>${role.role_name || role.observed_role_text || '—'}</td>
      <td>${role.review_status || 'unknown'}</td>
      <td>${role.tenure_status || 'unknown'}</td>
    </tr>
  `).join('');
}

function renderGraph(graphPayload) {
  if (state.cy) {
    state.cy.destroy();
    state.cy = null;
  }
  if (!graphPayload) {
    elements.graphMeta.textContent = 'Graph unavailable.';
    elements.graph.innerHTML = '';
    return;
  }
  elements.graphMeta.textContent = `${graphPayload.summary.node_count} nodes • ${graphPayload.summary.edge_count} edges`;
  if (!window.cytoscape) {
    elements.graph.textContent = JSON.stringify(graphPayload, null, 2);
    return;
  }
  state.cy = window.cytoscape({
    container: elements.graph,
    elements: [
      ...graphPayload.nodes.map((node) => ({ data: { id: node.id, label: node.label, type: node.entity_type } })),
      ...graphPayload.edges.map((edge, index) => ({ data: { id: `${edge.source}-${edge.target}-${index}`, source: edge.source, target: edge.target, label: edge.relation_name || '' } })),
    ],
    style: [
      { selector: 'node', style: { 'background-color': '#2563eb', label: 'data(label)', color: '#e2e8f0', 'font-size': 10, 'text-wrap': 'wrap', 'text-max-width': 90 } },
      { selector: 'edge', style: { width: 2, 'line-color': '#64748b', 'target-arrow-color': '#64748b', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', label: 'data(label)', 'font-size': 8, color: '#cbd5e1' } },
    ],
    layout: { name: 'cose', animate: false, fit: true, padding: 24 },
  });
}

async function refreshDocuments() {
  if (!state.token) {
    renderDocuments([]);
    return;
  }
  const documents = await apiJson('/documents');
  renderDocuments(documents);
}

async function selectDocument(docId) {
  state.selectedDocId = docId;
  elements.detailTitle.textContent = `Document ${docId}`;
  setStatus(`Loading ${docId}...`);

  const [bundleResult, graphResult] = await Promise.allSettled([
    apiJson(`/workbench/api/documents/${docId}/bundle`),
    apiJson(`/workbench/api/documents/${docId}/graph`),
  ]);

  if (bundleResult.status !== 'fulfilled') {
    clearDetails(bundleResult.reason.message);
    setStatus(bundleResult.reason.message, 'error');
    return;
  }

  const bundle = bundleResult.value;
  elements.summaryJson.textContent = JSON.stringify(bundle.doc, null, 2);
  elements.statsJson.textContent = JSON.stringify(bundle.stats || { detail: 'Document is not ready yet.' }, null, 2);
  elements.bundleJson.textContent = JSON.stringify(bundle, null, 2);
  renderEntities(bundle.entities || []);
  renderRoles(bundle.roles || []);

  if (graphResult.status === 'fulfilled') {
    renderGraph(graphResult.value);
    setStatus(`Loaded ${docId}.`, 'success');
  } else {
    renderGraph(null);
    elements.graphMeta.textContent = graphResult.reason.message;
    setStatus(`Loaded ${docId} without graph: ${graphResult.reason.message}`, 'info');
  }
}

function startPolling() {
  stopPolling();
  if (!state.token) return;
  state.pollHandle = window.setInterval(() => {
    refreshDocuments().catch((error) => setStatus(error.message, 'error'));
  }, 5000);
}

function stopPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

elements.loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = {
    tenant_id: document.getElementById('tenant-id').value.trim(),
    username: document.getElementById('username').value.trim(),
    password: document.getElementById('password').value,
  };
  try {
    const response = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.detail || 'Login failed');
    setToken(data.access_token);
    startPolling();
    await refreshDocuments();
    setStatus('Logged in.', 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
});

elements.logoutBtn.addEventListener('click', () => {
  setToken('');
  stopPolling();
  state.selectedDocId = null;
  setFiles([]);
  clearDetails('Select a document to inspect.');
  elements.documentsBody.innerHTML = '<tr><td colspan="3" class="muted">Log in to load documents.</td></tr>';
  setStatus('Logged out.');
});

elements.refreshDocsBtn.addEventListener('click', () => {
  refreshDocuments().then(() => setStatus('Document list refreshed.', 'success')).catch((error) => setStatus(error.message, 'error'));
});

elements.fileInput.addEventListener('change', (event) => setFiles(event.target.files));

['dragenter', 'dragover'].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove('dragover');
  });
});
elements.dropzone.addEventListener('drop', (event) => setFiles(event.dataTransfer.files));

elements.uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!state.token) {
    setStatus('Log in before uploading.', 'error');
    return;
  }
  if (!state.files.length) {
    setStatus('Choose one or more PDFs first.', 'error');
    return;
  }

  const formData = new FormData();
  state.files.forEach((file) => formData.append('files', file));
  const documentDate = document.getElementById('document-date').value.trim();
  const documentLang = document.getElementById('document-lang').value.trim();
  if (documentDate) formData.append('document_date', documentDate);
  if (documentLang) formData.append('document_lang', documentLang);

  try {
    const result = await apiJson('/documents', { method: 'POST', body: formData });
    setStatus(`Upload accepted: ${result.uploaded.length} file(s), ${result.failed.length} failed.`, 'success');
    setFiles([]);
    elements.fileInput.value = '';
    await refreshDocuments();
  } catch (error) {
    setStatus(error.message, 'error');
  }
});

elements.downloadRaw.addEventListener('click', async () => {
  if (!state.selectedDocId) {
    setStatus('Select a document first.', 'error');
    return;
  }
  try {
    const response = await authorizedFetch(`/documents/${state.selectedDocId}/raw`);
    if (!response.ok) {
      const text = await response.text();
      let detail = 'Raw document download failed';
      if (text) {
        try { detail = JSON.parse(text).detail || detail; } catch { detail = text; }
      }
      throw new Error(detail);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${state.selectedDocId}.pdf`;
    link.click();
    URL.revokeObjectURL(url);
    setStatus('Raw PDF downloaded.', 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
});

clearDetails();
if (state.token) {
  startPolling();
  refreshDocuments().catch((error) => setStatus(error.message, 'error'));
}
</script>
</body>
</html>
'''