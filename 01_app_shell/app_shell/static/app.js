const state = {
  status: null,
  workspaces: [],
  activeWorkspace: null,
  citationList: [],
  citationIndex: 0,
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.detail || `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function badge(text, cls = '') {
  return `<span class="badge ${cls}">${text}</span>`;
}

const SUPPORT_STATUS_LABELS = {
  slide_grounded: 'Grounded in your materials',
  not_grounded: 'Not grounded',
  partially_grounded: 'Partially grounded',
  annotation_grounded: 'Grounded in your notes',
  grounded: 'Grounded in your materials',
  ungrounded: 'Not grounded',
};

function formatSupportStatus(status) {
  return SUPPORT_STATUS_LABELS[status] || status?.replace(/_/g, ' ') || '';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showTransientMessage(text, type = 'muted') {
  const container = document.getElementById('global-toast-stack');
  if (!container) return;
  const div = document.createElement('div');
  div.className = `toast ${type}`;
  div.textContent = text;
  container.append(div);
  if (container.children.length > 5) container.removeChild(container.firstElementChild);
  setTimeout(() => {
    if (div.parentNode) div.parentNode.removeChild(div);
  }, 5000);
}

function setGlobalActivity(message, busy = true) {
  const banner = document.getElementById('global-activity');
  if (!banner) return;
  if (!message) {
    banner.textContent = '';
    banner.classList.add('hidden');
    return;
  }
  banner.textContent = busy ? `Working... ${message}` : message;
  banner.classList.remove('hidden');
}

function clearGlobalActivitySoon(delayMs = 1200) {
  setTimeout(() => setGlobalActivity(''), delayMs);
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let unitIndex = 0;
  let scaled = value;
  while (scaled >= 1024 && unitIndex < units.length - 1) {
    scaled /= 1024;
    unitIndex += 1;
  }
  return `${scaled.toFixed(scaled >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function selectedFileKey(file) {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

function uploadStatusLabel(status) {
  if (status === 'uploading') return 'Uploading';
  if (status === 'done') return 'Imported';
  if (status === 'skipped') return 'Skipped duplicate';
  if (status === 'failed') return 'Failed';
  return 'Queued';
}

function renderMaterialUploadStatus(items, active = false) {
  const container = document.getElementById('material-upload-status');
  const summary = document.getElementById('material-upload-summary');
  const count = document.getElementById('material-upload-count');
  const progress = document.getElementById('material-upload-progress');
  const list = document.getElementById('material-upload-list');
  if (!container || !summary || !count || !progress || !list) return;

  if (!items.length) {
    container.classList.add('hidden');
    summary.textContent = 'No upload in progress.';
    count.textContent = '';
    progress.value = 0;
    list.innerHTML = '';
    setGlobalActivity('');
    return;
  }

  container.classList.remove('hidden');
  const total = items.length;
  const done = items.filter((item) => item.status === 'done').length;
  const skipped = items.filter((item) => item.status === 'skipped').length;
  const failed = items.filter((item) => item.status === 'failed').length;
  const processed = done + skipped + failed;
  progress.value = Math.round((processed / total) * 100);
  count.textContent = `${processed}/${total}`;
  summary.textContent = active
    ? `Uploading files... ${done} imported, ${skipped} skipped, ${failed} failed`
    : `Upload complete: ${done} imported, ${skipped} skipped, ${failed} failed`;
  setGlobalActivity(summary.textContent, active);
  if (!active) clearGlobalActivitySoon(3000);

  list.innerHTML = items.map((item) => `
    <div class="upload-item ${item.status}">
      <div>${escapeHtml(item.name)}</div>
      <div class="muted">${escapeHtml(uploadStatusLabel(item.status))}</div>
      <div class="muted">${escapeHtml(formatBytes(item.size))}</div>
    </div>
  `).join('');
}

async function loadStatus() {
  state.status = await api('/api/status');
  renderSystemStatus();
}

function renderSystemStatus() {
  const container = document.getElementById('system-status');
  if (!state.status) {
    container.innerHTML = '<div class="muted small">Loading service status…</div>';
    return;
  }
  const content = state.status.services.content;
  const learning = state.status.services.learning;
  const mode = state.status.effective_mode;
  const allGood = content.available && learning.available;
  const openAttr = (!allGood && mode !== 'mock') ? 'open' : '';
  container.innerHTML = `
    <details ${openAttr} style="margin-bottom:8px;">
      <summary class="small muted" style="cursor:pointer;">Mode: ${escapeHtml(mode)}${!allGood && mode !== 'mock' ? ' — ⚠ services offline' : ''}</summary>
      <div class="card" style="margin-top:8px;">
        <div>${badge(`content: ${content.available ? 'available' : 'unavailable'}`, content.available ? 'ok' : 'danger')} ${badge(`learning: ${learning.available ? 'available' : 'unavailable'}`, learning.available ? 'ok' : 'danger')}</div>
      </div>
    </details>`;
  applyServiceGating();
}

function serviceAvailability() {
  const services = state.status?.services || {};
  return {
    contentAvailable: Boolean(services.content?.available),
    learningAvailable: Boolean(services.learning?.available),
  };
}

function setServiceNote(noteId, message) {
  const note = document.getElementById(noteId);
  if (!note) return;
  if (!message) {
    note.classList.add('hidden');
    note.textContent = '';
    return;
  }
  note.textContent = message;
  note.classList.remove('hidden');
}

function setDisabledForSelectors(selectors, disabled) {
  selectors.forEach((selector) => {
    document.querySelectorAll(selector).forEach((element) => {
      element.disabled = disabled;
    });
  });
}

function applyServiceGating() {
  const { contentAvailable, learningAvailable } = serviceAvailability();
  const contentMessage = contentAvailable
    ? ''
    : 'Content service is unavailable. Imports, ingestion updates, and source-grounded previews are disabled until it is healthy.';
  const learningMessage = learningAvailable
    ? ''
    : 'Learning service is unavailable. Study plans and grounded chat are disabled until it is healthy.';

  setServiceNote('materials-service-note', contentMessage);
  setServiceNote(
    'source-service-note',
    contentAvailable ? '' : 'Source viewer cannot resolve slide/page URLs while the content service is offline.'
  );
  setServiceNote('study-service-note', learningMessage);
  setServiceNote('chat-service-note', learningMessage);

  setDisabledForSelectors(
    [
      '#material-import-form input',
      '#material-import-form select',
      '#material-import-form textarea',
      '#material-import-form button',
      '.open-material-source',
      '#source-prev',
      '#source-next',
      '#source-open',
    ],
    !contentAvailable
  );
  setDisabledForSelectors(
    [
      '#study-plan-form input',
      '#study-plan-form select',
      '#study-plan-form textarea',
      '#study-plan-form button',
      '#revise-study-plan',
      '#new-conversation',
      '#clear-conversation',
      '#chat-form textarea',
      '#chat-form select',
      '#chat-form button',
    ],
    !learningAvailable
  );
}

async function loadWorkspaces() {
  const payload = await api('/api/workspaces');
  state.workspaces = payload.workspaces || [];
  renderWorkspaceList();
}

function renderWorkspaceList() {
  const container = document.getElementById('workspace-list');
  if (!state.workspaces.length) {
    container.innerHTML = '<div class="muted">No workspaces yet.</div>';
    return;
  }
  container.innerHTML = state.workspaces.map((workspace) => {
    const isActive = state.activeWorkspace?.workspace_id === workspace.workspace_id;
    const hasProcessing = workspace.material_counts.processing > 0;
    return `
    <div class="workspace-card${isActive ? ' active' : ''}">
      <div class="workspace-card-header">
        <span class="workspace-card-name">${escapeHtml(workspace.display_name)}</span>
        ${badge(workspace.grounding_mode || 'strict_lecture_only')}
      </div>
      <div class="workspace-card-meta small muted">Opened ${escapeHtml(formatDate(workspace.last_opened_at))}</div>
      <div class="workspace-card-stats small">
        <span class="ws-stat${hasProcessing ? ' processing' : ''}">
          ${workspace.material_counts.ready} material${workspace.material_counts.ready !== 1 ? 's' : ''}${hasProcessing ? ` · ${workspace.material_counts.processing} processing` : ''}
        </span>
        <span class="ws-stat-sep">·</span>
        <span class="ws-stat">${workspace.artifact_counts.study_plans} plan${workspace.artifact_counts.study_plans !== 1 ? 's' : ''}</span>
        <span class="ws-stat-sep">·</span>
        <span class="ws-stat">${workspace.artifact_counts.conversations} chat${workspace.artifact_counts.conversations !== 1 ? 's' : ''}</span>
        <span class="ws-stat-sep">·</span>
      </div>
      <div class="workspace-card-actions">
        <button type="button" class="ws-open-btn" data-open-workspace="${workspace.workspace_id}">${isActive ? 'Active' : 'Open'}</button>
        <button type="button" class="secondary ws-action-btn" data-duplicate-workspace="${workspace.workspace_id}">Duplicate</button>
        <button type="button" class="secondary ws-action-btn" data-archive-workspace="${workspace.workspace_id}">Archive</button>
        <button type="button" class="danger ws-action-btn" data-delete-workspace="${workspace.workspace_id}">Delete</button>
      </div>
    </div>
  `}).join('');

  container.querySelectorAll('[data-open-workspace]').forEach((button) => {
    button.addEventListener('click', () => openWorkspace(button.dataset.openWorkspace));
  });
  container.querySelectorAll('[data-duplicate-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      await api(`/api/workspaces/${button.dataset.duplicateWorkspace}/duplicate`, { method: 'POST' });
      await loadWorkspaces();
    });
  });
  container.querySelectorAll('[data-archive-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      await api(`/api/workspaces/${button.dataset.archiveWorkspace}/archive`, { method: 'POST' });
      await loadWorkspaces();
    });
  });
  container.querySelectorAll('[data-delete-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!window.confirm('Delete this workspace?')) return;
      await api(`/api/workspaces/${button.dataset.deleteWorkspace}`, { method: 'DELETE' });
      if (state.activeWorkspace?.workspace_id === button.dataset.deleteWorkspace) {
        state.activeWorkspace = null;
        renderWorkspace();
      }
      await loadWorkspaces();
    });
  });
}

async function openWorkspace(workspaceId) {
  const payload = await api(`/api/workspaces/${workspaceId}`);
  state.activeWorkspace = payload.workspace;
  renderWorkspace();
}

function setTab(tabName) {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `tab-${tabName}`);
  });
}

function materialPreferenceOptions(current) {
  return ['default', 'focus', 'exclude'].map((option) => `<option value="${option}" ${current === option ? 'selected' : ''}>${option}</option>`).join('');
}

function renderMaterials() {
  const list = document.getElementById('material-list');
  const workspace = state.activeWorkspace;
  const materials = workspace?.materials || [];
  if (!materials.length) {
    list.innerHTML = '<div class="muted">No materials in this workspace yet.</div>';
    return;
  }
  list.innerHTML = materials.map((material) => {
    const pref = workspace.material_preferences?.[material.material_id] || 'default';
    const statusCls = material.processing_status === 'ready' ? 'ok' : material.processing_status === 'failed' ? 'danger' : 'warn';
    return `
      <div class="card">
        <div class="row" style="justify-content:space-between;align-items:flex-start;">
          <div>
            <div><strong>${escapeHtml(material.title)}</strong></div>
            <div class="small muted">${escapeHtml(material.role)} • ${escapeHtml(material.kind)} • ${material.page_count ?? 0} pages/slides</div>
            <div>${badge(material.processing_status, statusCls)} ${badge(pref)}</div>
          </div>
          <div style="display:flex;gap:6px;">
            <button type="button" class="secondary open-material-source" data-material-id="${material.material_id}">Open source</button>
            <button type="button" class="danger delete-material" data-material-id="${material.material_id}">Delete</button>
          </div>
        </div>
        <div class="small muted" style="margin-top:8px;">${escapeHtml(material.quality_summary?.notes || '')}</div>
        <div class="row" style="margin-top:10px;">
          <label style="flex:1 1 220px; margin:0;">
            Grounding preference
            <select class="material-preference" data-material-id="${material.material_id}">${materialPreferenceOptions(pref)}</select>
          </label>
        </div>
      </div>
    `;
  }).join('');

  list.querySelectorAll('.material-preference').forEach((select) => {
    select.addEventListener('change', async () => {
      const payload = await api(`/api/workspaces/${workspace.workspace_id}/materials/${select.dataset.materialId}/preference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preference: select.value }),
      });
      state.activeWorkspace = payload.workspace;
      renderWorkspace();
      if (payload.warning) showTransientMessage(payload.warning, 'warning');
    });
  });

  list.querySelectorAll('.open-material-source').forEach((button) => {
    button.addEventListener('click', () => {
      const material = materials.find((item) => item.material_id === button.dataset.materialId);
      const slide = material?.slides?.[0];
      if (slide) {
        openCitation({
          material_id: material.material_id,
          material_title: material.title,
          slide_id: slide.slide_id,
          slide_number: slide.slide_number,
          snippet_text: slide.snippet_text,
          preview_url: slide.preview_url || '',
          source_open_url: slide.source_open_url || material.source_view_url || '',
          support_type: 'source_preview',
        }, [
          {
            material_id: material.material_id,
            material_title: material.title,
            slide_id: slide.slide_id,
            slide_number: slide.slide_number,
            snippet_text: slide.snippet_text,
            preview_url: slide.preview_url || '',
            source_open_url: slide.source_open_url || material.source_view_url || '',
            support_type: 'source_preview',
          },
        ]);
        return;
      }
      if (material?.source_view_url) {
        window.open(material.source_view_url, '_blank', 'noopener');
        return;
      }
      showTransientMessage('No source preview is available for this material yet.', 'warning');
    });
  });

  list.querySelectorAll('.delete-material').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!window.confirm('Delete this material? This cannot be undone.')) return;
      await api(`/api/workspaces/${workspace.workspace_id}/materials/${button.dataset.materialId}`, { method: 'DELETE' });
      const refreshed = await api(`/api/workspaces/${workspace.workspace_id}`);
      state.activeWorkspace = refreshed.workspace;
      renderWorkspace();
      await loadWorkspaces();
    });
  });
}

function renderCitationButtons(citations = []) {
  return citations.map((citation, index) => `
    <button type="button" class="secondary citation-button" data-citation-index="${index}" title="View source slide in the Source viewer">
      <span class="citation-icon">⬡</span> Slide ${escapeHtml(String(citation.slide_number || '?'))}
    </button>
  `).join('');
}

function wireCitationButtons(container, citations) {
  container.querySelectorAll('[data-citation-index]').forEach((button) => {
    button.addEventListener('click', () => {
      const idx = Number(button.dataset.citationIndex);
      openCitation(citations[idx], citations, idx);
    });
  });
}

function renderPlanTasks(tasks = []) {
  if (!tasks.length) return '<div class="small muted">No checklist items.</div>';
  return `
    <ul class="task-list">
      ${tasks.map((task) => `<li>${escapeHtml(task)}</li>`).join('')}
    </ul>
  `;
}

function renderTailoringSummary(summary) {
  if (!summary) return '';
  const usedInputs = summary.used_inputs || [];
  const missingInputs = summary.missing_inputs || [];
  const evidenceScope = summary.evidence_scope || {};
  const materialsText = (evidenceScope.material_titles || []).join(', ');
  const slideNumbers = (evidenceScope.slide_numbers || []).join(', ');
  return `
    <div class="card">
      <div class="small muted" style="margin-bottom:8px;">Inputs used to tailor this plan</div>
      <div class="plan-input-grid">
        ${usedInputs.map((item) => `
          <div class="plan-input-item">
            <div class="small muted">${escapeHtml(item.label || item.key || 'Input')}</div>
            <div>${escapeHtml(item.value || '')}</div>
          </div>
        `).join('')}
      </div>
      <div class="small muted plan-evidence-scope">
        Grounded in ${escapeHtml(String(evidenceScope.material_count || 0))} material${(evidenceScope.material_count || 0) === 1 ? '' : 's'}
        ${materialsText ? `: ${escapeHtml(materialsText)}` : ''}
        ${slideNumbers ? ` â€¢ cited slides ${escapeHtml(slideNumbers)}` : ''}
      </div>
      ${missingInputs.length ? `
        <div class="warning small plan-missing-inputs">
          ${missingInputs.map((item) => escapeHtml(item.message || `${item.label || item.key} was not provided.`)).join(' ')}
        </div>
      ` : ''}
    </div>
  `;
}

function optionalPositiveIntegerValue(inputId) {
  const raw = document.getElementById(inputId)?.value?.trim() || '';
  if (!raw) return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.floor(parsed);
}

function renderPlanUncertainty(items = []) {
  if (!items.length) return '';
  return `
    <div class="card">
      <div class="small muted" style="margin-bottom:8px;">Transparency notes</div>
      <div class="stack">
        ${items.map((item) => `<div class="warning small">${escapeHtml(item.message || '')}</div>`).join('')}
      </div>
    </div>
  `;
}

function renderStudyPlan() {
  const output = document.getElementById('study-plan-output');
  const workspace = state.activeWorkspace;
  document.getElementById('topic-text').value = workspace?.topic_text || '';
  document.getElementById('time-budget').value = workspace?.time_budget_minutes || '90';
  document.getElementById('grounding-mode').value = workspace?.grounding_mode || 'strict_lecture_only';
  document.getElementById('student-known').value = workspace?.student_context?.known || '';
  document.getElementById('student-weak').value = workspace?.student_context?.weak_areas || '';
  document.getElementById('student-goals').value = workspace?.student_context?.goals || '';

  const reviseBtn = document.getElementById('revise-study-plan');
  const plan = workspace?.active_study_plan;
  if (reviseBtn) reviseBtn.classList.toggle('hidden', !plan);
  if (!plan) {
    output.innerHTML = '<div class="muted">No study plan yet.</div>';
    return;
  }
  output.innerHTML = `
    <div class="card plan-summary-card">
      <div class="plan-summary-header">
        <div>
          <div class="small muted" style="margin-bottom:4px;">Generated study plan</div>
          <div style="font-weight:600;font-size:15px;">${escapeHtml(plan.topic_text || 'Study plan')}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
          ${badge(plan.grounding_mode || 'strict_lecture_only')} ${badge(`${plan.time_budget_minutes} min`)}
        </div>
      </div>
      <div class="small muted" style="margin-top:6px;">Created ${escapeHtml(formatDate(plan.created_at))}</div>
      <div class="small muted" style="margin-top:6px;">Each item below links back to its cited slide(s) in the source viewer.</div>
    </div>
    ${renderTailoringSummary(plan.tailoring_summary)}
    ${renderPlanUncertainty(plan.uncertainty || [])}
    ${renderSection('Prerequisite knowledge', plan.prerequisites, (item) => `
      <div><strong>${escapeHtml(item.concept_name)}</strong></div>
      <div class="small">${escapeHtml(item.why_needed)}</div>
      <div class="small muted">${escapeHtml(formatSupportStatus(item.support_status))}</div>
    `)}
    ${renderSection('Study sequence', plan.study_sequence, (item) => `
      <div><strong>${escapeHtml(item.title)}</strong></div>
      <div class="small">${escapeHtml(item.objective)}</div>
      <div class="small muted">${item.recommended_time_minutes} min · ${escapeHtml(formatSupportStatus(item.support_status))}</div>
      ${item.milestone ? `<div class="small plan-milestone"><strong>Milestone:</strong> ${escapeHtml(item.milestone)}</div>` : ''}
      ${renderPlanTasks(item.tasks || [])}
    `)}
    ${renderSection('Common mistakes', plan.common_mistakes, (item) => `
      <div><strong>${escapeHtml(item.pattern)}</strong></div>
      <div class="small">${escapeHtml(item.why_it_happens)}</div>
      <div class="small muted">${escapeHtml(item.prevention_advice)}</div>
      <div class="small muted">${escapeHtml(formatSupportStatus(item.support_status))}</div>
    `)}
  `;
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
}

function renderSection(title, items, itemRenderer) {
  const rendered = (items || []).map((item) => `
    <div class="card" data-section-citations='${JSON.stringify(item.citations || []).replace(/'/g, '&apos;')}'>
      ${itemRenderer(item)}
      <div>${renderCitationButtons(item.citations || [])}</div>
    </div>
  `).join('');
  return `<div class="stack"><h3>${escapeHtml(title)}</h3>${rendered || '<div class="muted">No items.</div>'}</div>`;
}

function renderChat() {
  const workspace = state.activeWorkspace;
  document.getElementById('chat-grounding-mode').value = workspace?.grounding_mode || 'strict_lecture_only';
  const responseStyle = document.getElementById('chat-response-style');
  if (!['standard', 'concise', 'step_by_step'].includes(responseStyle.value)) responseStyle.value = 'standard';
  const output = document.getElementById('chat-output');
  const conversation = workspace?.active_conversation;
  if (!conversation) {
    output.innerHTML = '<div class="muted">No active chat yet. Start a new chat on the current materials.</div>';
    return;
  }
  output.innerHTML = (conversation.messages || []).map((message) => {
    if (message.role === 'user') {
      return `
        <div class="card">
          <div><strong>You</strong></div>
          <div class="message-text">${escapeHtml(message.text || '')}</div>
          <div class="small muted">${escapeHtml(formatDate(message.created_at))}${message.pending ? ' • pending' : ''}</div>
        </div>`;
    }
    const sections = (message.reply_sections || []).map((section, index) => `
      <div class="card" data-section-citations='${JSON.stringify(section.citations || []).replace(/'/g, '&apos;')}'>
        <div><strong>${escapeHtml(section.heading || `Section ${index + 1}`)}</strong></div>
        <div class="message-text">${escapeHtml(section.text || '')}</div>
        <div class="small muted">${escapeHtml(formatSupportStatus(section.support_status))}</div>
        <div>${renderCitationButtons(section.citations || [])}</div>
      </div>
    `).join('');
    const clarifying = message.clarifying_question?.prompt ? `<div class="warning">Clarifying question: ${escapeHtml(message.clarifying_question.prompt)}</div>` : '';
    return `<div class="stack"><div class="small muted">Assistant • ${escapeHtml(formatDate(message.created_at))}</div>${sections}${clarifying}</div>`;
  }).join('');
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
  output.scrollTop = output.scrollHeight;
}

function renderPractice() {
  const workspace = state.activeWorkspace;
  const templates = (workspace?.materials || []).filter((material) => material.role === 'practice_template');
  const templateSelect = document.getElementById('practice-template');
  templateSelect.innerHTML = `<option value="">None</option>` + templates.map((material) => `<option value="${material.material_id}">${escapeHtml(material.title)}</option>`).join('');
  const practice = workspace?.active_practice_set;
  const output = document.getElementById('practice-output');
  if (!practice) {
    output.innerHTML = '<div class="muted">No practice set yet.</div>';
    return;
  }
  output.innerHTML = `
    <div class="card">
      <h3>Active practice set</h3>
      <div>${badge(practice.generation_mode || 'mixed')}</div>
      <div class="small muted">Created ${escapeHtml(formatDate(practice.created_at))}</div>
      <div class="small muted">Coverage: ${escapeHtml(practice.coverage_report?.notes || '')}</div>
    </div>
    ${(practice.questions || []).map((question) => `
      <div class="card" data-section-citations='${JSON.stringify(question.citations || []).replace(/'/g, '&apos;')}'>
        <div><strong>${escapeHtml(question.question_type)}</strong></div>
        <div>${escapeHtml(question.stem)}</div>
        <details>
          <summary>Expected answer</summary>
          <div class="small">${escapeHtml(question.expected_answer || '')}</div>
        </details>
        ${(question.rubric || []).length ? `<details><summary>Rubric</summary><div class="small">${escapeHtml((question.rubric || []).map((item) => `${item.criterion}: ${item.description} (${item.points})`).join(' • '))}</div></details>` : ''}
        <div class="small muted">Difficulty: ${escapeHtml(question.difficulty || '')} • Covered slides: ${escapeHtml((question.covered_slides || []).join(', '))}</div>
        <div>${renderCitationButtons(question.citations || [])}</div>
      </div>
    `).join('')}
  `;
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
}

function renderHistory() {
  const output = document.getElementById('history-output');
  const history = (state.activeWorkspace?.history || []).filter((entry) => entry.artifact_type !== 'practice_set');
  if (!history.length) {
    output.innerHTML = '<div class="muted">No history yet.</div>';
    return;
  }
  output.innerHTML = history.slice().sort((a, b) => (a.created_at > b.created_at ? -1 : 1)).map((entry) => `
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:flex-start;gap:12px;">
        <div>
          <div><strong>${escapeHtml(entry.artifact_type)}</strong> • ${escapeHtml(entry.title || '')}</div>
          <div class="small muted">${escapeHtml(formatDate(entry.created_at))} • parent ${escapeHtml(entry.parent_artifact_id || '—')} • ${entry.active ? 'current' : 'prior'}</div>
        </div>
        <button type="button" class="secondary history-open" data-artifact-type="${escapeHtml(entry.artifact_type)}" data-artifact-id="${escapeHtml(entry.artifact_id)}">Open</button>
      </div>
    </div>
  `).join('');
  output.querySelectorAll('.history-open').forEach((button) => {
    button.addEventListener('click', async () => {
      const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/history/${button.dataset.artifactType}/${button.dataset.artifactId}/activate`, {
        method: 'POST',
      });
      state.activeWorkspace = payload.workspace;
      renderWorkspace();
      if (button.dataset.artifactType === 'study_plan') setTab('overview');
      if (button.dataset.artifactType === 'conversation') setTab('ask');
    });
  });
}

function renderJobs() {
  const container = document.getElementById('material-jobs');
  const jobs = state.activeWorkspace?.jobs || [];
  if (!jobs.length) {
    container.innerHTML = '';
    return;
  }
  const activeStatuses = new Set(['queued', 'running', 'submitted', 'waiting_for_service', 'needs_user_input']);
  const currentJobs = jobs.filter((job) => activeStatuses.has(job.status) || (job.status === 'failed' && !job.finalized));
  container.innerHTML = currentJobs.map((job) => `
    <div class="card">
      <div><strong>${escapeHtml(job.operation)}</strong></div>
      <div class="small muted">${escapeHtml(job.stage || '')} • ${escapeHtml(job.status || '')} • progress ${escapeHtml(String(job.progress || 0))}%</div>
      <div class="small">${escapeHtml(job.message || '')}</div>
      ${job.error?.message ? `<div class="error small">${escapeHtml(job.error.message)}</div>` : ''}
      ${job.user_action?.prompt ? `<div class="warning small">Needs input: ${escapeHtml(job.user_action.prompt)}</div>` : ''}
    </div>
  `).join('');
}

function renderWorkspace() {
  const empty = document.getElementById('workspace-empty-state');
  const view = document.getElementById('workspace-view');
  if (!state.activeWorkspace) {
    empty.classList.remove('hidden');
    view.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  view.classList.remove('hidden');
  document.getElementById('workspace-title').textContent = state.activeWorkspace.display_name;
  document.getElementById('workspace-meta').textContent = `Opened ${formatDate(state.activeWorkspace.last_opened_at)} • ${state.activeWorkspace.grounding_mode}`;
  renderMaterials();
  renderJobs();
  renderStudyPlan();
  renderChat();
  renderHistory();
  applyServiceGating();
}

function scrollSourceViewerIntoView() {
  const panel = document.querySelector('.source-viewer');
  if (!panel) return;
  const rect = panel.getBoundingClientRect();
  const visible = rect.top >= 0 && rect.bottom <= window.innerHeight;
  if (!visible) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function openCitation(citation, citationList = [citation], index = 0) {
  state.citationList = citationList;
  state.citationIndex = index;
  document.querySelector('.source-viewer').classList.remove('hidden');
  const empty = document.getElementById('source-empty');
  const view = document.getElementById('source-view');
  empty.classList.add('hidden');
  view.classList.remove('hidden');
  renderSourceViewer(citation);
  scrollSourceViewerIntoView();
}

function renderSourceViewer(citation) {
  const { contentAvailable } = serviceAvailability();
  document.getElementById('source-meta').textContent = `${citation.material_title || citation.material_id || 'Source'} · slide ${citation.slide_number || '?'}`;
  const snippetEl = document.getElementById('source-snippet');
  const snippetText = citation.snippet_text || '';
  snippetEl.innerHTML = snippetText
    ? `<div class="source-snippet-label small muted">Cited passage</div><div class="source-snippet-text">${escapeHtml(snippetText)}</div>`
    : `<div class="source-snippet-label small muted">No snippet available.</div>`;
  document.getElementById('source-preview').src = citation.preview_url || '';
  if (!contentAvailable) {
    document.getElementById('source-message').textContent = 'Preview unavailable because the content service is offline. Citation metadata is still shown.';
  } else {
    document.getElementById('source-message').textContent = citation.preview_url ? '' : 'Preview unavailable. Citation metadata is still shown.';
  }
  const sourceButton = document.getElementById('source-open');
  sourceButton.disabled = !contentAvailable || !citation.source_open_url;
  sourceButton.onclick = () => {
    if (contentAvailable && citation.source_open_url) window.open(citation.source_open_url, '_blank', 'noopener');
  };
}

async function pollJob(jobId) {
  if (!state.activeWorkspace) return;
  const workspaceId = state.activeWorkspace.workspace_id;
  while (true) {
    const payload = await api(`/api/jobs/${jobId}?workspace_id=${workspaceId}`);
    const job = payload.job;
    setGlobalActivity(job.message || `${job.operation || 'Operation'} is running`, true);
    const refreshed = await api(`/api/workspaces/${workspaceId}`);
    state.activeWorkspace = refreshed.workspace;
    renderWorkspace();
    if (['succeeded', 'failed', 'needs_user_input'].includes(job.status)) {
      if (job.status === 'succeeded') showTransientMessage(job.message || 'Operation completed.', 'success');
      if (job.status === 'failed') showTransientMessage(job.error?.message || job.message || 'Operation failed.', 'error');
      if (job.status === 'needs_user_input') showTransientMessage(job.user_action?.prompt || 'The service needs more input.', 'warning');
      clearGlobalActivitySoon();
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
}

async function createConversationIfNeeded() {
  if (state.activeWorkspace?.active_conversation) return state.activeWorkspace.active_conversation;
  const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'Workspace Q&A', grounding_mode: state.activeWorkspace.grounding_mode }),
  });
  const refreshed = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
  state.activeWorkspace = refreshed.workspace;
  renderWorkspace();
  return payload.conversation;
}

function bindEvents() {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => setTab(button.dataset.tab));
  });

  document.getElementById('create-workspace-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const displayName = document.getElementById('workspace-name').value.trim();
    const payload = await api('/api/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: displayName }),
    });
    document.getElementById('workspace-name').value = '';
    await loadWorkspaces();
    state.activeWorkspace = payload.workspace;
    renderWorkspace();
  });

  document.getElementById('material-import-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const fileInput = document.getElementById('material-file');
    const files = Array.from(fileInput.files || []);
    const text = document.getElementById('material-text').value.trim();
    if (!files.length && !text) {
      showTransientMessage('Please enter text or select a file to import.', 'warning');
      return;
    }
    const submitButton = event.currentTarget.querySelector('button[type="submit"]');
    if (submitButton) submitButton.disabled = true;
    const title = document.getElementById('material-title').value;
    const role = document.getElementById('material-role').value;

    try {
      if (files.length) {
        const seen = new Set();
        const queue = [];
        const uploadItems = [];
        files.forEach((file) => {
          const key = selectedFileKey(file);
          if (seen.has(key)) {
            uploadItems.push({ name: file.name, size: file.size, status: 'skipped' });
            return;
          }
          seen.add(key);
          queue.push(file);
          uploadItems.push({ name: file.name, size: file.size, status: 'queued' });
        });
        renderMaterialUploadStatus(uploadItems, true);
        for (let index = 0; index < queue.length; index += 1) {
          const file = queue[index];
          const item = uploadItems.find((candidate) => candidate.name === file.name && candidate.size === file.size && candidate.status === 'queued');
          if (item) item.status = 'uploading';
          renderMaterialUploadStatus(uploadItems, true);

          const form = new FormData();
          // Preserve existing single-file title behavior while letting multi-file imports default to each filename.
          form.append('title', queue.length === 1 ? title : '');
          form.append('role', role);
          form.append('file', file);
          try {
            const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/materials/import`, { method: 'POST', body: form });
            const finishedJob = await pollJob(payload.job.job_id);
            const deduped = Boolean(finishedJob?.context?.deduplication?.skipped);
            if (item) item.status = deduped ? 'skipped' : 'done';
          } catch (_error) {
            if (item) item.status = 'failed';
            renderMaterialUploadStatus(uploadItems, true);
            throw _error;
          }
          renderMaterialUploadStatus(uploadItems, true);
        }
        renderMaterialUploadStatus(uploadItems, false);
      } else {
        renderMaterialUploadStatus([], false);
        const form = new FormData();
        form.append('title', title);
        form.append('role', role);
        form.append('kind', 'pasted_text');
        form.append('text', text);
        const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/materials/import`, { method: 'POST', body: form });
        await pollJob(payload.job.job_id);
      }

      document.getElementById('material-title').value = '';
      document.getElementById('material-text').value = '';
      document.getElementById('material-file').value = '';
      await loadWorkspaces();
    } finally {
      if (submitButton) submitButton.disabled = false;
    }
  });

  document.getElementById('study-plan-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const timeBudgetMinutes = optionalPositiveIntegerValue('time-budget');
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/study-plans/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic_text: document.getElementById('topic-text').value,
        time_budget_minutes: timeBudgetMinutes,
        grounding_mode: document.getElementById('grounding-mode').value,
        student_context: {
          known: document.getElementById('student-known').value,
          weak_areas: document.getElementById('student-weak').value,
          goals: document.getElementById('student-goals').value,
        },
      }),
    });
    await pollJob(payload.job.job_id);
    await loadWorkspaces();
  });

  document.getElementById('revise-study-plan').addEventListener('click', async () => {
    if (!state.activeWorkspace?.active_study_plan) return;
    const note = window.prompt('Optional correction note for this revision:', 'regenerate this section while preserving all locked items') || '';
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/study-plans/${state.activeWorkspace.active_study_plan.study_plan_id}/revise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_section: 'study_sequence',
        feedback_note: note,
        locked_sections: ['prerequisites'],
      }),
    });
    await pollJob(payload.job.job_id);
    await loadWorkspaces();
  });

  document.getElementById('new-conversation').addEventListener('click', async () => {
    if (!state.activeWorkspace) return;
    await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: `Chat ${new Date().toLocaleTimeString()}`, grounding_mode: state.activeWorkspace.grounding_mode }),
    });
    const refreshed = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
    state.activeWorkspace = refreshed.workspace;
    renderWorkspace();
    setTab('ask');
  });

  document.getElementById('clear-conversation').addEventListener('click', async () => {
    if (!state.activeWorkspace?.active_conversation) return;
    await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations/${state.activeWorkspace.active_conversation.conversation_id}/clear`, { method: 'POST' });
    const refreshed = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
    state.activeWorkspace = refreshed.workspace;
    renderWorkspace();
  });

  document.getElementById('chat-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const sendButton = event.currentTarget.querySelector('button[type="submit"]');
    const questionInput = document.getElementById('chat-question');
    if (sendButton) sendButton.disabled = true;
    const questionText = questionInput.value.trim();
    if (!questionText) {
      if (sendButton) sendButton.disabled = false;
      return;
    }
    try {
      const conversation = await createConversationIfNeeded();
      setGlobalActivity('Sending your question...', true);
      const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations/${conversation.conversation_id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: questionText,
          grounding_mode: document.getElementById('chat-grounding-mode').value,
          response_style: document.getElementById('chat-response-style').value,
        }),
      });
      questionInput.value = '';
      const refreshed = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
      state.activeWorkspace = refreshed.workspace;
      renderWorkspace();
      await pollJob(payload.job.job_id);
      await loadWorkspaces();
    } catch (error) {
      showTransientMessage(error.message, 'error');
      clearGlobalActivitySoon();
    } finally {
      if (sendButton) sendButton.disabled = false;
    }
  });

  document.getElementById('source-prev').addEventListener('click', () => {
    if (!state.citationList.length) return;
    state.citationIndex = (state.citationIndex - 1 + state.citationList.length) % state.citationList.length;
    renderSourceViewer(state.citationList[state.citationIndex]);
  });

  document.getElementById('source-next').addEventListener('click', () => {
    if (!state.citationList.length) return;
    state.citationIndex = (state.citationIndex + 1) % state.citationList.length;
    renderSourceViewer(state.citationList[state.citationIndex]);
  });
}

async function init() {
  bindEvents();
  await loadStatus();
  await loadWorkspaces();
}

init().catch((error) => {
  console.error(error);
  showTransientMessage(error.message, 'error');
});
