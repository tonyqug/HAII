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

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showTransientMessage(text, type = 'muted') {
  const container = document.getElementById('system-status');
  const div = document.createElement('div');
  div.className = type;
  div.textContent = text;
  container.prepend(div);
  setTimeout(() => {
    if (div.parentNode) div.parentNode.removeChild(div);
  }, 5000);
}

async function loadStatus() {
  state.status = await api('/api/status');
  renderSystemStatus();
}

function renderSystemStatus() {
  const container = document.getElementById('system-status');
  if (!state.status) {
    container.innerHTML = '<div class="muted">Loading service status…</div>';
    return;
  }
  const content = state.status.services.content;
  const learning = state.status.services.learning;
  container.innerHTML = `
    <div class="card">
      <div><strong>Mode:</strong> ${escapeHtml(state.status.effective_mode)}</div>
      <div>${badge(`content: ${content.available ? 'available' : 'unavailable'}`, content.available ? 'ok' : 'danger')} ${badge(`learning: ${learning.available ? 'available' : 'unavailable'}`, learning.available ? 'ok' : 'danger')}</div>
      <div class="small muted">Content URL: ${escapeHtml(content.base_url || '')}</div>
      <div class="small muted">Learning URL: ${escapeHtml(learning.base_url || '')}</div>
    </div>`;
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
    : 'Learning service is unavailable. Study plans, grounded chat, and practice generation are disabled until it is healthy.';

  setServiceNote('materials-service-note', contentMessage);
  setServiceNote(
    'source-service-note',
    contentAvailable ? '' : 'Source viewer cannot resolve slide/page URLs while the content service is offline.'
  );
  setServiceNote('study-service-note', learningMessage);
  setServiceNote('chat-service-note', learningMessage);
  setServiceNote('practice-service-note', learningMessage);

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
      '#practice-form input',
      '#practice-form select',
      '#practice-form button',
      '#revise-practice',
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
  container.innerHTML = state.workspaces.map((workspace) => `
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:flex-start;">
        <div>
          <div><strong>${escapeHtml(workspace.display_name)}</strong></div>
          <div class="small muted">Opened ${escapeHtml(formatDate(workspace.last_opened_at))}</div>
          <div class="small muted">Materials: ${workspace.material_counts.total}, ready ${workspace.material_counts.ready}, processing ${workspace.material_counts.processing}</div>
          <div class="small muted">Artifacts: plans ${workspace.artifact_counts.study_plans}, chats ${workspace.artifact_counts.conversations}, practice ${workspace.artifact_counts.practice_sets}</div>
        </div>
        <div>${badge(workspace.grounding_mode || 'strict_lecture_only')}</div>
      </div>
      <div class="row" style="margin-top:10px;">
        <button type="button" data-open-workspace="${workspace.workspace_id}">Open</button>
        <button type="button" class="secondary" data-duplicate-workspace="${workspace.workspace_id}">Duplicate</button>
        <button type="button" class="secondary" data-archive-workspace="${workspace.workspace_id}">Archive</button>
        <button type="button" class="danger" data-delete-workspace="${workspace.workspace_id}">Delete</button>
      </div>
    </div>
  `).join('');

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
          <button type="button" class="secondary open-material-source" data-material-id="${material.material_id}">Open source</button>
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
}

function renderCitationButtons(citations = []) {
  return citations.map((citation, index) => `
    <button type="button" class="secondary citation-button" data-citation-index="${index}">Slide ${escapeHtml(String(citation.slide_number || '?'))}</button>
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

function renderStudyPlan() {
  const output = document.getElementById('study-plan-output');
  const workspace = state.activeWorkspace;
  document.getElementById('topic-text').value = workspace?.topic_text || '';
  document.getElementById('time-budget').value = workspace?.time_budget_minutes || '';
  document.getElementById('grounding-mode').value = workspace?.grounding_mode || 'strict_lecture_only';
  document.getElementById('student-known').value = workspace?.student_context?.known || '';
  document.getElementById('student-weak').value = workspace?.student_context?.weak_areas || '';
  document.getElementById('student-goals').value = workspace?.student_context?.goals || '';

  const plan = workspace?.active_study_plan;
  if (!plan) {
    output.innerHTML = '<div class="muted">No study plan yet.</div>';
    return;
  }
  output.innerHTML = `
    <div class="card">
      <h3>Active plan</h3>
      <div>${badge(plan.grounding_mode || 'strict_lecture_only')} ${badge(`${plan.time_budget_minutes} min`)}</div>
      <div class="small muted">Created ${escapeHtml(formatDate(plan.created_at))} • ${escapeHtml(plan.topic_text || '')}</div>
    </div>
    ${renderSection('Prerequisite knowledge', plan.prerequisites, (item) => `
      <div><strong>${escapeHtml(item.concept_name)}</strong></div>
      <div class="small">${escapeHtml(item.why_needed)}</div>
      <div class="small muted">${escapeHtml(item.support_status)}</div>
    `)}
    ${renderSection('Study sequence', plan.study_sequence, (item) => `
      <div><strong>${escapeHtml(item.title)}</strong></div>
      <div class="small">${escapeHtml(item.objective)}</div>
      <div class="small muted">${escapeHtml(item.support_status)} • ${item.recommended_time_minutes} minutes</div>
      <div class="small muted">Tasks: ${escapeHtml((item.tasks || []).join('; '))}</div>
    `)}
    ${renderSection('Common mistakes', plan.common_mistakes, (item) => `
      <div><strong>${escapeHtml(item.pattern)}</strong></div>
      <div class="small">${escapeHtml(item.why_it_happens)}</div>
      <div class="small muted">${escapeHtml(item.prevention_advice)}</div>
      <div class="small muted">${escapeHtml(item.support_status)}</div>
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
          <div>${escapeHtml(message.text || '')}</div>
          <div class="small muted">${escapeHtml(formatDate(message.created_at))}${message.pending ? ' • pending' : ''}</div>
        </div>`;
    }
    const sections = (message.reply_sections || []).map((section, index) => `
      <div class="card" data-section-citations='${JSON.stringify(section.citations || []).replace(/'/g, '&apos;')}'>
        <div><strong>${escapeHtml(section.heading || `Section ${index + 1}`)}</strong></div>
        <div>${escapeHtml(section.text || '')}</div>
        <div class="small muted">${escapeHtml(section.support_status || '')}</div>
        <div>${renderCitationButtons(section.citations || [])}</div>
      </div>
    `).join('');
    const clarifying = message.clarifying_question?.prompt ? `<div class="warning">Clarifying question: ${escapeHtml(message.clarifying_question.prompt)}</div>` : '';
    return `<div class="stack">${sections}${clarifying}</div>`;
  }).join('');
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
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
  const history = state.activeWorkspace?.history || [];
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
      if (button.dataset.artifactType === 'practice_set') setTab('practice');
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
  const currentJobs = jobs.filter((job) => ['queued', 'running', 'submitted', 'waiting_for_service', 'needs_user_input', 'failed'].includes(job.status) || !job.finalized);
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
  renderPractice();
  renderHistory();
  applyServiceGating();
}

function openCitation(citation, citationList = [citation], index = 0) {
  state.citationList = citationList;
  state.citationIndex = index;
  const empty = document.getElementById('source-empty');
  const view = document.getElementById('source-view');
  empty.classList.add('hidden');
  view.classList.remove('hidden');
  renderSourceViewer(citation);
}

function renderSourceViewer(citation) {
  const { contentAvailable } = serviceAvailability();
  document.getElementById('source-meta').textContent = `${citation.material_title || citation.material_id || 'Source'} • slide ${citation.slide_number || '?'} • ${citation.support_type || ''}`;
  document.getElementById('source-snippet').textContent = citation.snippet_text || 'No snippet available.';
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
    const refreshed = await api(`/api/workspaces/${workspaceId}`);
    state.activeWorkspace = refreshed.workspace;
    renderWorkspace();
    if (['succeeded', 'failed', 'needs_user_input'].includes(job.status)) {
      if (job.status === 'failed') showTransientMessage(job.error?.message || job.message || 'Operation failed.', 'error');
      if (job.status === 'needs_user_input') showTransientMessage(job.user_action?.prompt || 'The service needs more input.', 'warning');
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
    const form = new FormData();
    form.append('title', document.getElementById('material-title').value);
    form.append('role', document.getElementById('material-role').value);
    const text = document.getElementById('material-text').value;
    const fileInput = document.getElementById('material-file');
    if (fileInput.files[0]) {
      form.append('file', fileInput.files[0]);
    } else {
      form.append('kind', 'pasted_text');
      form.append('text', text);
    }
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/materials/import`, { method: 'POST', body: form });
    await pollJob(payload.job.job_id);
    document.getElementById('material-title').value = '';
    document.getElementById('material-text').value = '';
    document.getElementById('material-file').value = '';
    await loadWorkspaces();
  });

  document.getElementById('study-plan-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/study-plans/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic_text: document.getElementById('topic-text').value,
        time_budget_minutes: Number(document.getElementById('time-budget').value),
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
    try {
      const conversation = await createConversationIfNeeded();
      const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations/${conversation.conversation_id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: document.getElementById('chat-question').value,
          grounding_mode: document.getElementById('chat-grounding-mode').value,
          response_style: document.getElementById('chat-response-style').value,
        }),
      });
      document.getElementById('chat-question').value = '';
      const refreshed = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
      state.activeWorkspace = refreshed.workspace;
      renderWorkspace();
      await pollJob(payload.job.job_id);
      await loadWorkspaces();
    } catch (error) {
      showTransientMessage(error.message, 'error');
    }
  });

  document.getElementById('practice-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        generation_mode: document.getElementById('practice-mode').value,
        question_count: Number(document.getElementById('practice-count').value),
        difficulty: document.getElementById('practice-difficulty').value,
        answer_key: document.getElementById('practice-answer-key').checked,
        rubric: document.getElementById('practice-rubrics').checked,
        template_material_id: document.getElementById('practice-template').value || null,
        grounding_mode: state.activeWorkspace.grounding_mode,
      }),
    });
    await pollJob(payload.job.job_id);
    await loadWorkspaces();
  });

  document.getElementById('revise-practice').addEventListener('click', async () => {
    if (!state.activeWorkspace?.active_practice_set) return;
    const practiceSet = state.activeWorkspace.active_practice_set;
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/${practiceSet.practice_set_id}/revise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'create_variant',
        locked_question_ids: practiceSet.questions.slice(0, 1).map((item) => item.question_id),
      }),
    });
    await pollJob(payload.job.job_id);
    await loadWorkspaces();
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
