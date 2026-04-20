const state = {
  status: null,
  workspaces: [],
  activeWorkspace: null,
  citationList: [],
  citationIndex: 0,
};

const SUPPORT_STATUS_LABELS = {
  slide_grounded: 'Grounded in your materials',
  inferred_from_slides: 'Grounded across your materials',
  annotation_grounded: 'Grounded in your notes',
  grounded: 'Grounded in your materials',
  partially_grounded: 'Partially grounded',
  insufficient_evidence: 'Insufficient lecture evidence',
  external_supplement: 'External supplement',
  not_grounded: 'Not grounded',
  ungrounded: 'Not grounded',
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

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let scaled = value;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  return `${scaled.toFixed(scaled >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function badge(text, cls = '') {
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

function formatSupportStatus(status) {
  return SUPPORT_STATUS_LABELS[status] || String(status || '').replace(/_/g, ' ');
}

function showTransientMessage(text, type = 'muted') {
  const container = document.getElementById('global-toast-stack');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = text;
  container.append(toast);
  if (container.children.length > 5) container.removeChild(container.firstElementChild);
  setTimeout(() => {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
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

function optionalPositiveIntegerValue(inputId) {
  const raw = document.getElementById(inputId)?.value?.trim() || '';
  if (!raw) return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.floor(parsed);
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
    note.textContent = '';
    note.classList.add('hidden');
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

  const done = items.filter((item) => item.status === 'done').length;
  const skipped = items.filter((item) => item.status === 'skipped').length;
  const failed = items.filter((item) => item.status === 'failed').length;
  const processed = done + skipped + failed;
  container.classList.remove('hidden');
  progress.value = Math.round((processed / items.length) * 100);
  count.textContent = `${processed}/${items.length}`;
  summary.textContent = active
    ? `Uploading files... ${done} imported, ${skipped} skipped, ${failed} failed`
    : `Upload complete: ${done} imported, ${skipped} skipped, ${failed} failed`;
  list.innerHTML = items.map((item) => `
    <div class="upload-item ${item.status}">
      <div>${escapeHtml(item.name)}</div>
      <div class="muted">${escapeHtml(uploadStatusLabel(item.status))}</div>
      <div class="muted">${escapeHtml(formatBytes(item.size))}</div>
    </div>
  `).join('');
  setGlobalActivity(summary.textContent, active);
  if (!active) clearGlobalActivitySoon(3000);
}

async function loadStatus() {
  state.status = await api('/api/status');
  renderSystemStatus();
}

function renderSystemStatus() {
  const container = document.getElementById('system-status');
  if (!container) return;
  if (!state.status) {
    container.innerHTML = '<div class="muted small">Loading service status...</div>';
    return;
  }
  const content = state.status.services.content;
  const learning = state.status.services.learning;
  const mode = state.status.effective_mode;
  const degraded = !content.available || !learning.available;
  const openAttr = degraded && mode !== 'mock' ? 'open' : '';
  container.innerHTML = `
    <details ${openAttr} style="margin-bottom:8px;">
      <summary class="small muted" style="cursor:pointer;">Mode: ${escapeHtml(mode)}${degraded && mode !== 'mock' ? ' - services offline' : ''}</summary>
      <div class="card" style="margin-top:8px;">
        <div>${badge(`content: ${content.available ? 'available' : 'unavailable'}`, content.available ? 'ok' : 'danger')} ${badge(`learning: ${learning.available ? 'available' : 'unavailable'}`, learning.available ? 'ok' : 'danger')}</div>
      </div>
    </details>
  `;
  applyServiceGating();
}

function applyServiceGating() {
  const { contentAvailable, learningAvailable } = serviceAvailability();
  setServiceNote(
    'materials-service-note',
    contentAvailable ? '' : 'Content service is unavailable. Imports, ingestion updates, and source-grounded previews are disabled until it recovers.'
  );
  setServiceNote(
    'source-service-note',
    contentAvailable ? '' : 'Source viewer cannot resolve slide or page URLs while the content service is offline.'
  );
  setServiceNote(
    'practice-service-note',
    learningAvailable ? '' : 'Learning service is unavailable. Practice generation and revision are disabled until it recovers.'
  );
  setServiceNote(
    'chat-service-note',
    learningAvailable ? '' : 'Learning service is unavailable. Grounded chat is disabled until it recovers.'
  );

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
      '#practice-form input',
      '#practice-form select',
      '#practice-form textarea',
      '#practice-form button',
      '#revise-practice',
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
  if (!container) return;
  if (!state.workspaces.length) {
    container.innerHTML = '<div class="muted">No workspaces yet.</div>';
    return;
  }
  container.innerHTML = state.workspaces.map((workspace) => {
    const active = state.activeWorkspace?.workspace_id === workspace.workspace_id;
    const hasProcessing = workspace.material_counts.processing > 0;
    return `
      <div class="workspace-card${active ? ' active' : ''}">
        <div class="workspace-card-header">
          <span class="workspace-card-name">${escapeHtml(workspace.display_name)}</span>
          ${badge(workspace.grounding_mode || 'strict_lecture_only')}
        </div>
        <div class="workspace-card-meta small muted">Opened ${escapeHtml(formatDate(workspace.last_opened_at))}</div>
        <div class="workspace-card-stats small">
          <span class="ws-stat${hasProcessing ? ' processing' : ''}">
            ${workspace.material_counts.ready} material${workspace.material_counts.ready !== 1 ? 's' : ''}${hasProcessing ? ` - ${workspace.material_counts.processing} processing` : ''}
          </span>
          <span class="ws-stat-sep">|</span>
          <span class="ws-stat">${workspace.artifact_counts.practice_sets} test${workspace.artifact_counts.practice_sets !== 1 ? 's' : ''}</span>
          <span class="ws-stat-sep">|</span>
          <span class="ws-stat">${workspace.artifact_counts.conversations} chat${workspace.artifact_counts.conversations !== 1 ? 's' : ''}</span>
        </div>
        <div class="workspace-card-actions">
          <button type="button" class="ws-open-btn" data-open-workspace="${workspace.workspace_id}">${active ? 'Active' : 'Open'}</button>
          <button type="button" class="secondary ws-action-btn" data-duplicate-workspace="${workspace.workspace_id}">Duplicate</button>
          <button type="button" class="secondary ws-action-btn" data-archive-workspace="${workspace.workspace_id}">Archive</button>
          <button type="button" class="danger ws-action-btn" data-delete-workspace="${workspace.workspace_id}">Delete</button>
        </div>
      </div>
    `;
  }).join('');

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
  return ['default', 'focus', 'exclude']
    .map((option) => `<option value="${option}" ${current === option ? 'selected' : ''}>${option}</option>`)
    .join('');
}

function renderMaterials() {
  const list = document.getElementById('material-list');
  const workspace = state.activeWorkspace;
  const materials = workspace?.materials || [];
  if (!list) return;
  if (!materials.length) {
    list.innerHTML = '<div class="muted">No materials in this workspace yet.</div>';
    return;
  }
  list.innerHTML = materials.map((material) => {
    const preference = workspace.material_preferences?.[material.material_id] || 'default';
    const statusClass = material.processing_status === 'ready'
      ? 'ok'
      : material.processing_status === 'failed'
        ? 'danger'
        : 'warn';
    return `
      <div class="card">
        <div class="row" style="justify-content:space-between;align-items:flex-start;">
          <div>
            <div><strong>${escapeHtml(material.title)}</strong></div>
            <div class="small muted">${escapeHtml(material.role)} | ${escapeHtml(material.kind)} | ${material.page_count ?? 0} pages/slides</div>
            <div>${badge(material.processing_status, statusClass)} ${badge(preference)}</div>
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
            <select class="material-preference" data-material-id="${material.material_id}">${materialPreferenceOptions(preference)}</select>
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
        const citation = {
          material_id: material.material_id,
          material_title: material.title,
          slide_id: slide.slide_id,
          slide_number: slide.slide_number,
          snippet_text: slide.snippet_text,
          preview_url: slide.preview_url || '',
          source_open_url: slide.source_open_url || material.source_view_url || '',
          support_type: 'source_preview',
        };
        openCitation(citation, [citation], 0);
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
    <button type="button" class="secondary citation-button" data-citation-index="${index}" title="View source slide in the source viewer">
      Slide ${escapeHtml(String(citation.slide_number || '?'))}
    </button>
  `).join('');
}

function wireCitationButtons(container, citations) {
  container.querySelectorAll('[data-citation-index]').forEach((button) => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.citationIndex);
      openCitation(citations[index], citations, index);
    });
  });
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
  const sourceViewer = document.querySelector('.source-viewer');
  if (sourceViewer) sourceViewer.classList.remove('hidden');
  document.getElementById('source-empty')?.classList.add('hidden');
  document.getElementById('source-view')?.classList.remove('hidden');
  renderSourceViewer(citation);
  scrollSourceViewerIntoView();
}

function renderSourceViewer(citation) {
  const { contentAvailable } = serviceAvailability();
  const preview = document.getElementById('source-preview');
  const openButton = document.getElementById('source-open');
  const message = document.getElementById('source-message');
  document.getElementById('source-meta').textContent = `${citation.material_title || citation.material_id || 'Source'} | slide ${citation.slide_number || '?'}`;
  document.getElementById('source-snippet').innerHTML = citation.snippet_text
    ? `<div class="source-snippet-label small muted">Cited passage</div><div class="source-snippet-text">${escapeHtml(citation.snippet_text)}</div>`
    : '<div class="source-snippet-label small muted">No snippet available.</div>';
  if (preview) preview.src = citation.preview_url || '';
  if (message) {
    message.textContent = !contentAvailable
      ? 'Preview unavailable because the content service is offline. Citation metadata is still shown.'
      : citation.preview_url
        ? ''
        : 'Preview unavailable. Citation metadata is still shown.';
  }
  if (openButton) {
    openButton.disabled = !contentAvailable || !citation.source_open_url;
    openButton.onclick = () => {
      if (contentAvailable && citation.source_open_url) {
        window.open(citation.source_open_url, '_blank', 'noopener');
      }
    };
  }
}

function renderChat() {
  const workspace = state.activeWorkspace;
  const output = document.getElementById('chat-output');
  const grounding = document.getElementById('chat-grounding-mode');
  const responseStyle = document.getElementById('chat-response-style');
  if (grounding) grounding.value = workspace?.grounding_mode || 'strict_lecture_only';
  if (responseStyle && !['standard', 'concise', 'step_by_step'].includes(responseStyle.value)) {
    responseStyle.value = 'standard';
  }
  const conversation = workspace?.active_conversation;
  if (!output) return;
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
          <div class="small muted">${escapeHtml(formatDate(message.created_at))}${message.pending ? ' | pending' : ''}</div>
        </div>
      `;
    }
    const sections = (message.reply_sections || []).map((section, index) => `
      <div class="card" data-section-citations='${JSON.stringify(section.citations || []).replace(/'/g, '&apos;')}'>
        <div><strong>${escapeHtml(section.heading || `Section ${index + 1}`)}</strong></div>
        <div class="message-text">${escapeHtml(section.text || '')}</div>
        <div class="small muted">${escapeHtml(formatSupportStatus(section.support_status))}</div>
        <div>${renderCitationButtons(section.citations || [])}</div>
      </div>
    `).join('');
    const clarifying = message.clarifying_question?.prompt
      ? `<div class="warning">Clarifying question: ${escapeHtml(message.clarifying_question.prompt)}</div>`
      : '';
    return `<div class="stack"><div class="small muted">Assistant | ${escapeHtml(formatDate(message.created_at))}</div>${sections}${clarifying}</div>`;
  }).join('');
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
  output.scrollTop = output.scrollHeight;
}

function formatQuestionType(questionType) {
  return String(questionType || 'question').replace(/_/g, ' ');
}

function latestPracticeUserAction() {
  const jobs = state.activeWorkspace?.jobs || [];
  return jobs
    .filter((job) => ['practice_generate', 'practice_revise'].includes(job.operation))
    .sort((left, right) => String(right.updated_at || right.created_at || '').localeCompare(String(left.updated_at || left.created_at || '')))
    .find((job) => job.status === 'needs_user_input' && job.user_action?.prompt) || null;
}

function renderHumanLoopSummary(summary) {
  if (!summary) return '';
  const usedInputs = summary.used_inputs || [];
  const followUps = summary.follow_up_actions || [];
  return `
    <div class="card">
      <div class="small muted" style="margin-bottom:8px;">Human-in-the-loop summary</div>
      ${usedInputs.length ? `
        <div class="plan-input-grid" style="margin-bottom:10px;">
          ${usedInputs.map((item) => `
            <div class="plan-input-item">
              <div class="small muted">${escapeHtml(item.label || item.key || 'Input')}</div>
              <div>${escapeHtml(item.value || '')}</div>
            </div>
          `).join('')}
        </div>
      ` : '<div class="small muted">No explicit inputs were captured for this draft.</div>'}
      ${followUps.length ? `
        <div class="small muted" style="margin-bottom:6px;">Recommended next checks</div>
        <ul class="task-list">
          ${followUps.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
        </ul>
      ` : ''}
    </div>
  `;
}

function renderPracticeClarification() {
  const container = document.getElementById('practice-clarification');
  if (!container) return;
  const pendingAction = latestPracticeUserAction();
  if (!pendingAction) {
    container.innerHTML = '';
    return;
  }
  const options = pendingAction.user_action?.options || [];
  container.innerHTML = `
    <div class="card">
      <div class="small muted" style="margin-bottom:6px;">Confirmation needed before generation</div>
      <div class="warning" style="margin-bottom:10px;">${escapeHtml(pendingAction.user_action.prompt || '')}</div>
      ${options.length ? `
        <div class="small muted" style="margin-bottom:8px;">Suggested narrower grounded topics</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          ${options.map((option) => `<button type="button" class="secondary clarification-option" data-topic-option="${escapeHtml(option)}">${escapeHtml(option)}</button>`).join('')}
        </div>
      ` : ''}
      <div class="small muted" style="margin-top:10px;">Choose a narrower topic or edit the topic field, then generate again. Weakly covered areas remain marked instead of being presented as fully grounded.</div>
    </div>
  `;
  container.querySelectorAll('.clarification-option').forEach((button) => {
    button.addEventListener('click', () => {
      const input = document.getElementById('practice-topic-text');
      if (input) input.value = button.dataset.topicOption || '';
      showTransientMessage('Topic focus updated. Review the request and generate again.', 'success');
    });
  });
}

function renderQuestionChoices(question) {
  const choices = question.answer_choices || [];
  if (!choices.length) return '';
  return `
    <div class="stack" style="gap:6px;margin-top:8px;">
      ${choices.map((choice, index) => `
        <div class="small"><strong>${String.fromCharCode(65 + index)}.</strong> ${escapeHtml(choice)}</div>
      `).join('')}
    </div>
  `;
}

function renderQuestionRubric(question) {
  const rubric = question.rubric || [];
  if (!rubric.length) return '';
  return `
    <details style="margin-top:10px;">
      <summary>Rubric</summary>
      <div class="stack" style="margin-top:8px;">
        ${rubric.map((item) => `
          <div class="small">
            <strong>${escapeHtml(item.criterion || 'Criterion')}</strong>: ${escapeHtml(item.description || '')}
            ${item.points != null ? ` (${escapeHtml(String(item.points))} pts)` : ''}
          </div>
        `).join('')}
      </div>
    </details>
  `;
}

function renderPracticeQuestion(question, index) {
  return `
    <div class="card" data-section-citations='${JSON.stringify(question.citations || []).replace(/'/g, '&apos;')}'>
      <div class="row" style="justify-content:space-between;align-items:flex-start;gap:12px;">
        <div>
          <div><strong>Question ${index + 1}</strong></div>
          <div class="small muted">${escapeHtml(formatQuestionType(question.question_type))} | ${escapeHtml(question.difficulty || 'mixed difficulty')} | ${escapeHtml(String(question.estimated_minutes || '?'))} min</div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <label class="small" style="margin:0;">
            <input type="checkbox" class="practice-target-toggle" data-question-id="${escapeHtml(question.question_id)}" />
            Regenerate
          </label>
          <label class="small" style="margin:0;">
            <input type="checkbox" class="practice-lock-toggle" data-question-id="${escapeHtml(question.question_id)}" />
            Lock
          </label>
        </div>
      </div>
      <div class="message-text" style="margin-top:10px;">${escapeHtml(question.stem || '')}</div>
      ${renderQuestionChoices(question)}
      ${question.expected_answer ? `
        <details style="margin-top:10px;">
          <summary>Answer key</summary>
          <div class="message-text small" style="margin-top:8px;">${escapeHtml(question.expected_answer)}</div>
          ${question.scoring_guide_text ? `<div class="small muted" style="margin-top:8px;">${escapeHtml(question.scoring_guide_text)}</div>` : ''}
        </details>
      ` : ''}
      ${renderQuestionRubric(question)}
      <div class="small muted" style="margin-top:10px;">Covered slides: ${escapeHtml((question.covered_slides || []).join(', ') || 'not specified')}</div>
      <div>${renderCitationButtons(question.citations || [])}</div>
    </div>
  `;
}

function renderPractice() {
  const workspace = state.activeWorkspace;
  const preferences = workspace?.practice_preferences || {};
  const topicInput = document.getElementById('practice-topic-text');
  const countInput = document.getElementById('practice-count');
  const modeInput = document.getElementById('practice-mode');
  const difficultyInput = document.getElementById('practice-difficulty');
  const coverageInput = document.getElementById('practice-coverage');
  const groundingInput = document.getElementById('practice-grounding-mode');
  const answerKeyInput = document.getElementById('practice-answer-key');
  const rubricsInput = document.getElementById('practice-rubrics');
  if (topicInput) topicInput.value = preferences.topic_text || '';
  if (countInput) countInput.value = preferences.question_count || 6;
  if (modeInput) modeInput.value = preferences.generation_mode || 'mixed';
  if (difficultyInput) difficultyInput.value = preferences.difficulty_profile || 'harder';
  if (coverageInput) coverageInput.value = preferences.coverage_mode || 'balanced';
  if (groundingInput) groundingInput.value = workspace?.grounding_mode || 'strict_lecture_only';
  if (answerKeyInput) answerKeyInput.checked = preferences.include_answer_key === true;
  if (rubricsInput) rubricsInput.checked = preferences.include_rubrics !== false;

  renderPracticeClarification();

  const output = document.getElementById('practice-output');
  const reviseButton = document.getElementById('revise-practice');
  const practice = workspace?.active_practice_set;
  if (reviseButton) reviseButton.classList.toggle('hidden', !practice);
  if (!output) return;
  if (!practice) {
    output.innerHTML = '<div class="muted">No practice test yet. Confirm a format and generate a grounded draft.</div>';
    return;
  }
  const coverage = practice.coverage_report || {};
  output.innerHTML = `
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:flex-start;gap:12px;">
        <div>
          <h3 style="margin:0 0 6px;">Active practice test</h3>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            ${badge(practice.generation_mode || 'mixed')}
            ${badge(`${practice.questions?.length || 0} questions`)}
            ${badge(`${practice.estimated_duration_minutes || '?'} min`)}
          </div>
        </div>
        <div class="small muted">Created ${escapeHtml(formatDate(practice.created_at))}</div>
      </div>
      ${practice.topic_text ? `<div class="small" style="margin-top:10px;"><strong>Topic focus:</strong> ${escapeHtml(practice.topic_text)}</div>` : ''}
      <div class="small muted" style="margin-top:8px;">${escapeHtml(coverage.notes || 'Coverage notes are not available yet.')}</div>
      ${coverage.uncited_or_skipped_slides?.length ? `<div class="warning small" style="margin-top:8px;">Areas still marked as weak or uncovered: slides ${escapeHtml(coverage.uncited_or_skipped_slides.join(', '))}</div>` : ''}
    </div>
    ${renderHumanLoopSummary(practice.human_loop_summary)}
    ${(practice.questions || []).map((question, index) => renderPracticeQuestion(question, index)).join('')}
  `;
  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = JSON.parse(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
}

function renderHistory() {
  const output = document.getElementById('history-output');
  const history = (state.activeWorkspace?.history || []).filter((entry) => entry.artifact_type !== 'study_plan');
  if (!output) return;
  if (!history.length) {
    output.innerHTML = '<div class="muted">No history yet.</div>';
    return;
  }
  output.innerHTML = history
    .slice()
    .sort((left, right) => (left.created_at > right.created_at ? -1 : 1))
    .map((entry) => `
      <div class="card">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:12px;">
          <div>
            <div><strong>${escapeHtml(entry.artifact_type)}</strong> | ${escapeHtml(entry.title || '')}</div>
            <div class="small muted">${escapeHtml(formatDate(entry.created_at))} | parent ${escapeHtml(entry.parent_artifact_id || '-')} | ${entry.active ? 'current' : 'prior'}</div>
          </div>
          <button type="button" class="secondary history-open" data-artifact-type="${escapeHtml(entry.artifact_type)}" data-artifact-id="${escapeHtml(entry.artifact_id)}">Open</button>
        </div>
      </div>
    `)
    .join('');
  output.querySelectorAll('.history-open').forEach((button) => {
    button.addEventListener('click', async () => {
      const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/history/${button.dataset.artifactType}/${button.dataset.artifactId}/activate`, {
        method: 'POST',
      });
      state.activeWorkspace = payload.workspace;
      renderWorkspace();
      if (button.dataset.artifactType === 'practice_set') setTab('practice');
      if (button.dataset.artifactType === 'conversation') setTab('ask');
    });
  });
}

function renderJobs() {
  const container = document.getElementById('material-jobs');
  const jobs = state.activeWorkspace?.jobs || [];
  if (!container) return;
  if (!jobs.length) {
    container.innerHTML = '';
    return;
  }
  const activeStatuses = new Set(['queued', 'running', 'submitted', 'waiting_for_service', 'needs_user_input']);
  const currentJobs = jobs.filter((job) => activeStatuses.has(job.status) || (job.status === 'failed' && !job.finalized));
  container.innerHTML = currentJobs.map((job) => `
    <div class="card">
      <div><strong>${escapeHtml(job.operation)}</strong></div>
      <div class="small muted">${escapeHtml(job.stage || '')} | ${escapeHtml(job.status || '')} | progress ${escapeHtml(String(job.progress || 0))}%</div>
      <div class="small">${escapeHtml(job.message || '')}</div>
      ${job.error?.message ? `<div class="error small">${escapeHtml(job.error.message)}</div>` : ''}
      ${job.user_action?.prompt ? `<div class="warning small">Needs input: ${escapeHtml(job.user_action.prompt)}</div>` : ''}
    </div>
  `).join('');
}

function renderWorkspace() {
  const emptyState = document.getElementById('workspace-empty-state');
  const view = document.getElementById('workspace-view');
  if (!state.activeWorkspace) {
    emptyState?.classList.remove('hidden');
    view?.classList.add('hidden');
    return;
  }
  emptyState?.classList.add('hidden');
  view?.classList.remove('hidden');
  document.getElementById('workspace-title').textContent = state.activeWorkspace.display_name;
  document.getElementById('workspace-meta').textContent = `Opened ${formatDate(state.activeWorkspace.last_opened_at)} | ${state.activeWorkspace.grounding_mode}`;
  renderMaterials();
  renderJobs();
  renderPractice();
  renderChat();
  renderHistory();
  applyServiceGating();
}

async function pollJob(jobId) {
  if (!state.activeWorkspace) return null;
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
          form.append('title', queue.length === 1 ? title : '');
          form.append('role', role);
          form.append('file', file);
          try {
            const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/materials/import`, { method: 'POST', body: form });
            const finishedJob = await pollJob(payload.job.job_id);
            const deduped = Boolean(finishedJob?.context?.deduplication?.skipped);
            if (item) item.status = deduped ? 'skipped' : 'done';
          } catch (error) {
            if (item) item.status = 'failed';
            renderMaterialUploadStatus(uploadItems, true);
            throw error;
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

  document.getElementById('practice-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;
    const request = {
      topic_text: document.getElementById('practice-topic-text').value.trim(),
      question_count: optionalPositiveIntegerValue('practice-count') || 6,
      generation_mode: document.getElementById('practice-mode').value,
      difficulty_profile: document.getElementById('practice-difficulty').value,
      coverage_mode: document.getElementById('practice-coverage').value,
      grounding_mode: document.getElementById('practice-grounding-mode').value,
      include_answer_key: document.getElementById('practice-answer-key').checked,
      include_rubrics: document.getElementById('practice-rubrics').checked,
    };
    const confirmation = [
      'Generate this grounded practice test?',
      `Topic: ${request.topic_text || 'all ready lecture materials'}`,
      `Format: ${request.generation_mode}`,
      `Question count: ${request.question_count}`,
      `Difficulty: ${request.difficulty_profile}`,
      `Coverage: ${request.coverage_mode}`,
      `Grounding: ${request.grounding_mode}`,
      `Answer key: ${request.include_answer_key ? 'yes' : 'no'}`,
      `Rubrics: ${request.include_rubrics ? 'yes' : 'no'}`,
    ].join('\n');
    if (!window.confirm(confirmation)) return;
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    await pollJob(payload.job.job_id);
    await loadWorkspaces();
  });

  document.getElementById('revise-practice').addEventListener('click', async () => {
    const practice = state.activeWorkspace?.active_practice_set;
    if (!practice) return;
    const targetQuestionIds = Array.from(document.querySelectorAll('.practice-target-toggle:checked'))
      .map((input) => input.dataset.questionId)
      .filter(Boolean);
    const lockedQuestionIds = Array.from(document.querySelectorAll('.practice-lock-toggle:checked'))
      .map((input) => input.dataset.questionId)
      .filter(Boolean);
    const overlapping = targetQuestionIds.filter((questionId) => lockedQuestionIds.includes(questionId));
    if (overlapping.length) {
      showTransientMessage('A question cannot be both locked and marked for regeneration.', 'warning');
      return;
    }
    const feedbackNote = window.prompt(
      'Optional note for this revision.',
      targetQuestionIds.length
        ? 'Regenerate the selected questions with less redundancy and stronger lecture grounding.'
        : 'Create a cleaner variant while preserving the locked questions and overall coverage.'
    ) || '';
    const confirmation = [
      'Revise this practice test?',
      `Regenerate: ${targetQuestionIds.length} question(s)`,
      `Lock: ${lockedQuestionIds.length} question(s)`,
      'Maintain coverage: yes',
    ].join('\n');
    if (!window.confirm(confirmation)) return;
    const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/${practice.practice_set_id}/revise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_question_ids: targetQuestionIds,
        locked_question_ids: lockedQuestionIds,
        feedback_note: feedbackNote,
        maintain_coverage: true,
        action: targetQuestionIds.length ? 'revise_selected' : 'create_variant',
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
