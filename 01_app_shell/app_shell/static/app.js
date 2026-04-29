const state = {
  status: null,
  workspaces: [],
  activeWorkspace: null,
  citationList: [],
  citationIndex: 0,
  currentTab: 'practice',
  ui: {
    practiceDrafts: {},
    chatDrafts: {},
    review: null,
  },
};

const SUPPORT_STATUS_LABELS = {
  slide_grounded: 'Grounded in your materials',
  inferred_from_slides: 'Grounded across your materials',
  annotation_grounded: 'Grounded in your notes',
  grounded: 'Grounded in your materials',
  partially_grounded: 'Partially grounded',
  insufficient_evidence: 'Insufficient lecture evidence',
  external_supplement: 'External supplement',
  supplemental_note: 'Supplemental note',
  not_grounded: 'Not grounded',
  ungrounded: 'Not grounded',
  source_preview: 'Source preview',
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

function humanize(value) {
  return String(value ?? '').replace(/_/g, ' ').trim();
}

function titleCaseLabel(value) {
  return humanize(value).replace(/\b\w/g, (match) => match.toUpperCase());
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
  const className = cls ? `badge ${cls}` : 'badge';
  return `<span class="${className}">${escapeHtml(text)}</span>`;
}

function encodeData(value) {
  return encodeURIComponent(JSON.stringify(value ?? []));
}

function decodeData(value) {
  try {
    return JSON.parse(decodeURIComponent(value || ''));
  } catch (error) {
    console.warn('Could not decode dataset payload', error);
    return [];
  }
}

function formatSupportStatus(status) {
  return SUPPORT_STATUS_LABELS[status] || titleCaseLabel(status || 'Unknown');
}

function supportStatusClass(status) {
  if (['insufficient_evidence', 'not_grounded', 'ungrounded'].includes(status)) return 'danger';
  if (['external_supplement', 'partially_grounded', 'supplemental_note'].includes(status)) return 'warn';
  return 'ok';
}

function formatFallbackReason(reason) {
  const labels = {
    gemini_not_configured: 'Gemini unavailable',
    llm_generation_failed: 'Gemini response failed',
    invalid_response_json: 'Gemini returned invalid structured output',
    invalid_response_text: 'Gemini returned invalid text output',
    invalid_response_json_text: 'Gemini returned invalid output',
    invalid_response_json_payload: 'Gemini returned invalid output',
    over_strict_insufficient_evidence: 'Gemini was too strict about lecture coverage',
    request_timeout: 'Gemini request timed out',
    request_exception: 'Gemini request failed',
    service_unavailable: 'Gemini temporarily unavailable',
    upstream_error: 'Gemini upstream error',
    authentication_failed: 'Gemini authentication failed',
    bad_request: 'Gemini request was rejected',
    empty_response: 'Gemini returned no answer',
    rate_limit_exceeded: 'Gemini rate limit hit',
    rate_limit_exhausted: 'All Gemini tiers were rate limited',
  };
  return labels[reason] || titleCaseLabel(reason || 'fallback');
}

function summarizeAnswerComposition(message) {
  const statuses = new Set((message.reply_sections || []).map((section) => section.support_status).filter(Boolean));
  if (statuses.has('insufficient_evidence')) {
    return 'This answer is explicitly saying the current materials do not support a confident lecture-grounded response.';
  }
  if (statuses.has('external_supplement') && (statuses.has('slide_grounded') || statuses.has('inferred_from_slides'))) {
    return 'This answer combines grounded lecture content with a separately labeled external supplement.';
  }
  if (statuses.has('external_supplement')) {
    return 'This answer is primarily a labeled external supplement, not a purely lecture-grounded reply.';
  }
  if (statuses.has('slide_grounded') || statuses.has('inferred_from_slides')) {
    return 'This answer is grounded in the current uploaded materials.';
  }
  return 'Review the section labels below before relying on this answer.';
}

function renderAnswerSource(message) {
  const source = message.answer_source;
  if (!source) return '';

  const isLlm = source.path === 'llm';
  const matchedCount = Number(source.matched_evidence_count || 0);
  const rateLimitedModels = source.rate_limited_models || [];
  const attemptedModels = source.attempted_models || [];
  const badges = [
    badge(isLlm ? 'LLM answer' : 'Deterministic fallback', isLlm ? 'ok' : 'warn'),
    badge(`${matchedCount} matched source${matchedCount === 1 ? '' : 's'}`, matchedCount ? 'ok' : 'warn'),
  ];

  if (source.model) badges.push(badge(source.model));
  if (source.reasoning_enabled) badges.push(badge('Reasoning on', 'ok'));
  if (source.evidence_match) {
    badges.push(badge(titleCaseLabel(source.evidence_match), source.evidence_match === 'strong_match' ? 'ok' : 'warn'));
  }
  if (!isLlm && source.fallback_reason) {
    badges.push(badge(formatFallbackReason(source.fallback_reason), 'warn'));
  }
  if (rateLimitedModels.length) {
    badges.push(badge(`Rate-limited earlier tier${rateLimitedModels.length === 1 ? '' : 's'}`, 'warn'));
  }

  let detail = isLlm
    ? `Primary answer generated with ${escapeHtml(source.model || 'Gemini')}.`
    : `Primary answer generated by the deterministic fallback instead of Gemini.`;
  if (rateLimitedModels.length) {
    detail += ` Earlier tier${rateLimitedModels.length === 1 ? '' : 's'} hit rate limits: ${escapeHtml(rateLimitedModels.join(', '))}.`;
  }
  if (!isLlm && source.fallback_reason) {
    detail += ` Reason: ${escapeHtml(formatFallbackReason(source.fallback_reason))}.`;
  }
  const technicalDetails = [];
  if (!isLlm && source.fallback_detail) {
    technicalDetails.push(`Technical detail: ${escapeHtml(source.fallback_detail)}`);
  }
  if (!isLlm && attemptedModels.length) {
    technicalDetails.push(`Tried models: ${escapeHtml(attemptedModels.join(', '))}`);
  }

  return `
    <div class="answer-source-card ${isLlm ? 'llm' : 'fallback'}">
      <div class="answer-source-header">
        <div>
          <div class="small muted">Answer source</div>
          <strong>${escapeHtml(isLlm ? 'Gemini-generated reply' : 'Deterministic fallback reply')}</strong>
        </div>
        <div class="answer-source-badges">
          ${badges.join('')}
        </div>
      </div>
      <div class="small">${detail}</div>
      ${technicalDetails.length ? `<div class="small muted">${technicalDetails.join(' ')}</div>` : ''}
      <div class="small muted">${escapeHtml(summarizeAnswerComposition(message))}</div>
    </div>
  `;
}

function showTransientMessage(text, type = 'muted') {
  const container = document.getElementById('global-toast-stack');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = text;
  container.append(toast);
  if (container.children.length > 5) {
    container.removeChild(container.firstElementChild);
  }
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

function handleError(error) {
  console.error(error);
  showTransientMessage(error.message || 'Something went wrong.', 'error');
  setGlobalActivity('');
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

function currentServices() {
  return state.activeWorkspace?.status?.services || state.status?.services || {};
}

function serviceAvailability() {
  const services = currentServices();
  return {
    contentAvailable: Boolean(services.content?.available),
    contentReady: services.content?.ready !== false,
    learningAvailable: Boolean(services.learning?.available),
    learningReady: Boolean(services.learning?.ready),
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

function groundedMaterials(workspace) {
  const materials = workspace?.materials || [];
  const preferences = workspace?.material_preferences || {};
  return materials.filter((material) => {
    const preference = preferences[material.material_id] || 'default';
    return material.processing_status === 'ready' && preference !== 'exclude' && material.role !== 'practice_template';
  });
}

function readyMaterialCount(workspace) {
  return groundedMaterials(workspace).length;
}

function allReadyMaterials(workspace) {
  return (workspace?.materials || []).filter((material) => material.processing_status === 'ready');
}

function countPreferences(workspace, value) {
  return Object.values(workspace?.material_preferences || {}).filter((preference) => preference === value).length;
}

function collectPracticeCitations(practice) {
  return (practice?.questions || []).flatMap((question) => question.citations || []);
}

function collectConversationCitations(conversation) {
  return (conversation?.messages || []).flatMap((message) =>
    (message.reply_sections || []).flatMap((section) => section.citations || [])
  );
}

function uniqueCitationCount(workspace) {
  const citations = [
    ...collectPracticeCitations(workspace?.active_practice_set),
    ...collectConversationCitations(workspace?.active_conversation),
  ];
  const unique = new Set(citations.map((citation) => citation.citation_id || `${citation.material_id}|${citation.slide_id}|${citation.slide_number}`));
  return unique.size;
}

function defaultPracticeDraft(workspace) {
  const preferences = workspace?.practice_preferences || {};
  return {
    topic_text: preferences.topic_text || '',
    question_count: preferences.question_count || 6,
    generation_mode: 'short_answer',
    difficulty_profile: preferences.difficulty_profile || 'harder',
    coverage_mode: preferences.coverage_mode || 'balanced',
    grounding_mode: workspace?.grounding_mode || 'lecture_with_fallback',
    include_answer_key: preferences.include_answer_key === true,
    include_rubrics: preferences.include_rubrics !== false,
  };
}

function ensurePracticeDraft(workspace) {
  if (!workspace) return defaultPracticeDraft(null);
  const workspaceId = workspace.workspace_id;
  if (!state.ui.practiceDrafts[workspaceId]) {
    state.ui.practiceDrafts[workspaceId] = defaultPracticeDraft(workspace);
  }
  return state.ui.practiceDrafts[workspaceId];
}

function setPracticeDraft(workspace, partial) {
  if (!workspace) return;
  const current = ensurePracticeDraft(workspace);
  state.ui.practiceDrafts[workspace.workspace_id] = {
    ...current,
    ...partial,
  };
}

function defaultChatDraft(workspace) {
  return {
    text: '',
    grounding_mode: workspace?.grounding_mode || 'lecture_with_fallback',
    response_style: 'standard',
  };
}

function ensureChatDraft(workspace) {
  if (!workspace) return defaultChatDraft(null);
  const workspaceId = workspace.workspace_id;
  if (!state.ui.chatDrafts[workspaceId]) {
    state.ui.chatDrafts[workspaceId] = defaultChatDraft(workspace);
  }
  return state.ui.chatDrafts[workspaceId];
}

function setChatDraft(workspace, partial) {
  if (!workspace) return;
  const current = ensureChatDraft(workspace);
  state.ui.chatDrafts[workspace.workspace_id] = {
    ...current,
    ...partial,
  };
}

function syncBodyLayout() {
  const viewer = document.querySelector('.source-viewer');
  const open = Boolean(viewer) && !viewer.classList.contains('hidden');
  document.body.classList.toggle('source-viewer-open', open);
}

function closeSourceViewer() {
  state.citationList = [];
  state.citationIndex = 0;
  document.querySelector('.source-viewer')?.classList.add('hidden');
  document.getElementById('source-view')?.classList.add('hidden');
  document.getElementById('source-empty')?.classList.remove('hidden');
  const counter = document.getElementById('source-counter');
  if (counter) counter.textContent = '';
  syncBodyLayout();
}

function setActiveWorkspace(workspace, { resetSource = false } = {}) {
  const previousWorkspaceId = state.activeWorkspace?.workspace_id;
  state.activeWorkspace = workspace;
  if (resetSource || previousWorkspaceId !== workspace?.workspace_id) {
    closeSourceViewer();
  }
  if (!workspace) {
    closeDecisionReview();
  } else {
    ensurePracticeDraft(workspace);
    ensureChatDraft(workspace);
  }
  renderWorkspace();
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
}

async function loadStatus() {
  state.status = await api('/api/status');
  renderSystemStatus();
}

function renderSystemStatus() {
  const container = document.getElementById('system-status');
  if (!container) return;
  if (!state.status) {
    container.innerHTML = '<div class="card small muted">Loading service status...</div>';
    return;
  }

  const content = state.status.services.content;
  const learning = state.status.services.learning;
  const mode = state.status.effective_mode;
  const degraded = !content.available || !learning.available;
  const learningFallbackOnly = learning.available && learning.ready === false;

  container.innerHTML = `
    <div class="card runtime-card">
      <div class="eyebrow">Runtime Health</div>
      <strong>The shell stays on the real service path.</strong>
      <div class="runtime-pill-row">
        ${badge(titleCaseLabel(mode))}
        ${badge(content.available ? 'Evidence viewer available' : 'Evidence viewer paused', content.available ? 'ok' : 'danger')}
        ${badge(learning.available ? 'Ask and practice available' : 'Ask and practice paused', learning.available ? 'ok' : 'danger')}
        ${learningFallbackOnly ? badge('Primary model degraded', 'warn') : ''}
      </div>
      <div class="small muted">${degraded
        ? 'Unavailable services disable dependent actions in the UI, so the app stays usable without pretending the full workflow is healthy.'
        : learningFallbackOnly
          ? 'The learning service is online but its primary model path is degraded. Ask and practice stay on the real service path instead of switching to mock data.'
          : 'Live service state stays visible, so you can see exactly what is available without hidden fallbacks.'}</div>
      ${degraded
        ? '<div class="service-note small warning" style="margin-top:12px;">Unavailable features stay disabled instead of failing silently.</div>'
        : ''}
    </div>
  `;

  applyServiceGating();
}

function applyServiceGating() {
  const { contentAvailable, learningAvailable, learningReady } = serviceAvailability();
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
    !learningAvailable
      ? 'Learning service is unavailable. Practice generation and revision are disabled until it recovers.'
      : !learningReady
        ? 'Learning service is online, but its primary model path is degraded. Practice still runs through the real backend and may use the service fallback path.'
        : ''
  );
  setServiceNote(
    'chat-service-note',
    !learningAvailable
      ? 'Learning service is unavailable. Grounded chat is disabled until it recovers.'
      : !learningReady
        ? 'Learning service is online, but its primary model path is degraded. Chat still runs through the real backend and may use the service fallback path.'
        : ''
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
  if (state.activeWorkspace) {
    const stillExists = state.workspaces.some((workspace) => workspace.workspace_id === state.activeWorkspace.workspace_id);
    if (!stillExists) {
      setActiveWorkspace(null, { resetSource: true });
    }
  }
  renderWorkspaceList();
}

function renderWorkspaceList() {
  renderWorkspaceListInto('workspace-list');
  renderWorkspaceListInto('landing-workspace-list');
}

function renderWorkspaceListInto(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!state.workspaces.length) {
    container.innerHTML = '<div class="muted small">No workspaces yet.</div>';
    return;
  }

  container.innerHTML = state.workspaces.map((workspace) => {
    const active = state.activeWorkspace?.workspace_id === workspace.workspace_id;
    const hasProcessing = workspace.material_counts.processing > 0;
    const statusBadge = workspace.material_counts.ready
      ? badge(`${workspace.material_counts.ready} ready`, 'ok')
      : badge('No ready sources', 'warn');
    return `
      <div class="workspace-card${active ? ' active' : ''}">
        <div class="workspace-card-header">
          <div>
            <div class="workspace-card-name">${escapeHtml(workspace.display_name)}</div>
            <div class="workspace-card-meta small muted">Opened ${escapeHtml(formatDate(workspace.last_opened_at))}</div>
          </div>
          ${badge(titleCaseLabel(workspace.grounding_mode || 'lecture_with_fallback'))}
        </div>
        <div class="workspace-card-stats">
          ${statusBadge}
          ${hasProcessing ? badge(`${workspace.material_counts.processing} processing`, 'warn') : ''}
          ${badge(`${workspace.artifact_counts.practice_sets} practice`) }
          ${badge(`${workspace.artifact_counts.conversations} chats`) }
        </div>
        <div class="small muted">Each workspace keeps its own evidence library, grounded outputs, and audit trail.</div>
        <div class="workspace-card-actions">
          <button type="button" class="ws-open-btn" data-open-workspace="${workspace.workspace_id}">${active ? 'Active workspace' : 'Open workspace'}</button>
          <button type="button" class="secondary ws-action-btn" data-duplicate-workspace="${workspace.workspace_id}">Duplicate</button>
          <button type="button" class="secondary ws-action-btn" data-archive-workspace="${workspace.workspace_id}">Archive</button>
          <button type="button" class="danger ws-action-btn" data-delete-workspace="${workspace.workspace_id}">Delete</button>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('[data-open-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await openWorkspace(button.dataset.openWorkspace);
      } catch (error) {
        handleError(error);
      }
    });
  });

  container.querySelectorAll('[data-duplicate-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await api(`/api/workspaces/${button.dataset.duplicateWorkspace}/duplicate`, { method: 'POST' });
        showTransientMessage('Workspace duplicated.', 'success');
        await loadWorkspaces();
      } catch (error) {
        handleError(error);
      }
    });
  });

  container.querySelectorAll('[data-archive-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await api(`/api/workspaces/${button.dataset.archiveWorkspace}/archive`, { method: 'POST' });
        showTransientMessage('Workspace archived.', 'success');
        await loadWorkspaces();
      } catch (error) {
        handleError(error);
      }
    });
  });

  container.querySelectorAll('[data-delete-workspace]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!window.confirm('Delete this workspace?')) return;
      try {
        await api(`/api/workspaces/${button.dataset.deleteWorkspace}`, { method: 'DELETE' });
        if (state.activeWorkspace?.workspace_id === button.dataset.deleteWorkspace) {
          setActiveWorkspace(null, { resetSource: true });
        }
        showTransientMessage('Workspace deleted.', 'success');
        await loadWorkspaces();
      } catch (error) {
        handleError(error);
      }
    });
  });
}

async function openWorkspace(workspaceId) {
  const payload = await api(`/api/workspaces/${workspaceId}`);
  setActiveWorkspace(payload.workspace, { resetSource: true });
}

function setTab(tabName) {
  state.currentTab = tabName;
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `tab-${tabName}`);
  });
}

function materialPreferenceOptions(current) {
  return ['default', 'focus', 'exclude']
    .map((option) => `<option value="${option}" ${current === option ? 'selected' : ''}>${titleCaseLabel(option)}</option>`)
    .join('');
}

function renderWorkspaceOverview() {
  const workspace = state.activeWorkspace;
  if (!workspace) return;

  const groundedReady = readyMaterialCount(workspace);
  const totalReady = allReadyMaterials(workspace).length;
  const historyCount = (workspace.history || []).length;
  const citationCount = uniqueCitationCount(workspace);
  const pendingAction = latestPracticeUserAction();
  const focusCount = countPreferences(workspace, 'focus');
  const excludeCount = countPreferences(workspace, 'exclude');
  const jobs = (workspace.jobs || []).filter((job) => ['queued', 'running', 'submitted', 'waiting_for_service', 'needs_user_input'].includes(job.status));
  const pills = document.getElementById('workspace-health-pills');
  const overview = document.getElementById('workspace-overview');
  const loopMap = document.getElementById('workspace-loop-map');
  const oversightState = pendingAction ? 'Clarification waiting' : jobs.length ? 'Live job visible' : 'Ready';

  if (pills) {
    pills.innerHTML = `
      ${badge(`${groundedReady} ready source${groundedReady !== 1 ? 's' : ''}`, groundedReady ? 'ok' : 'warn')}
      ${badge(`${citationCount} visible citation${citationCount !== 1 ? 's' : ''}`, citationCount ? 'ok' : 'warn')}
      ${badge(oversightState, pendingAction ? 'warn' : 'ok')}
    `;
  }

  if (overview) {
    overview.innerHTML = `
      <div class="summary-card summary-card-primary">
        <div class="summary-kicker">Grounding Scope</div>
        <div class="summary-value">${groundedReady}<span class="small muted"> / ${totalReady || 0}</span></div>
        <div class="summary-body">Only ready, non-excluded lecture materials can ground chat and practice generation.</div>
        <div class="summary-detail">${focusCount} prioritized source${focusCount !== 1 ? 's' : ''}; ${excludeCount} excluded source${excludeCount !== 1 ? 's' : ''}.</div>
      </div>
      <div class="summary-card">
        <div class="summary-kicker">Evidence Visibility</div>
        <div class="summary-value">${citationCount}</div>
        <div class="summary-body">Citations remain attached to answers and questions so users can inspect the exact supporting slide or page.</div>
        <div class="summary-detail">${historyCount} stored artifact${historyCount !== 1 ? 's' : ''} keep the audit trail legible over time.</div>
      </div>
      <div class="summary-card">
        <div class="summary-kicker">Oversight State</div>
        <div class="summary-value">${pendingAction ? 'Hold' : jobs.length ? 'Live' : 'Ready'}</div>
        <div class="summary-body">${pendingAction ? 'The system is waiting for a narrower, user-approved request before generating.' : jobs.length ? 'Background work is active and visible instead of happening silently.' : 'The workspace is ready for grounded chat or approved practice generation.'}</div>
        <div class="summary-detail">Nothing should generate invisibly or rely on hidden evidence assumptions.</div>
      </div>
      <div class="summary-card summary-card-plain">
        <div class="summary-kicker">Design Promise</div>
        <div class="summary-body">Smooth use comes from fewer steps, but every important AI boundary stays explicit: evidence in scope, answer source, uncertainty, and history.</div>
      </div>
    `;
  }

  if (loopMap) {
    loopMap.innerHTML = `
      <div class="workflow-card">
        <div class="workflow-card-header">
          <div>
            <div class="eyebrow">Visible Workflow</div>
            <h3>One clear interaction pattern across the app</h3>
          </div>
          <div class="workflow-chip-row">
            ${badge('Frame inputs', 'ok')}
            ${badge('Approve or clarify', pendingAction ? 'warn' : 'ok')}
            ${badge('Inspect evidence', citationCount ? 'ok' : 'warn')}
            ${badge('Audit history', historyCount ? 'ok' : 'warn')}
          </div>
        </div>
        <div class="workflow-rail">
          <div class="workflow-step">
            <div class="workflow-step-number">1</div>
            <strong>Set the scope</strong>
            <span class="small muted">Materials, grounding mode, topic, and difficulty are explicit user choices.</span>
          </div>
          <div class="workflow-step">
            <div class="workflow-step-number">2</div>
            <strong>Review before action</strong>
            <span class="small muted">Ambiguous requests pause for clarification instead of silently guessing.</span>
          </div>
          <div class="workflow-step">
            <div class="workflow-step-number">3</div>
            <strong>See what the AI used</strong>
            <span class="small muted">Every answer keeps source labels, support status, and citation access visible.</span>
          </div>
          <div class="workflow-step">
            <div class="workflow-step-number">4</div>
            <strong>Keep the audit trail</strong>
            <span class="small muted">History preserves past artifacts so revision stays explainable instead of overwrite-heavy.</span>
          </div>
        </div>
      </div>
    `;
  }
}

function renderMaterials() {
  const list = document.getElementById('material-list');
  const workspace = state.activeWorkspace;
  const materials = workspace?.materials || [];
  if (!list) return;
  if (!materials.length) {
    list.innerHTML = '<div class="card muted">No materials in this workspace yet. Add lecture evidence before asking or generating grounded outputs.</div>';
    return;
  }

  list.innerHTML = materials.map((material) => {
    const preference = workspace.material_preferences?.[material.material_id] || 'default';
    const statusClass = material.processing_status === 'ready'
      ? 'ok'
      : material.processing_status === 'failed'
        ? 'danger'
        : 'warn';
    const preferenceLabel = preference === 'focus'
      ? 'Prioritized for grounding'
      : preference === 'exclude'
        ? 'Excluded from grounding'
        : 'Used normally';
    return `
      <div class="material-card">
        <div class="material-card-header">
          <div>
            <div><strong>${escapeHtml(material.title)}</strong></div>
            <div class="small muted">${escapeHtml(titleCaseLabel(material.role))} | ${escapeHtml(titleCaseLabel(material.kind))} | ${escapeHtml(String(material.page_count ?? 0))} page(s)</div>
            <div class="card-chip-row">
              ${badge(titleCaseLabel(material.processing_status), statusClass)}
              ${badge(preferenceLabel, preference === 'exclude' ? 'danger' : preference === 'focus' ? 'ok' : '')}
            </div>
          </div>
          <div class="material-actions">
            <button type="button" class="secondary open-material-source" data-material-id="${material.material_id}">Open source</button>
            <button type="button" class="danger delete-material" data-material-id="${material.material_id}">Delete</button>
          </div>
        </div>
        <div class="small muted">${escapeHtml(material.quality_summary?.notes || 'No quality note available.')}</div>
        <div class="row" style="margin-top:12px;">
          <label class="field-grow">
            Grounding preference
            <select class="material-preference" data-material-id="${material.material_id}">
              ${materialPreferenceOptions(preference)}
            </select>
          </label>
        </div>
      </div>
    `;
  }).join('');

  list.querySelectorAll('.material-preference').forEach((select) => {
    select.addEventListener('change', async () => {
      try {
        const payload = await api(`/api/workspaces/${workspace.workspace_id}/materials/${select.dataset.materialId}/preference`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preference: select.value }),
        });
        setActiveWorkspace(payload.workspace);
        if (payload.warning) showTransientMessage(payload.warning, 'warning');
      } catch (error) {
        handleError(error);
      }
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
      try {
        const payload = await api(`/api/workspaces/${workspace.workspace_id}/materials/${button.dataset.materialId}`, { method: 'DELETE' });
        setActiveWorkspace(payload.workspace);
        await loadWorkspaces();
        showTransientMessage('Material deleted.', 'success');
      } catch (error) {
        handleError(error);
      }
    });
  });
}

function renderCitationButtons(citations = []) {
  if (!citations.length) return '';
  return `<div class="citation-row">${
    citations.map((citation, index) => `
      <button type="button" class="secondary citation-button" data-citation-index="${index}" title="Open the cited source in the evidence viewer">
        Slide ${escapeHtml(String(citation.slide_number || '?'))}
      </button>
    `).join('')
  }</div>`;
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
  if (!panel || panel.classList.contains('hidden')) return;
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
  syncBodyLayout();
  scrollSourceViewerIntoView();
}

function renderSourceViewer(citation) {
  const { contentAvailable } = serviceAvailability();
  const preview = document.getElementById('source-preview');
  const openButton = document.getElementById('source-open');
  const message = document.getElementById('source-message');
  const counter = document.getElementById('source-counter');

  document.getElementById('source-meta').textContent = `${citation.material_title || citation.material_id || 'Source'} | slide ${citation.slide_number || '?'}`;
  document.getElementById('source-snippet').innerHTML = citation.snippet_text
    ? `<div class="source-snippet-label small muted">Cited passage</div><div class="source-snippet-text">${escapeHtml(citation.snippet_text)}</div><div class="small muted" style="margin-top:10px;">${escapeHtml(formatSupportStatus(citation.support_type))}</div>`
    : '<div class="source-snippet-label small muted">No snippet available.</div>';

  if (counter) {
    counter.textContent = state.citationList.length ? `Citation ${state.citationIndex + 1} of ${state.citationList.length}` : '';
  }

  if (preview) {
    if (citation.preview_url) {
      preview.src = citation.preview_url;
      preview.classList.remove('hidden');
    } else {
      preview.classList.add('hidden');
      preview.removeAttribute('src');
    }
  }

  if (message) {
    const visibilityMessage = !contentAvailable
      ? 'Preview unavailable because the content service is offline. Citation metadata is still shown.'
      : citation.preview_url
        ? 'Evidence preview loaded. Use Open source page for the full source.'
        : 'Preview unavailable. Citation metadata is still shown so you can still inspect the grounding trace.';
    message.textContent = visibilityMessage;
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

function renderAskPolicy() {
  const container = document.getElementById('ask-policy');
  const workspace = state.activeWorkspace;
  if (!container || !workspace) return;
  const draft = ensureChatDraft(workspace);
  const strictMode = draft.grounding_mode === 'strict_lecture_only';

  container.innerHTML = `
    <div class="audit-summary-card">
      <div class="eyebrow">Trust and transparency</div>
      <h3>Answer policy for this chat</h3>
      <div class="policy-grid">
        <div class="policy-card">
          <div class="small muted">Evidence boundary</div>
          <strong>${strictMode ? 'Lecture-only grounding' : 'Lecture first, fallback labeled'}</strong>
          <div class="small muted">${strictMode ? 'Weak matches should decline or ask for clarification instead of inventing an answer.' : 'If the system needs outside knowledge, it should mark that content as an external supplement rather than blending it invisibly into lecture evidence.'}</div>
        </div>
        <div class="policy-card">
          <div class="small muted">Interpretability</div>
          <strong>Every assistant section keeps visible citations</strong>
          <div class="small muted">Use the evidence viewer to inspect the exact page or slide behind a grounded answer.</div>
        </div>
        <div class="policy-card">
          <div class="small muted">LLM routing</div>
          <strong>Gemini first, deterministic fallback last</strong>
          <div class="small muted">The app now tries Gemini 3 Flash Preview first, then Gemini 2.5 Flash, then Gemini 2.5 Flash-Lite on rate limits before using the non-LLM fallback.</div>
        </div>
      </div>
    </div>
  `;
}

function renderChat() {
  const workspace = state.activeWorkspace;
  const output = document.getElementById('chat-output');
  if (!workspace || !output) return;

  const draft = ensureChatDraft(workspace);
  const grounding = document.getElementById('chat-grounding-mode');
  const responseStyle = document.getElementById('chat-response-style');
  const questionInput = document.getElementById('chat-question');
  if (grounding) grounding.value = draft.grounding_mode;
  if (responseStyle) responseStyle.value = ['standard', 'concise', 'step_by_step'].includes(draft.response_style) ? draft.response_style : 'standard';
  if (questionInput && questionInput.value !== draft.text) questionInput.value = draft.text;

  const conversation = workspace.active_conversation;
  if (!conversation) {
    output.innerHTML = '<div class="card muted">No active chat yet. Type a grounded question to create one automatically, or start a new empty chat thread first.</div>';
    return;
  }

  output.innerHTML = (conversation.messages || []).map((message) => {
    if (message.role === 'user') {
      return `
        <div class="message-item user">
          <div class="message-meta">
            <strong>You</strong>
            <div class="small muted">${escapeHtml(formatDate(message.created_at))}${message.pending ? ' | pending' : ''}</div>
          </div>
          <div class="message-text">${escapeHtml(message.text || '')}</div>
        </div>
      `;
    }

    const sections = (message.reply_sections || []).map((section, index) => `
      <div class="assistant-section" data-section-citations="${encodeData(section.citations || [])}">
        <div class="message-meta">
          <strong>${escapeHtml(section.heading || `Section ${index + 1}`)}</strong>
          ${badge(formatSupportStatus(section.support_status), supportStatusClass(section.support_status))}
        </div>
        <div class="message-text">${escapeHtml(section.text || '')}</div>
        ${renderCitationButtons(section.citations || [])}
      </div>
    `).join('');

    const clarifying = message.clarifying_question?.prompt
      ? `<div class="service-note warning small">Clarifying question: ${escapeHtml(message.clarifying_question.prompt)}</div>`
      : '';

    return `
      <div class="message-item assistant">
        <div class="message-meta">
          <strong>Assistant</strong>
          <div class="small muted">${escapeHtml(formatDate(message.created_at))}</div>
        </div>
        <div class="assistant-stack">
          ${renderAnswerSource(message)}
          ${sections || '<div class="small muted">No assistant sections returned.</div>'}
          ${clarifying}
        </div>
      </div>
    `;
  }).join('');

  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = decodeData(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });
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
    <div class="audit-summary-card">
      <div class="eyebrow">Human-in-the-loop summary</div>
      <h3>What shaped this draft</h3>
      ${usedInputs.length ? `
        <div class="plan-input-grid">
          ${usedInputs.map((item) => `
            <div class="plan-input-item">
              <div class="small muted">${escapeHtml(item.label || item.key || 'Input')}</div>
              <div>${escapeHtml(item.value || '')}</div>
            </div>
          `).join('')}
        </div>
      ` : '<div class="small muted">No explicit inputs were captured for this draft.</div>'}
      ${followUps.length ? `
        <div class="small muted" style="margin-top:12px;">Recommended next checks</div>
        <ul class="task-list">
          ${followUps.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
        </ul>
      ` : ''}
    </div>
  `;
}

function renderPracticePreflight() {
  const container = document.getElementById('practice-preflight');
  const workspace = state.activeWorkspace;
  if (!container || !workspace) return;

  const draft = ensurePracticeDraft(workspace);
  const readyCount = readyMaterialCount(workspace);
  const processingCount = (workspace.materials || []).filter((material) => material.processing_status !== 'ready').length;
  const pendingAction = latestPracticeUserAction();

  container.innerHTML = `
    <div class="preflight-card">
      <div class="eyebrow">Visible review step</div>
      <h3>What will be reviewed before generation</h3>
      <div class="review-grid">
        <div class="review-item">
          <div class="small muted">Topic</div>
          <div>${escapeHtml(draft.topic_text || 'All ready lecture materials')}</div>
        </div>
        <div class="review-item">
          <div class="small muted">Question count</div>
          <div>${escapeHtml(String(draft.question_count))} questions</div>
        </div>
        <div class="review-item">
          <div class="small muted">Difficulty and coverage</div>
          <div>${escapeHtml(titleCaseLabel(draft.difficulty_profile))} | ${escapeHtml(titleCaseLabel(draft.coverage_mode))}</div>
        </div>
        <div class="review-item">
          <div class="small muted">Grounding and outputs</div>
          <div>${escapeHtml(titleCaseLabel(draft.grounding_mode))} | ${draft.include_answer_key ? 'answer key' : 'no answer key'} | ${draft.include_rubrics ? 'rubrics' : 'no rubrics'}</div>
        </div>
      </div>
      <div class="small muted" style="margin-top:12px;">
        Clicking <strong>Review request</strong> opens an approval step. The system should not silently generate a test from an ambiguous or weakly grounded request.
      </div>
      <div class="card-chip-row" style="margin-top:12px;">
        ${readyCount ? badge(`${readyCount} grounded source${readyCount !== 1 ? 's' : ''} ready`, 'ok') : badge('No ready grounded sources', 'warn')}
        ${processingCount ? badge(`${processingCount} source${processingCount !== 1 ? 's' : ''} still processing`, 'warn') : ''}
        ${pendingAction ? badge('Clarification requested', 'warn') : badge('Approval required before generation', 'ok')}
      </div>
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
    <div class="service-note">
      <div class="eyebrow">Clarification instead of guessing</div>
      <h3>Generation paused for a narrower request</h3>
      <div class="warning">${escapeHtml(pendingAction.user_action.prompt || '')}</div>
      ${options.length ? `
        <div class="small muted" style="margin-top:12px;">Suggested grounded topics from your uploaded evidence</div>
        <div class="citation-row" style="margin-top:8px;">
          ${options.map((option) => `
            <button type="button" class="secondary clarification-option" data-topic-option="${escapeHtml(option)}">${escapeHtml(option)}</button>
          `).join('')}
        </div>
      ` : ''}
      <div class="small muted" style="margin-top:12px;">Choose a narrower topic or edit the form, then review the request again. Weak coverage stays visible instead of being presented as fully grounded.</div>
    </div>
  `;

  container.querySelectorAll('.clarification-option').forEach((button) => {
    button.addEventListener('click', () => {
      const topic = button.dataset.topicOption || '';
      setPracticeDraft(state.activeWorkspace, { topic_text: topic });
      renderPractice();
      showTransientMessage('Topic focus updated. Review the request and generate again.', 'success');
    });
  });
}

function renderQuestionChoices(question) {
  const choices = question.answer_choices || [];
  if (!choices.length) return '';
  return `
    <div class="question-stack">
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
    <details>
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
    <div class="question-card" data-section-citations="${encodeData(question.citations || [])}">
      <div class="question-header">
        <div>
          <div><strong>Question ${index + 1}</strong></div>
          <div class="small muted">${escapeHtml(titleCaseLabel(question.difficulty || 'mixed'))} | ${escapeHtml(String(question.estimated_minutes || '?'))} min</div>
        </div>
        <div class="question-controls">
          <label class="inline-toggle">
            <input type="checkbox" class="practice-target-toggle" data-question-id="${escapeHtml(question.question_id)}" />
            <span>Regenerate</span>
          </label>
          <label class="inline-toggle">
            <input type="checkbox" class="practice-lock-toggle" data-question-id="${escapeHtml(question.question_id)}" />
            <span>Lock</span>
          </label>
        </div>
      </div>
      <div class="question-stack">
        <div class="message-text">${escapeHtml(question.stem || '')}</div>
        ${renderQuestionChoices(question)}
        ${question.expected_answer ? `
          <details>
            <summary>Answer key</summary>
            <div class="message-text small" style="margin-top:8px;">${escapeHtml(question.expected_answer)}</div>
            ${question.scoring_guide_text ? `<div class="small muted" style="margin-top:8px;">${escapeHtml(question.scoring_guide_text)}</div>` : ''}
          </details>
        ` : ''}
        ${renderQuestionRubric(question)}
        <div class="small muted">Covered slides: ${escapeHtml((question.covered_slides || []).join(', ') || 'not specified')}</div>
        ${renderCitationButtons(question.citations || [])}
      </div>
    </div>
  `;
}

function updatePracticeSelectionSummary() {
  const summary = document.getElementById('practice-selection-summary');
  if (!summary || !state.activeWorkspace?.active_practice_set) return;
  const targetQuestionIds = Array.from(document.querySelectorAll('.practice-target-toggle:checked'))
    .map((input) => input.dataset.questionId)
    .filter(Boolean);
  const lockedQuestionIds = Array.from(document.querySelectorAll('.practice-lock-toggle:checked'))
    .map((input) => input.dataset.questionId)
    .filter(Boolean);
  const overlapping = targetQuestionIds.filter((questionId) => lockedQuestionIds.includes(questionId));

  summary.innerHTML = `
    <div class="review-item">
      <div class="small muted">Questions to regenerate</div>
      <div>${escapeHtml(String(targetQuestionIds.length))}</div>
    </div>
    <div class="review-item">
      <div class="small muted">Questions to lock</div>
      <div>${escapeHtml(String(lockedQuestionIds.length))}</div>
    </div>
    <div class="review-item">
      <div class="small muted">Coverage rule</div>
      <div>Maintain coverage</div>
    </div>
    <div class="review-item">
      <div class="small muted">Conflict check</div>
      <div>${overlapping.length ? `${overlapping.length} overlap${overlapping.length !== 1 ? 's' : ''}` : 'No conflicts'}</div>
    </div>
  `;
}

function renderPractice() {
  const workspace = state.activeWorkspace;
  const output = document.getElementById('practice-output');
  const reviseButton = document.getElementById('revise-practice');
  if (!workspace || !output || !reviseButton) return;

  const draft = ensurePracticeDraft(workspace);
  const topicInput = document.getElementById('practice-topic-text');
  const countInput = document.getElementById('practice-count');
  const difficultyInput = document.getElementById('practice-difficulty');
  const coverageInput = document.getElementById('practice-coverage');
  const groundingInput = document.getElementById('practice-grounding-mode');
  const answerKeyInput = document.getElementById('practice-answer-key');
  const rubricsInput = document.getElementById('practice-rubrics');
  if (topicInput && topicInput.value !== draft.topic_text) topicInput.value = draft.topic_text || '';
  if (countInput && String(countInput.value) !== String(draft.question_count)) countInput.value = draft.question_count || 6;
  if (difficultyInput) difficultyInput.value = draft.difficulty_profile || 'harder';
  if (coverageInput) coverageInput.value = draft.coverage_mode || 'balanced';
  if (groundingInput) groundingInput.value = draft.grounding_mode || workspace.grounding_mode || 'lecture_with_fallback';
  if (answerKeyInput) answerKeyInput.checked = draft.include_answer_key === true;
  if (rubricsInput) rubricsInput.checked = draft.include_rubrics !== false;

  renderPracticePreflight();
  renderPracticeClarification();

  const practice = workspace.active_practice_set;
  reviseButton.classList.toggle('hidden', !practice);
  if (!practice) {
    output.innerHTML = '<div class="card muted">No practice test yet. Review the request, approve it, and generate a grounded draft.</div>';
    return;
  }

  const coverage = practice.coverage_report || {};
  output.innerHTML = `
    <div class="audit-summary-card">
      <div class="panel-subheading">
        <div>
          <div class="eyebrow">Current practice artifact</div>
          <h3>Active grounded practice test</h3>
        </div>
        <div class="card-chip-row">
          ${badge(`${practice.questions?.length || 0} questions`)}
          ${badge(`${practice.estimated_duration_minutes || '?'} min`)}
        </div>
      </div>
      ${practice.topic_text ? `<div class="small"><strong>Topic focus:</strong> ${escapeHtml(practice.topic_text)}</div>` : ''}
      <div class="small muted" style="margin-top:8px;">${escapeHtml(coverage.notes || 'Coverage notes are not available yet.')}</div>
    </div>
    ${renderHumanLoopSummary(practice.human_loop_summary)}
    <div class="audit-summary-card">
      <div class="eyebrow">Selective revision</div>
      <h3>Keep the strong questions and target the weak ones</h3>
      <div id="practice-selection-summary" class="review-grid"></div>
      <div class="small muted" style="margin-top:12px;">Use the checkboxes below, then choose <strong>Review revision</strong>. The old practice set stays in history so you can audit what changed.</div>
    </div>
    ${(practice.questions || []).map((question, index) => renderPracticeQuestion(question, index)).join('')}
  `;

  output.querySelectorAll('[data-section-citations]').forEach((section) => {
    const citations = decodeData(section.dataset.sectionCitations);
    wireCitationButtons(section, citations);
  });

  output.querySelectorAll('.practice-target-toggle, .practice-lock-toggle').forEach((input) => {
    input.addEventListener('change', updatePracticeSelectionSummary);
  });
  updatePracticeSelectionSummary();
}

function renderHistorySummary() {
  const container = document.getElementById('history-summary');
  const workspace = state.activeWorkspace;
  if (!container || !workspace) return;
  const history = workspace.history || [];
  const currentPractice = history.filter((entry) => entry.artifact_type === 'practice_set' && entry.active).length;
  const currentConversation = history.filter((entry) => entry.artifact_type === 'conversation' && entry.active).length;

  container.innerHTML = `
    <div class="audit-summary-card">
      <div class="eyebrow">Auditability</div>
      <h3>Why history matters in this design</h3>
      <div class="policy-grid">
        <div class="policy-card">
          <div class="small muted">Stored artifacts</div>
          <strong>${history.length}</strong>
          <div class="small muted">Prior practice sets and conversations remain accessible instead of being overwritten invisibly.</div>
        </div>
        <div class="policy-card">
          <div class="small muted">Current active artifacts</div>
          <strong>${currentPractice + currentConversation}</strong>
          <div class="small muted">Only one practice set and one conversation stay active at a time, so the current state stays legible.</div>
        </div>
      </div>
    </div>
  `;
}

function renderHistory() {
  const output = document.getElementById('history-output');
  const history = state.activeWorkspace?.history || [];
  if (!output) return;
  if (!history.length) {
    output.innerHTML = '<div class="card muted">No history yet. Generated artifacts and chat threads will appear here for audit and replay.</div>';
    return;
  }

  output.innerHTML = history
    .slice()
    .sort((left, right) => (left.created_at > right.created_at ? -1 : 1))
    .map((entry) => `
      <div class="history-card">
        <div class="history-card-header">
          <div>
            <div><strong>${escapeHtml(titleCaseLabel(entry.artifact_type))}</strong> | ${escapeHtml(entry.title || '')}</div>
            <div class="small muted">${escapeHtml(formatDate(entry.created_at))} | parent ${escapeHtml(entry.parent_artifact_id || '-')}</div>
          </div>
          <div class="card-chip-row">
            ${badge(entry.active ? 'Current' : 'Prior', entry.active ? 'ok' : '')}
            <button type="button" class="secondary history-open" data-artifact-type="${escapeHtml(entry.artifact_type)}" data-artifact-id="${escapeHtml(entry.artifact_id)}">Open</button>
          </div>
        </div>
      </div>
    `)
    .join('');

  output.querySelectorAll('.history-open').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/history/${button.dataset.artifactType}/${button.dataset.artifactId}/activate`, {
          method: 'POST',
        });
        setActiveWorkspace(payload.workspace);
        if (button.dataset.artifactType === 'practice_set') setTab('practice');
        if (button.dataset.artifactType === 'conversation') setTab('ask');
      } catch (error) {
        handleError(error);
      }
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
  if (!currentJobs.length) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = currentJobs.map((job) => `
    <div class="job-card">
      <div class="message-meta">
        <strong>${escapeHtml(titleCaseLabel(job.operation))}</strong>
        ${badge(titleCaseLabel(job.status || 'unknown'), job.status === 'failed' ? 'danger' : job.status === 'needs_user_input' ? 'warn' : 'ok')}
      </div>
      <div class="small muted">${escapeHtml(job.stage || 'Waiting')} | progress ${escapeHtml(String(job.progress || 0))}%</div>
      <div class="small">${escapeHtml(job.message || '')}</div>
      ${job.error?.message ? `<div class="error small">${escapeHtml(job.error.message)}</div>` : ''}
      ${job.user_action?.prompt ? `<div class="service-note warning small">Needs input: ${escapeHtml(job.user_action.prompt)}</div>` : ''}
    </div>
  `).join('');
}

function renderWorkspace() {
  const emptyState = document.getElementById('workspace-empty-state');
  const view = document.getElementById('workspace-view');
  if (!state.activeWorkspace) {
    document.body.classList.remove('has-workspace');
    emptyState?.classList.remove('hidden');
    view?.classList.add('hidden');
    closeSourceViewer();
    return;
  }

  document.body.classList.add('has-workspace');
  emptyState?.classList.add('hidden');
  view?.classList.remove('hidden');
  document.getElementById('workspace-title').textContent = state.activeWorkspace.display_name;
  document.getElementById('workspace-meta').textContent = `Opened ${formatDate(state.activeWorkspace.last_opened_at)} | ${titleCaseLabel(state.activeWorkspace.grounding_mode)}`;
  renderWorkspaceOverview();
  renderMaterials();
  renderJobs();
  renderPractice();
  renderAskPolicy();
  renderChat();
  renderHistorySummary();
  renderHistory();
  applyServiceGating();
  setTab(state.currentTab);
}

async function refreshActiveWorkspace() {
  if (!state.activeWorkspace) return null;
  const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}`);
  setActiveWorkspace(payload.workspace);
  return payload.workspace;
}

async function pollJob(jobId) {
  if (!state.activeWorkspace) return null;
  const workspaceId = state.activeWorkspace.workspace_id;
  while (true) {
    const payload = await api(`/api/jobs/${jobId}?workspace_id=${workspaceId}`);
    const job = payload.job;
    setGlobalActivity(job.message || `${titleCaseLabel(job.operation || 'operation')} is running`, true);
    await refreshActiveWorkspace();
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
  const chatDraft = ensureChatDraft(state.activeWorkspace);
  const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'Workspace Q&A', grounding_mode: chatDraft.grounding_mode }),
  });
  await refreshActiveWorkspace();
  return payload.conversation;
}

function buildPracticeRequestFromForm() {
  return {
    topic_text: document.getElementById('practice-topic-text').value.trim(),
    question_count: optionalPositiveIntegerValue('practice-count') || 6,
    generation_mode: 'short_answer',
    difficulty_profile: document.getElementById('practice-difficulty').value,
    coverage_mode: document.getElementById('practice-coverage').value,
    grounding_mode: document.getElementById('practice-grounding-mode').value,
    include_answer_key: document.getElementById('practice-answer-key').checked,
    include_rubrics: document.getElementById('practice-rubrics').checked,
  };
}

function capturePracticeDraft() {
  if (!state.activeWorkspace) return;
  setPracticeDraft(state.activeWorkspace, buildPracticeRequestFromForm());
  renderPracticePreflight();
}

function captureChatDraft() {
  if (!state.activeWorkspace) return;
  setChatDraft(state.activeWorkspace, {
    text: document.getElementById('chat-question').value,
    grounding_mode: document.getElementById('chat-grounding-mode').value,
    response_style: document.getElementById('chat-response-style').value,
  });
  renderAskPolicy();
}

function openDecisionReview(review) {
  state.ui.review = review;
  renderDecisionReview();
}

function closeDecisionReview() {
  state.ui.review = null;
  renderDecisionReview();
}

function renderDecisionReview() {
  const overlay = document.getElementById('decision-review');
  const review = state.ui.review;
  if (!overlay) return;
  if (!review) {
    overlay.classList.add('hidden');
    overlay.setAttribute('aria-hidden', 'true');
    return;
  }

  overlay.classList.remove('hidden');
  overlay.setAttribute('aria-hidden', 'false');
  document.getElementById('decision-review-kicker').textContent = review.kicker || '';
  document.getElementById('decision-review-title').textContent = review.title || '';
  document.getElementById('decision-review-description').textContent = review.description || '';
  document.getElementById('decision-review-body').innerHTML = review.sections?.length
    ? `<div class="review-grid">${
        review.sections.map((section) => `
          <div class="review-item">
            <div class="small muted">${escapeHtml(section.label)}</div>
            <div>${escapeHtml(section.value)}</div>
          </div>
        `).join('')
      }</div>`
    : '<div class="small muted">No review details available.</div>';
  document.getElementById('decision-review-footnote').textContent = review.footnote || '';
  document.getElementById('decision-review-confirm').textContent = review.confirmLabel || 'Approve and continue';

  const noteWrap = document.getElementById('decision-review-note-wrap');
  const noteLabel = document.getElementById('decision-review-note-label');
  const noteInput = document.getElementById('decision-review-note');
  if (review.noteField) {
    noteWrap.classList.remove('hidden');
    noteLabel.textContent = review.noteField.label;
    noteInput.placeholder = review.noteField.placeholder || '';
    noteInput.value = review.noteField.defaultValue || '';
  } else {
    noteWrap.classList.add('hidden');
    noteInput.value = '';
    noteInput.placeholder = '';
  }
}

function buildPracticeReviewSections(request) {
  return [
    { label: 'Topic', value: request.topic_text || 'All ready lecture materials' },
    { label: 'Question count', value: String(request.question_count) },
    { label: 'Difficulty', value: titleCaseLabel(request.difficulty_profile) },
    { label: 'Coverage', value: titleCaseLabel(request.coverage_mode) },
    { label: 'Grounding mode', value: titleCaseLabel(request.grounding_mode) },
    { label: 'Answer key', value: request.include_answer_key ? 'Included' : 'Not included' },
    { label: 'Rubrics', value: request.include_rubrics ? 'Included' : 'Not included' },
  ];
}

function buildRevisionSelections() {
  const targetQuestionIds = Array.from(document.querySelectorAll('.practice-target-toggle:checked'))
    .map((input) => input.dataset.questionId)
    .filter(Boolean);
  const lockedQuestionIds = Array.from(document.querySelectorAll('.practice-lock-toggle:checked'))
    .map((input) => input.dataset.questionId)
    .filter(Boolean);
  return { targetQuestionIds, lockedQuestionIds };
}

function bindEvents() {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => setTab(button.dataset.tab));
  });

  document.getElementById('landing-new-workspace')?.addEventListener('click', () => {
    document.getElementById('landing-create-form')?.classList.remove('hidden');
    document.getElementById('workspace-name')?.focus();
  });

  document.getElementById('landing-cancel-create')?.addEventListener('click', () => {
    document.getElementById('landing-create-form')?.classList.add('hidden');
    document.getElementById('workspace-name').value = '';
  });

  document.getElementById('create-workspace-form')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const displayName = document.getElementById('workspace-name').value.trim();
      const payload = await api('/api/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: displayName }),
      });
      document.getElementById('workspace-name').value = '';
      document.getElementById('landing-create-form')?.classList.add('hidden');
      await loadWorkspaces();
      setActiveWorkspace(payload.workspace, { resetSource: true });
      showTransientMessage('Workspace created.', 'success');
    } catch (error) {
      handleError(error);
    }
  });

  document.getElementById('practice-form').addEventListener('input', capturePracticeDraft);
  document.getElementById('practice-form').addEventListener('change', capturePracticeDraft);
  document.getElementById('chat-form').addEventListener('input', captureChatDraft);
  document.getElementById('chat-form').addEventListener('change', captureChatDraft);

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
      await refreshActiveWorkspace();
      showTransientMessage('Material import finished.', 'success');
    } catch (error) {
      handleError(error);
    } finally {
      if (submitButton) submitButton.disabled = false;
    }
  });

  document.getElementById('practice-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;

    const request = buildPracticeRequestFromForm();
    setPracticeDraft(state.activeWorkspace, request);

    openDecisionReview({
      kicker: 'Human approval before generation',
      title: 'Review this grounded practice request',
      description: 'The system should only generate after you confirm the exact topic, coverage, difficulty, and grounding behavior.',
      sections: buildPracticeReviewSections(request),
      footnote: 'If the topic is weakly grounded, the service can pause and request clarification instead of silently producing a brittle draft.',
      confirmLabel: 'Generate grounded draft',
      onConfirm: async () => {
        const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        });
        await pollJob(payload.job.job_id);
        await loadWorkspaces();
      },
    });
  });

  document.getElementById('revise-practice').addEventListener('click', () => {
    const practice = state.activeWorkspace?.active_practice_set;
    if (!practice) return;

    const { targetQuestionIds, lockedQuestionIds } = buildRevisionSelections();
    const overlapping = targetQuestionIds.filter((questionId) => lockedQuestionIds.includes(questionId));
    if (overlapping.length) {
      showTransientMessage('A question cannot be both locked and marked for regeneration.', 'warning');
      return;
    }

    openDecisionReview({
      kicker: 'Selective revision',
      title: targetQuestionIds.length ? 'Review this targeted practice revision' : 'Review this practice variant request',
      description: 'Locked questions stay fixed while selected questions regenerate. If you regenerate nothing, the system creates a cleaner variant while preserving the locked questions and coverage.',
      sections: [
        { label: 'Questions to regenerate', value: String(targetQuestionIds.length) },
        { label: 'Questions to lock', value: String(lockedQuestionIds.length) },
        { label: 'Coverage rule', value: 'Maintain coverage' },
        { label: 'History behavior', value: 'Prior practice set stays available in audit history' },
      ],
      noteField: {
        label: 'Revision note',
        placeholder: 'Describe what should improve in the regenerated questions.',
        defaultValue: targetQuestionIds.length
          ? 'Regenerate the selected questions with less redundancy and stronger lecture grounding.'
          : 'Create a cleaner variant while preserving the locked questions and overall coverage.',
      },
      footnote: 'This review step makes the revision intent explicit before the assistant changes any questions.',
      confirmLabel: 'Revise practice test',
      onConfirm: async (note) => {
        const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/practice-sets/${practice.practice_set_id}/revise`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_question_ids: targetQuestionIds,
            locked_question_ids: lockedQuestionIds,
            feedback_note: note,
            maintain_coverage: true,
            action: targetQuestionIds.length ? 'revise_selected' : 'create_variant',
          }),
        });
        await pollJob(payload.job.job_id);
        await loadWorkspaces();
      },
    });
  });

  document.getElementById('new-conversation').addEventListener('click', async () => {
    if (!state.activeWorkspace) return;
    try {
      const chatDraft = ensureChatDraft(state.activeWorkspace);
      await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: `Chat ${new Date().toLocaleTimeString()}`,
          grounding_mode: chatDraft.grounding_mode,
        }),
      });
      await refreshActiveWorkspace();
      setTab('ask');
      document.getElementById('chat-question')?.focus();
      showTransientMessage('New empty chat started. Send a message to use the model.', 'success');
    } catch (error) {
      handleError(error);
    }
  });

  document.getElementById('clear-conversation').addEventListener('click', async () => {
    if (!state.activeWorkspace?.active_conversation) return;
    try {
      await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations/${state.activeWorkspace.active_conversation.conversation_id}/clear`, { method: 'POST' });
      await refreshActiveWorkspace();
      showTransientMessage('Current chat cleared.', 'success');
    } catch (error) {
      handleError(error);
    }
  });

  document.getElementById('chat-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.activeWorkspace) return;

    const sendButton = event.currentTarget.querySelector('button[type="submit"]');
    if (sendButton) sendButton.disabled = true;

    try {
      captureChatDraft();
      const draft = ensureChatDraft(state.activeWorkspace);
      const questionText = draft.text.trim();
      if (!questionText) {
        if (sendButton) sendButton.disabled = false;
        return;
      }

      const conversation = await createConversationIfNeeded();
      setGlobalActivity('Sending your grounded question...', true);
      const payload = await api(`/api/workspaces/${state.activeWorkspace.workspace_id}/conversations/${conversation.conversation_id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: questionText,
          grounding_mode: draft.grounding_mode,
          response_style: draft.response_style,
        }),
      });
      setChatDraft(state.activeWorkspace, { text: '' });
      await refreshActiveWorkspace();
      await pollJob(payload.job.job_id);
      await loadWorkspaces();
    } catch (error) {
      handleError(error);
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

  function openSidebar() { document.body.classList.add('sidebar-open'); }
  function closeSidebar() { document.body.classList.remove('sidebar-open'); }

  document.getElementById('sidebar-toggle')?.addEventListener('click', openSidebar);
  document.getElementById('sidebar-close')?.addEventListener('click', closeSidebar);
  document.getElementById('sidebar-backdrop')?.addEventListener('click', closeSidebar);

  document.getElementById('source-close')?.addEventListener('click', closeSourceViewer);

  ['decision-review-dismiss', 'decision-review-close', 'decision-review-cancel'].forEach((id) => {
    document.getElementById(id)?.addEventListener('click', closeDecisionReview);
  });

  document.getElementById('decision-review-confirm')?.addEventListener('click', async () => {
    const review = state.ui.review;
    if (!review) return;

    const confirmButton = document.getElementById('decision-review-confirm');
    const note = document.getElementById('decision-review-note').value.trim();
    confirmButton.disabled = true;
    try {
      await review.onConfirm?.(note);
      closeDecisionReview();
    } catch (error) {
      handleError(error);
    } finally {
      confirmButton.disabled = false;
    }
  });
}

async function init() {
  bindEvents();
  await loadStatus();
  await loadWorkspaces();
  syncBodyLayout();
}

init().catch((error) => {
  handleError(error);
});
