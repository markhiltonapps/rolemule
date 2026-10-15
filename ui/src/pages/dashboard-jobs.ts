/**
 * Find Jobs page — saved searches, watched careers pages, search runs, and leads.
 */
import { getApiBase, getAuthToken, requireLogin } from '../shared/auth';
import { errorMessageFromBody } from '../shared/auth-api';
import { escapeHtml } from '../shared/dom-security';
import { notify } from '../shared/notify';
import { addTrackedSession } from '../shared/workflow-tracking';

type LeadStatus = 'new' | 'saved' | 'dismissed' | 'analyzed';

interface CompanyStatus {
  ok: boolean;
  jobs: number;
  error: string | null;
}

interface CompanyEntry {
  platform: string;
  token: string;
  name: string;
  page_url: string;
  last_status: CompanyStatus | null;
}

interface LastRun {
  status: 'running' | 'completed' | 'failed';
  started_at?: string;
  finished_at?: string;
  new_leads?: number;
  board_matches?: number;
  web_search?: boolean;
  web_matches?: number;
  web_errors?: string[];
  companies_failed?: number;
  error?: string;
}

interface JobFinderState {
  searches: string[];
  companies: CompanyEntry[];
  last_run: LastRun | null;
  running: boolean;
}

interface Lead {
  id: string;
  source: string;
  title: string;
  company: string;
  location: string | null;
  is_remote: boolean;
  url: string;
  has_description: boolean;
  posted_at: string | null;
  matched_searches: string[];
  status: LeadStatus;
  description?: string | null;
}

interface LeadsResponse {
  leads: Lead[];
  counts: Record<LeadStatus, number>;
}

interface ApiErrorBody {
  error_code?: string;
  message?: string;
  detail?: string;
}

const POLL_MS = 3000;
const AUTO_RUN_AFTER_MS = 12 * 60 * 60 * 1000;

let currentStatus: LeadStatus = 'new';
let pollTimer: number | null = null;
let leadsCache: Lead[] = [];

function api(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${getAuthToken() ?? ''}`);
  if (init.body && typeof init.body === 'string') headers.set('Content-Type', 'application/json');
  return fetch(`${getApiBase()}${path}`, { ...init, headers });
}

async function readError(response: Response, fallback: string): Promise<ApiErrorBody & { text: string }> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    return { ...body, text: errorMessageFromBody(body, fallback) };
  } catch {
    return { text: fallback };
  }
}

function showApiKeyAlert(): void {
  const container = document.getElementById('alertContainer');
  if (!container) return;
  container.innerHTML = `
    <div class="alert alert-warning alert-dismissible fade show" role="alert">
      <i class="fas fa-key me-2"></i>
      <strong>API key required.</strong>
      To use AI web search or analyze jobs, add your AI provider in
      <a href="/settings?tab=ai-setup" class="alert-link">Settings &rarr; AI Setup</a>.
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>`;
}

function safeHttpUrl(url: string): string | null {
  return /^https?:\/\//i.test(url) ? url : null;
}

function timeAgo(iso: string | undefined): string {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

function formatDate(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ''
    : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

// =============================================================================
// Status + settings
// =============================================================================

function renderRunStatus(state: JobFinderState): void {
  const statusEl = document.getElementById('runStatus');
  const warningsEl = document.getElementById('runWarnings');
  const runBtn = document.getElementById('runSearchBtn') as HTMLButtonElement | null;
  if (!statusEl || !warningsEl) return;

  const run = state.last_run;
  statusEl.classList.toggle('is-running', state.running);
  if (runBtn) runBtn.disabled = state.running;

  if (state.running) {
    statusEl.textContent = 'Searching… this can take a minute (longer with AI web search).';
  } else if (!run) {
    statusEl.textContent = 'No searches yet — click Search now.';
  } else if (run.status === 'failed') {
    statusEl.textContent = run.error || 'The last search failed. Please try again.';
  } else if (run.status === 'running') {
    statusEl.textContent = 'The last search did not finish. Click Search now to try again.';
  } else {
    const parts = [`Last searched ${timeAgo(run.finished_at)}`];
    parts.push(`${run.new_leads ?? 0} new job${run.new_leads === 1 ? '' : 's'}`);
    if (run.web_search) parts.push(`AI web search found ${run.web_matches ?? 0}`);
    statusEl.textContent = parts.join(' · ');
  }

  const warnings: string[] = [];
  if (!state.running && run?.status === 'completed') {
    if (run.companies_failed) {
      warnings.push(
        `${run.companies_failed} company careers page${run.companies_failed === 1 ? '' : 's'} couldn't be loaded — see "Searches & companies to watch".`,
      );
    }
    if (run.web_errors?.length) {
      warnings.push(`AI web search problems: ${run.web_errors.slice(0, 3).join(' | ')}`);
    }
  }
  warningsEl.textContent = warnings.join(' ');
  warningsEl.classList.toggle('is-hidden', warnings.length === 0);
}

function renderSettings(state: JobFinderState): void {
  const searchesInput = document.getElementById('searchesInput') as HTMLTextAreaElement | null;
  const companiesInput = document.getElementById('companiesInput') as HTMLTextAreaElement | null;
  const statusList = document.getElementById('companyStatusList');
  if (searchesInput && document.activeElement !== searchesInput) {
    searchesInput.value = state.searches.join('\n');
  }
  if (companiesInput && document.activeElement !== companiesInput) {
    companiesInput.value = state.companies.map((c) => `${c.name} | ${c.page_url}`).join('\n');
  }
  if (statusList) {
    statusList.innerHTML = state.companies
      .filter((c) => c.last_status)
      .map((c) => {
        const s = c.last_status as CompanyStatus;
        const cls = s.ok ? 'status-ok' : 'status-failed';
        const text = s.ok ? `${s.jobs} open roles` : s.error || 'could not load';
        return `<li><span class="${cls}">${s.ok ? '✓' : '✗'}</span> ${escapeHtml(c.name)} — ${escapeHtml(text)}</li>`;
      })
      .join('');
  }
}

async function loadState(): Promise<JobFinderState | null> {
  const response = await api('/job-finder');
  if (!response.ok) {
    const err = await readError(response, 'Could not load Find Jobs.');
    notify(err.text, 'danger');
    return null;
  }
  const state = (await response.json()) as JobFinderState;
  renderRunStatus(state);
  renderSettings(state);
  return state;
}

async function saveSettings(event: Event): Promise<void> {
  event.preventDefault();
  const searchesInput = document.getElementById('searchesInput') as HTMLTextAreaElement | null;
  const companiesInput = document.getElementById('companiesInput') as HTMLTextAreaElement | null;
  const saveBtn = document.getElementById('saveSettingsBtn') as HTMLButtonElement | null;
  if (!searchesInput || !companiesInput) return;
  const lines = (text: string) =>
    text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);

  if (saveBtn) saveBtn.disabled = true;
  try {
    const response = await api('/job-finder/config', {
      method: 'PUT',
      body: JSON.stringify({
        searches: lines(searchesInput.value),
        companies: lines(companiesInput.value),
      }),
    });
    if (!response.ok) {
      const err = await readError(response, 'Could not save.');
      notify(err.text, 'danger');
      return;
    }
    const state = (await response.json()) as JobFinderState;
    searchesInput.blur();
    companiesInput.blur();
    renderSettings(state);
    renderRunStatus(state);
    notify('Saved. Click Search now to look for jobs with the new settings.', 'success');
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

// =============================================================================
// Runs
// =============================================================================

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function schedulePoll(): void {
  stopPolling();
  pollTimer = window.setTimeout(async () => {
    const state = await loadState();
    if (state?.running) {
      schedulePoll();
    } else {
      pollTimer = null;
      await loadLeads();
    }
  }, POLL_MS);
}

async function startRun(includeWeb: boolean, quiet = false): Promise<void> {
  const response = await api('/job-finder/run', {
    method: 'POST',
    body: JSON.stringify({ include_web: includeWeb }),
  });
  if (!response.ok) {
    const err = await readError(response, 'Could not start the search.');
    if (err.error_code === 'CFG_6001') {
      showApiKeyAlert();
    } else if (!quiet || response.status !== 409) {
      notify(err.text, response.status === 409 ? 'info' : 'danger');
    }
    if (response.status === 409) schedulePoll();
    return;
  }
  const body = (await response.json()) as { web_search_skipped_reason?: string | null };
  if (body.web_search_skipped_reason) notify(body.web_search_skipped_reason, 'warning');
  const statusEl = document.getElementById('runStatus');
  if (statusEl) {
    statusEl.classList.add('is-running');
    statusEl.textContent = 'Searching… this can take a minute (longer with AI web search).';
  }
  const runBtn = document.getElementById('runSearchBtn') as HTMLButtonElement | null;
  if (runBtn) runBtn.disabled = true;
  schedulePoll();
}

// =============================================================================
// Leads
// =============================================================================

function sourceLabel(lead: Lead): string {
  return lead.source === 'web'
    ? '<span class="lead-badge lead-badge--web">AI web search — double-check details</span>'
    : '<span class="lead-badge">Company careers page</span>';
}

function leadActions(lead: Lead): string {
  const url = safeHttpUrl(lead.url);
  const id = escapeHtml(lead.id);
  const open = url
    ? `<a class="btn btn-sm btn-outline-secondary" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer"><i class="fas fa-external-link-alt me-1" aria-hidden="true"></i>Open posting</a>`
    : '';
  const analyze =
    lead.status === 'analyzed'
      ? ''
      : `<button type="button" class="btn btn-sm btn-primary" data-action="analyze" data-id="${id}"><i class="fas fa-magic me-1" aria-hidden="true"></i>Analyze</button>`;
  const save =
    lead.status === 'saved'
      ? `<button type="button" class="btn btn-sm btn-outline-secondary" data-action="set-status" data-status="new" data-id="${id}">Unsave</button>`
      : `<button type="button" class="btn btn-sm btn-outline-secondary" data-action="set-status" data-status="saved" data-id="${id}"><i class="far fa-bookmark me-1" aria-hidden="true"></i>Save</button>`;
  const dismiss =
    lead.status === 'dismissed'
      ? `<button type="button" class="btn btn-sm btn-outline-secondary" data-action="set-status" data-status="new" data-id="${id}">Restore</button>`
      : `<button type="button" class="btn btn-sm btn-outline-secondary" data-action="set-status" data-status="dismissed" data-id="${id}">Not interested</button>`;
  return `${analyze}${open}${save}${dismiss}`;
}

function renderLeads(data: LeadsResponse): void {
  leadsCache = data.leads;
  (Object.keys(data.counts) as LeadStatus[]).forEach((status) => {
    const el = document.querySelector(`[data-count="${status}"]`);
    if (el) el.textContent = String(data.counts[status] ?? 0);
  });

  const list = document.getElementById('leadList');
  if (!list) return;
  if (!data.leads.length) {
    const emptyText: Record<LeadStatus, string> = {
      new: 'No new jobs yet. Click Search now, or add more searches and companies below.',
      saved: 'Jobs you save will appear here.',
      analyzed: 'Jobs you analyze will appear here.',
      dismissed: 'Jobs you mark "Not interested" will appear here.',
    };
    list.innerHTML = `<div class="leads-empty">${escapeHtml(emptyText[currentStatus])}</div>`;
    return;
  }

  list.innerHTML = data.leads
    .map((lead) => {
      const meta = [lead.company, lead.location].filter(Boolean).map((v) => escapeHtml(v)).join(' · ');
      const posted = formatDate(lead.posted_at);
      const badges = [
        lead.is_remote ? '<span class="lead-badge lead-badge--remote">Remote</span>' : '',
        sourceLabel(lead),
        posted ? `<span class="lead-badge">Posted ${escapeHtml(posted)}</span>` : '',
        ...lead.matched_searches
          .slice(0, 3)
          .map((q) => `<span class="lead-badge">Matches: ${escapeHtml(q)}</span>`),
      ].join('');
      return `
        <article class="lead-card" data-lead-id="${escapeHtml(lead.id)}">
          <h3 class="lead-title">${escapeHtml(lead.title)}</h3>
          <div class="lead-meta">${meta}</div>
          <div class="lead-badges">${badges}</div>
          <div class="lead-actions">${leadActions(lead)}</div>
        </article>`;
    })
    .join('');
}

async function loadLeads(): Promise<void> {
  const response = await api(`/job-finder/leads?status=${encodeURIComponent(currentStatus)}`);
  if (!response.ok) {
    const err = await readError(response, 'Could not load jobs.');
    notify(err.text, 'danger');
    return;
  }
  renderLeads((await response.json()) as LeadsResponse);
}

async function setLeadStatus(id: string, status: LeadStatus): Promise<boolean> {
  const response = await api(`/job-finder/leads/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
  if (!response.ok) {
    const err = await readError(response, 'Could not update the job.');
    notify(err.text, 'danger');
    return false;
  }
  return true;
}

async function analyzeLead(id: string, button: HTMLButtonElement): Promise<void> {
  const cached = leadsCache.find((l) => l.id === id);
  if (cached && !cached.has_description) {
    const url = cached ? safeHttpUrl(cached.url) : null;
    if (url) window.open(url, '_blank', 'noopener,noreferrer');
    notify(
      'This job came from AI web search, so RoleMule doesn\'t have the full description. On the posting page, use the RoleMule extension\'s "Analyze This Job" — or paste the description into New Application.',
      'info',
    );
    return;
  }

  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="fas fa-spinner fa-spin me-1" aria-hidden="true"></i>Starting…';
  try {
    const detailRes = await api(`/job-finder/leads/${encodeURIComponent(id)}`);
    if (!detailRes.ok) {
      const err = await readError(detailRes, 'Could not load the job.');
      notify(err.text, 'danger');
      return;
    }
    const lead = (await detailRes.json()) as Lead;
    const formData = new FormData();
    formData.append('job_text', `${lead.title}\n${lead.company}\n${lead.location ?? ''}\n\n${lead.description ?? ''}`);
    formData.append('detected_title', lead.title);
    formData.append('detected_company', lead.company);
    const url = safeHttpUrl(lead.url);
    if (url) formData.append('job_url', url);

    const response = await fetch(`${getApiBase()}/workflow/start`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${getAuthToken() ?? ''}` },
      body: formData,
    });
    if (!response.ok) {
      const err = await readError(response, 'Could not start the analysis.');
      if (err.error_code === 'CFG_6001') {
        showApiKeyAlert();
      } else if (err.error_code === 'RES_3002') {
        await setLeadStatus(id, 'analyzed');
        notify('You already have this job on your applications list — open it from your dashboard.', 'warning');
        await loadLeads();
      } else {
        notify(err.text, 'danger');
      }
      return;
    }
    const started = (await response.json()) as { session_id?: string };
    if (started.session_id) addTrackedSession(started.session_id);
    await setLeadStatus(id, 'analyzed');
    sessionStorage.setItem(
      'new_application_toast',
      'Application submitted! AI agents are analyzing it in the background.',
    );
    window.location.href = '/dashboard';
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

// =============================================================================
// Wiring
// =============================================================================

function selectTab(status: LeadStatus): void {
  currentStatus = status;
  document.querySelectorAll<HTMLButtonElement>('.lead-tab').forEach((tab) => {
    const active = tab.dataset.status === status;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  void loadLeads();
}

function attachListeners(): void {
  document.addEventListener('click', (event) => {
    const target = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-action]');
    if (!target) return;
    const action = target.dataset.action;
    if (action === 'run-search') {
      const includeWeb = (document.getElementById('includeWebSearch') as HTMLInputElement | null)?.checked ?? false;
      void startRun(includeWeb);
    } else if (action === 'show-status' && target.dataset.status) {
      selectTab(target.dataset.status as LeadStatus);
    } else if (action === 'set-status' && target.dataset.id && target.dataset.status) {
      const id = target.dataset.id;
      const status = target.dataset.status as LeadStatus;
      void setLeadStatus(id, status).then((ok) => {
        if (ok) void loadLeads();
      });
    } else if (action === 'analyze' && target.dataset.id) {
      void analyzeLead(target.dataset.id, target as HTMLButtonElement);
    }
  });
  document.getElementById('jobFinderSettingsForm')?.addEventListener('submit', (e) => {
    void saveSettings(e);
  });
}

async function init(): Promise<void> {
  if (!requireLogin()) return;
  if (
    typeof window.syncProfileCompletionFromApi !== 'function' ||
    !(await window.syncProfileCompletionFromApi())
  ) {
    return;
  }
  attachListeners();
  const state = await loadState();
  await loadLeads();
  if (!state) return;
  if (state.running) {
    schedulePoll();
    return;
  }
  // Keep results fresh: quietly re-check careers pages (free) when stale
  const finished = state.last_run?.finished_at ? Date.parse(state.last_run.finished_at) : NaN;
  if (Number.isNaN(finished) || Date.now() - finished > AUTO_RUN_AFTER_MS) {
    await startRun(false, true);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    void init();
  });
} else {
  void init();
}
