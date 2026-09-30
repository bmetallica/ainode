/* AINode Command Center — Single Page Application */

// Coarse phase → [label, percent]. Module level because two things read it:
// the instance card's one-word status and the details dialog. Keep in step
// with LOAD_PHASE_ORDER in ainode/engine/load_phase.py; a phase missing here
// renders as the `idle` fallback, a flat 8% that reads as a hang.
const INSTANCE_PHASE_INFO = {
  idle: ['starting', 8], starting: ['starting', 12],
  distributing: ['copying weights to peers', 26],
  distributed_init: ['connecting nodes', 34],
  loading_weights: ['loading weights', 50],
  profiling: ['profiling', 84], ready: ['ready', 100],
  failed: ['failed', 100],
};

const AINode = {
  state: {
    status: null,
    nodes: [],
    metrics: null,
    messages: [],
    conversations: [],
    currentConversation: null,
    streaming: false,
    streamMetrics: { ttft: 0, tps: 0, tokens: 0 },
    topology: null,
    trainingJobs: [],
    trainingView: 'list',        // legacy — kept for detail/form routing
    trainingTab: 'overview',     // new: overview | datasets | runs | templates | benchmarks
    trainingDetailId: null,
    trainingLossData: [],
    trainingStats: null,
    trainingTemplates: [],
    datasets: [],
    runsFilter: 'all',
    expandedDatasetId: null,
    wizardState: null,
    pollInterval: null,
    metricsInterval: null,
    abortController: null,
    shardingStatus: null,
    currentView: 'dashboard',
    modelsFilter: 'catalog',
    modelsSearch: '',
    modelsSort: 'recommended',
    configSection: 'credentials',
    configData: {
      secrets: null,
      cluster: null,
      config: null,
    },
    configRevealed: {},
  },

  // ========================================================================
  //  INITIALIZATION
  // ========================================================================

  init() {
    this.watchForSignOut();
    this.checkDefaultPassword();
    this.loadConversations();
    this.bindNav();
    this.bindChat();
    this.bindLaunchForm();
    this.initTopology();
    this.initClusterUpdateBtn();
    this.startPolling();
    this.renderConversationList();
    // Restore any in-flight downloads from before page refresh
    this.loadActiveDownloads();
    // Also ask the server if there are active jobs we missed
    setTimeout(() => this.reconcileActiveDownloads(), 500);
  },

  initTopology() {
    var canvas = document.getElementById('topology-canvas');
    if (canvas && typeof Topology !== 'undefined') {
      this.state.topology = new Topology(canvas);
    }
  },

  // ========================================================================
  //  TOAST NOTIFICATION SYSTEM
  // ========================================================================

  toast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toast-container');
    if (!container) return;
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML = '<span class="toast-message">' + this.esc(message) + '</span><button class="toast-close">&times;</button>';
    container.appendChild(toast);
    requestAnimationFrame(function () { toast.classList.add('toast-visible'); });
    var dismiss = function () {
      toast.classList.remove('toast-visible');
      toast.classList.add('toast-fade-out');
      setTimeout(function () { toast.remove(); }, 300);
    };
    toast.querySelector('.toast-close').addEventListener('click', dismiss);
    setTimeout(dismiss, 4000);
  },

  // ========================================================================
  //  CONVERSATION HISTORY (localStorage-backed)
  // ========================================================================

  loadConversations() {
    try {
      this.state.conversations = JSON.parse(localStorage.getItem('ainode_conversations') || '[]');
    } catch (e) {
      this.state.conversations = [];
    }
    // Restore last active conversation
    if (this.state.conversations.length > 0 && !this.state.currentConversation) {
      // Don't auto-load; let user pick
    }
  },

  saveConversations() {
    localStorage.setItem('ainode_conversations', JSON.stringify(this.state.conversations));
  },

  getCurrentConversation() {
    return this.state.conversations.find(function (c) { return c.id === AINode.state.currentConversation; }) || null;
  },

  newConversation() {
    var id = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    var sel = document.getElementById('chat-model');
    var conv = { id: id, title: 'New Chat', messages: [], created_at: Date.now(), model: sel ? sel.value : '' };
    this.state.conversations.unshift(conv);
    this.state.currentConversation = id;
    this.state.messages = [];
    this.saveConversations();
    this.renderConversationList();
    this.renderChatMessages();
  },

  loadConversation(id) {
    var conv = this.state.conversations.find(function (c) { return c.id === id; });
    if (!conv) return;
    this.state.currentConversation = id;
    this.state.messages = conv.messages.slice();
    var select = document.getElementById('chat-model');
    if (select && conv.model) {
      for (var i = 0; i < select.options.length; i++) {
        if (select.options[i].value === conv.model) { select.value = conv.model; break; }
      }
    }
    this.renderConversationList();
    this.renderChatMessages();
  },

  deleteConversation(id) {
    this.state.conversations = this.state.conversations.filter(function (c) { return c.id !== id; });
    if (this.state.currentConversation === id) {
      this.state.currentConversation = null;
      this.state.messages = [];
      this.renderChatMessages();
    }
    this.saveConversations();
    this.renderConversationList();
    this.toast('Conversation deleted', 'info');
  },

  saveCurrentConversation() {
    var conv = this.getCurrentConversation();
    if (!conv) return;
    conv.messages = this.state.messages.slice();
    var sel = document.getElementById('chat-model');
    if (sel) conv.model = sel.value || conv.model;
    var firstUser = conv.messages.find(function (m) { return m.role === 'user'; });
    if (firstUser) conv.title = firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '...' : '');
    this.saveConversations();
    this.renderConversationList();
  },

  renderConversationList() {
    var list = document.getElementById('conversation-list');
    if (!list) return;
    var self = this;
    var searchInput = document.getElementById('chat-search');
    var query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    var filtered = this.state.conversations.filter(function (c) {
      if (!query) return true;
      return c.title.toLowerCase().indexOf(query) !== -1;
    });

    if (filtered.length === 0) {
      list.innerHTML = '<div class="conv-empty">' + (query ? 'No matches' : 'No conversations yet') + '</div>';
      return;
    }

    list.innerHTML = filtered.map(function (conv) {
      var active = conv.id === self.state.currentConversation ? ' active' : '';
      var dateStr = new Date(conv.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      return '<div class="conv-item' + active + '" data-conv-id="' + self.esc(conv.id) + '">' +
        '<div class="conv-item-content">' +
        '<div class="conv-item-title">' + self.esc(conv.title) + '</div>' +
        '<div class="conv-item-date">' + dateStr + '</div>' +
        '</div>' +
        '<button class="conv-delete" data-delete-id="' + self.esc(conv.id) + '" title="Delete">&times;</button>' +
        '</div>';
    }).join('');

    list.querySelectorAll('.conv-item').forEach(function (el) {
      el.addEventListener('click', function (e) {
        if (e.target.closest('.conv-delete')) return;
        self.loadConversation(el.dataset.convId);
      });
    });
    list.querySelectorAll('.conv-delete').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.deleteConversation(btn.dataset.deleteId);
      });
    });
  },

  // ========================================================================
  //  PROFILES — the set of models this node should be serving
  // ========================================================================

  _profileState: { editing: null, applying: '', job: null, jobChecked: false, dismissed: '' },

  async renderProfiles() {
    var mount = document.getElementById('profiles-content');
    if (!mount) return;
    var self = this;
    var data = await this.fetchJSON('/api/profiles');
    var profiles = (data && data.profiles) || [];
    var defaultName = (data && data.default) || '';
    // An apply started in another tab, or before a reload, is picked up here.
    if (!this._profileState.jobChecked) {
      this._profileState.jobChecked = true;
      var current = await this.fetchJSON('/api/profiles/jobs/current').catch(function () { return null; });
      if (current && current.job) {
        this._profileState.job = current.job;
        if (current.job.state === 'running') this._pollProfileJob();
      }
    }

    var html = '<div id="profile-apply-panel">' + this._renderProfileJob() + '</div>';

    // Capture is the realistic way to a first profile: get the deployment
    // right by hand, then keep it — rather than filling in a dozen fields
    // per model in a form and hoping it matches what worked.
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Save what is running</h3>';
    html += '  </div>';
    html += '  <div class="profile-capture-row">';
    html += '    <input class="form-input" id="profile-capture-name" placeholder="Profile name (e.g. Endausbau)">';
    html += '    <input class="form-input" id="profile-capture-desc" placeholder="What it is for (optional)">';
    html += '    <button class="btn-nvidia" id="profile-capture-btn">Save current state</button>';
    html += '  </div>';
    html += '  <div class="server-hint">Records every model running right now — which nodes it spans, its memory fraction, context length and engine flags.</div>';
    html += '</section>';

    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Profiles</h3>';
    html += '    <span class="server-section-meta">' + profiles.length + ' saved</span>';
    html += '    <button class="btn-nvidia server-btn-sm" id="profile-wizard-btn">New profile (wizard)</button>';
    html += '  </div>';
    if (!profiles.length) {
      html += '  <div class="server-empty">No profiles yet. Load the models you want, then click <strong>Save current state</strong>.</div>';
    } else {
      profiles.forEach(function (p) {
        html += self._renderProfileCard(p, defaultName);
      });
    }
    html += '</section>';

    mount.innerHTML = html;
    this._bindProfileActions();
  },

  _renderProfileCard(profile, defaultName) {
    var self = this;
    var isDefault = profile.name === defaultName;
    var h = '<div class="profile-card" data-profile="' + this.esc(profile.name) + '">';
    h += '  <div class="profile-card-head">';
    h += '    <span class="profile-name">' + this.esc(profile.name) + '</span>';
    if (isDefault) h += '    <span class="profile-default-badge">DEFAULT · loaded at startup</span>';
    if (profile.wizard) h += '    <span class="pw-kind">planned in the wizard</span>';
    h += '    <span class="profile-card-actions">';
    h += '      <button class="btn-nvidia server-btn-sm" data-profile-apply="' + this.esc(profile.name) + '">Apply</button>';
    h += '      <button class="btn-ghost server-btn-sm" data-profile-wizard="' + this.esc(profile.name) + '">Edit in wizard</button>';
    h += '      <button class="btn-ghost server-btn-sm" data-profile-opencode="' + this.esc(profile.name) + '">OpenCode config</button>';
    h += '      <button class="btn-ghost server-btn-sm" data-profile-default="' + this.esc(profile.name) + '">' + (isDefault ? 'Unset default' : 'Set default') + '</button>';
    h += '      <button class="btn-ghost server-btn-sm" data-profile-delete="' + this.esc(profile.name) + '">Delete</button>';
    h += '    </span>';
    h += '  </div>';
    if (profile.description) {
      h += '  <div class="profile-desc">' + this.esc(profile.description) + '</div>';
    }
    var entries = profile.entries || [];
    if (!entries.length) {
      h += '  <div class="server-empty">Empty profile — applying it stops everything.</div>';
    } else {
      h += '  <table class="profile-table"><thead><tr>' +
           '<th>Model</th><th>Nodes</th><th>Split</th><th>Memory</th><th>Context</th><th>KV</th><th>Measured</th>' +
           '</tr></thead><tbody>';
      // Node names where the cluster knows them; a profile stores ids.
      var names = {};
      (self.state.nodes || []).forEach(function (n) { names[n.node_id] = n.node_name || n.node_id; });
      entries.forEach(function (e) {
        var nodes = (e.node_ids && e.node_ids.length)
          ? e.node_ids.map(function (id) { return names[id] || id; }).join(', ') : 'this node';
        var split = e.kind === 'embedding' ? 'embedding'
          : (e.kind === 'image' ? 'image'
             : (e.strategy || ((e.node_ids || []).length > 1 ? 'auto' : 'solo')));
        var mem = e.gpu_memory_utilization ? Math.round(e.gpu_memory_utilization * 100) + '%' : '—';
        var ctx = e.max_model_len ? self.formatNumber(e.max_model_len) : '—';
        h += '<tr>' +
             '<td class="mono">' + self.esc(e.model) + '</td>' +
             '<td>' + self.esc(nodes) + '</td>' +
             '<td>' + self.esc(split) + '</td>' +
             '<td>' + mem + '</td>' +
             '<td>' + ctx + '</td>' +
             '<td>' + self.esc(e.kv_cache_dtype || 'auto') + '</td>' +
             '<td>' + self._measuredCell((profile.measured || {})[e.model + '@' + (e.node_ids || []).join(',')]) + '</td>' +
             '</tr>';
      });
      h += '  </tbody></table>';
    }
    h += '  <div class="profile-report" data-keep id="profile-report-' + this.esc(profile.name) + '"></div>';
    h += '</div>';
    return h;
  },

  _bindProfileActions() {
    var self = this;

    document.querySelectorAll('[data-profile-opencode]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.showProfileOpencode(btn.getAttribute('data-profile-opencode'));
      });
    });
    this._bindProfileJob();

    var wizardBtn = document.getElementById('profile-wizard-btn');
    if (wizardBtn) {
      wizardBtn.addEventListener('click', function () { self.openProfileWizard(); });
    }
    document.querySelectorAll('[data-profile-wizard]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.openProfileWizard(btn.getAttribute('data-profile-wizard'));
      });
    });

    var captureBtn = document.getElementById('profile-capture-btn');
    if (captureBtn) {
      captureBtn.addEventListener('click', function () { self.captureProfile(); });
    }

    document.querySelectorAll('[data-profile-apply]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.applyProfile(btn.getAttribute('data-profile-apply'), btn);
      });
    });
    document.querySelectorAll('[data-profile-default]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.setDefaultProfile(btn.getAttribute('data-profile-default'),
                               btn.textContent.indexOf('Unset') === 0);
      });
    });
    document.querySelectorAll('[data-profile-delete]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.deleteProfile(btn.getAttribute('data-profile-delete'));
      });
    });
  },

  async captureProfile() {
    var nameEl = document.getElementById('profile-capture-name');
    var descEl = document.getElementById('profile-capture-desc');
    var name = (nameEl && nameEl.value || '').trim();
    if (!name) {
      this.toast('Give the profile a name first', 'error');
      return;
    }
    var body = { name: name, description: (descEl && descEl.value || '').trim() };
    var resp = await fetch('/api/profiles/capture', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var data = await resp.json().catch(function () { return {}; });
    if (resp.status === 409) {
      if (!confirm('A profile named "' + name + '" already exists. Replace it?')) return;
      body.overwrite = true;
      resp = await fetch('/api/profiles/capture', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      data = await resp.json().catch(function () { return {}; });
    }
    if (!resp.ok) {
      this.toast(data.error || 'Could not save the profile', 'error');
      return;
    }
    this.toast('Saved "' + name + '" with ' + (data.captured || 0) + ' model(s)', 'success');
    if (nameEl) nameEl.value = '';
    if (descEl) descEl.value = '';
    this.renderProfiles();
  },

  async applyProfile(name, btn, confirmed) {
    if (!name) return;
    var job = this._profileState.job;
    if (job && job.state === 'running') {
      this.toast('"' + job.profile + '" is still being applied', 'warning');
      return;
    }
    // Applying converges: it stops what the profile does not list. Say so
    // before doing it, because "apply" reads like "add" to most people. The
    // wizard has already shown exactly what will stop, so it does not ask.
    if (!confirmed && !confirm('Apply "' + name + '"?\n\nOn every node this profile uses, models it does not list there will be stopped, and missing ones started one after another. This can take several minutes.')) return;
    var resp = await fetch('/api/profiles/' + encodeURIComponent(name) + '/apply', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ background: true }),
    });
    var data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      this.toast(data.error || 'Could not apply "' + name + '"', 'error');
      return;
    }
    this._profileState.job = data.job;
    this._profileState.dismissed = '';
    this._drawProfileJob();
    this._pollProfileJob();
  },

  // -- the apply job (profiles/jobs.py) ------------------------------------

  _pollProfileJob() {
    var self = this;
    if (this._profileJobTimer) return;
    var tick = async function () {
      var job = self._profileState.job;
      if (!job) { self._profileJobTimer = null; return; }
      var data = await self.fetchJSON('/api/profiles/jobs/' + job.id).catch(function () { return null; });
      if (data && data.job) {
        var was = job.state;
        self._profileState.job = data.job;
        self._drawProfileJob();
        if (data.job.state !== 'running') {
          self._profileJobTimer = null;
          if (was === 'running') {
            self.toast(data.job.state === 'done' ? 'Applied "' + data.job.profile + '"'
                       : 'Applying "' + data.job.profile + '": ' + data.job.state,
                       data.job.state === 'done' ? 'success' : 'error');
            self.renderProfiles();
            self.refresh();
          }
          return;
        }
      }
      self._profileJobTimer = setTimeout(tick, 1500);
    };
    this._profileJobTimer = setTimeout(tick, 800);
  },

  _drawProfileJob() {
    var slot = document.getElementById('profile-apply-panel');
    if (!slot) return;
    slot.innerHTML = this._renderProfileJob();
    this._bindProfileJob();
  },

  _renderProfileJob() {
    var self = this;
    var job = this._profileState.job;
    if (!job || job.id === this._profileState.dismissed) return '';
    var names = {};
    (this.state.nodes || []).forEach(function (n) { names[n.node_id] = n.node_name || n.node_id; });
    var clock = function (seconds) { return self.formatSeconds(Math.max(0, seconds || 0)); };
    var now = job.now || (Date.now() / 1000);
    var running = job.state === 'running';
    var title = (job.restore_of ? 'Restoring what ran before “' + job.restore_of + '”'
                                : (running ? 'Applying' : 'Applied') + ' “' + job.profile + '”');
    var labels = {
      pending: 'waiting', starting: 'starting', loading: 'loading', ready: 'ready',
      unchanged: 'already running', slow: 'still loading', failed: 'failed', skipped: 'skipped',
    };
    var h = '<section class="server-section profile-job profile-job-' + job.state + '">';
    h += '<div class="server-section-header"><h3 class="server-section-title">' + this.esc(title) + '</h3>';
    h += '<span class="server-section-meta">' + this.esc(job.phase || '') + ' · ' +
      clock((job.finished_at || now) - job.started_at) + '</span>';
    h += '<span class="profile-card-actions">';
    if (running && !job.cancel_requested) {
      h += '<button class="btn-ghost server-btn-sm" data-profile-job="cancel">Cancel</button>';
    }
    if (running && job.cancel_requested) {
      h += '<span class="pw-kind">cancelling after the current model…</span>';
    }
    if (!running && job.can_restore && !job.restore_of) {
      h += '<button class="btn-ghost server-btn-sm" data-profile-job="restore">Restore what ran before</button>';
    }
    if (!running) {
      h += '<button class="btn-ghost server-btn-sm" data-profile-job="dismiss">Dismiss</button>';
    }
    h += '</span></div>';
    h += '<table class="profile-table"><thead><tr><th>Model</th><th>Nodes</th><th>State</th>' +
      '<th></th><th>Measured</th></tr></thead><tbody>';
    (job.entries || []).forEach(function (e) {
      var nodes = (e.node_ids || []).map(function (id) { return names[id] || id; }).join(', ') || 'head';
      var since = e.since ? ' · ' + clock(now - e.since) : '';
      var live = (e.state === 'starting' || e.state === 'loading') && running;
      h += '<tr><td class="mono">' + self.esc(e.model) + '</td><td>' + self.esc(nodes) + '</td>' +
        '<td><span class="job-state job-state-' + self.esc(e.state) + '">' +
          self.esc(labels[e.state] || e.state) + '</span>' + (live ? since : '') + '</td>' +
        '<td class="job-detail">' + self.esc(e.detail || '') + '</td>' +
        '<td>' + self._measuredCell((job.measured || {})[e.key]) + '</td></tr>';
    });
    h += '</tbody></table>';
    if ((job.stopped || []).length) {
      h += '<div class="profile-report-line">■ stopped: ' +
        job.stopped.map(function (m) { return self.esc(m); }).join(', ') + '</div>';
    }
    if ((job.unreachable || []).length) {
      h += '<div class="profile-report-line error">Not cleared (older build or unreachable): ' +
        job.unreachable.map(function (id) { return self.esc(names[id] || id); }).join(', ') + '</div>';
    }
    if (job.error) h += '<div class="profile-report-line error">' + this.esc(job.error) + '</div>';
    return h + '</section>';
  },

  _bindProfileJob() {
    var self = this;
    document.querySelectorAll('[data-profile-job]').forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', async function () {
        var job = self._profileState.job;
        if (!job) return;
        var action = btn.getAttribute('data-profile-job');
        if (action === 'dismiss') {
          self._profileState.dismissed = job.id;
          self._drawProfileJob();
          return;
        }
        if (action === 'restore' && !confirm('Put back what ran before "' + job.profile +
            '" was applied? This converges the same nodes again.')) return;
        var resp = await fetch('/api/profiles/jobs/' + job.id + '/' + action, { method: 'POST' });
        var data = await resp.json().catch(function () { return {}; });
        if (!resp.ok) {
          self.toast(data.error || 'Could not ' + action, 'error');
          return;
        }
        self._profileState.job = data.job;
        self._drawProfileJob();
        self._pollProfileJob();
      });
    });
  },

  // What a model took the last time its profile was applied.
  _measuredCell(m) {
    if (!m || !m.memory_gb) return '<span class="pw-kind">—</span>';
    var parts = [m.memory_gb + ' GB'];
    if (m.kv_tokens) parts.push(this.formatNumber(m.kv_tokens) + ' tok');
    var byNode = m.memory_by_node || {};
    var title = Object.keys(byNode).map(function (id) { return id + ': ' + byNode[id] + ' GB'; }).join(', ');
    return '<span title="' + this.esc(title) + '">' + this.esc(parts.join(' · ')) + '</span>';
  },

  async showProfileOpencode(name) {
    var self = this;
    var slot = document.getElementById('profile-report-' + name);
    if (!slot) return;
    var base = location.protocol + '//' + location.host;
    var data = await this.fetchJSON('/api/profiles/' + encodeURIComponent(name) +
                                    '/opencode?base_url=' + encodeURIComponent(base))
      .catch(function () { return null; });
    slot = document.getElementById('profile-report-' + name) || slot;
    if (!data || !data.config) {
      slot.innerHTML = '<div class="profile-report-line error">Could not build it.</div>';
      return;
    }
    var text = JSON.stringify(data.config, null, 2);
    slot.innerHTML = (data.notes || []).map(function (n) {
      return '<div class="profile-report-line">' + self.esc(n) + '</div>';
    }).join('') +
      '<div style="display:flex;gap:8px;margin:8px 0"><button class="btn-nvidia server-btn-sm" ' +
      'data-copy="' + this.esc(text) + '">Copy</button><span class="pw-kind" style="align-self:center">' +
      'For what this profile serves once applied · save as ~/.config/opencode/opencode.json</span></div>' +
      '<pre class="profile-opencode">' + this.esc(text) + '</pre>';
    slot.querySelectorAll('[data-copy]').forEach(function (b) {
      b.addEventListener('click', function () {
        navigator.clipboard.writeText(b.dataset.copy).then(function () {
          self.toast('Config copied', 'success');
        }).catch(function () { self.toast('Copy failed', 'error'); });
      });
    });
  },

  async setDefaultProfile(name, clear) {
    var resp = await fetch('/api/profiles/' + encodeURIComponent(name) + '/default', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clear ? { default: false } : {}),
    });
    var data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      this.toast(data.error || 'Could not change the default', 'error');
      return;
    }
    this.toast(data.default ? '"' + data.default + '" is loaded at startup' : 'No profile is loaded at startup',
               'success');
    this.renderProfiles();
  },

  async deleteProfile(name) {
    if (!confirm('Delete the profile "' + name + '"?')) return;
    // Its models may be running. Stopping them is a separate decision, and
    // stops only what this profile placed — nothing else on those nodes.
    var stop = confirm('Also stop the models of "' + name + '" on the nodes it uses?\n\nOK = stop them · Cancel = leave them running');
    var resp = await fetch('/api/profiles/' + encodeURIComponent(name) + (stop ? '?stop=1' : ''),
                           { method: 'DELETE' });
    if (!resp.ok) {
      this.toast('Could not delete "' + name + '"', 'error');
      return;
    }
    this.toast('Deleted "' + name + '"', 'info');
    this.renderProfiles();
  },

  // ========================================================================
  //  NAVIGATION
  // ========================================================================

  bindNav() {
    var self = this;
    // Top nav pills: Dashboard, Downloads
    document.querySelectorAll('.nav-pill').forEach(function (el) {
      el.addEventListener('click', function () {
        var view = el.dataset.view;
        if (view) self.navigate(view);
      });
    });
  },

  navigate(view) {
    var prevView = this.state.currentView;
    this.state.currentView = view;
    // Reset downloads-view init flag when leaving the view, so we do a full render next time
    if (prevView === 'downloads' && view !== 'downloads') {
      this._downloadsViewInitialized = false;
    }
    if (view === 'downloads' && prevView !== 'downloads') {
      this._downloadsViewInitialized = false;  // force full render on entry
    }
    if (view === 'training' && prevView !== 'training') {
      this._autodataViewInitialized = false;   // force full render of the AutoData panel on entry
    }
    if (view === 'config' && prevView !== 'config') {
      this._configViewInitialized = false;     // render once on entry, then leave the forms alone
    }
    // Update nav pill active state
    document.querySelectorAll('.nav-pill').forEach(function (el) {
      el.classList.toggle('active', el.dataset.view === view);
    });
    // Show/hide views in center-stage
    document.querySelectorAll('#center-stage > .view').forEach(function (el) {
      el.style.display = 'none';
    });
    var target = document.getElementById('view-' + view);
    if (target) target.style.display = '';
    // Context-switch left panel: training shows training sidebar, else chat sidebar.
    this.updateLeftPanelContext(view);
    // Context-switch right panel: server view swaps to Model Info
    this.updateRightPanelContext(view);
    // Start/stop Server log polling
    if (view === 'server') {
      this.startServerLogPolling();
    } else {
      this.stopServerLogPolling();
    }
    // Draw the page now, then refresh the status behind it. Drawing was left
    // to refresh(), which first waited for five status requests and then —
    // because the click that got here counts as an interaction — skipped the
    // draw as "the user is busy". A page opened empty and filled in on the
    // next five-second tick: most visible on Config, which has nothing to
    // show until it is drawn.
    this._renderCurrentView(true);
    this.refresh();
  },

  updateRightPanelContext(view) {
    var def = document.getElementById('right-panel-default');
    var srv = document.getElementById('right-panel-server');
    if (!def || !srv) return;
    if (view === 'server') {
      def.style.display = 'none';
      srv.style.display = '';
    } else {
      def.style.display = '';
      srv.style.display = 'none';
    }
  },

  updateLeftPanelContext(view) {
    var chatSide = document.getElementById('left-panel-chat');
    var trainSide = document.getElementById('left-panel-training');
    if (!chatSide || !trainSide) return;
    if (view === 'training') {
      chatSide.style.display = 'none';
      trainSide.style.display = '';
      this.renderTrainingSidebar();
    } else {
      trainSide.style.display = 'none';
      chatSide.style.display = '';
    }
  },

  // ========================================================================
  //  SIGN-IN
  // ========================================================================

  // Every fetch in this file, not only fetchJSON: a session that expires while
  // the page is open should land on the sign-in page, not leave panels that
  // quietly stop updating. Wrapped once, at the source, so a fetch written next
  // month is covered without anyone remembering to.
  watchForSignOut() {
    if (window.__ainodeFetchWrapped) return;
    window.__ainodeFetchWrapped = true;
    var original = window.fetch.bind(window);
    window.fetch = async function (input, init) {
      var response = await original(input, init);
      if (response.status === 401) {
        var url = typeof input === 'string' ? input : (input && input.url) || '';
        if (url.indexOf('/api/') === 0 || url.indexOf(location.origin + '/api/') === 0) {
          var body = await response.clone().json().catch(function () { return {}; });
          if (body && body.login) {
            location.href = '/login?next=' + encodeURIComponent(location.pathname + location.search);
          }
        }
      }
      return response;
    };
  },

  async signOut() {
    await fetch('/api/logout', { method: 'POST' }).catch(function () {});
    location.href = '/login';
  },

  async checkDefaultPassword() {
    var status = await this.fetchJSON('/api/auth/web/status');
    this.state.webDefaultPassword = !!(status && status.is_default);
    this.state.webUser = (status && status.user) || '';
    this.renderDefaultPasswordBanner();
  },

  // On every page until it is changed. admin/admin on a node that holds the
  // docker socket is the one setting that should not be easy to live with.
  renderDefaultPasswordBanner() {
    var existing = document.getElementById('default-password-banner');
    if (!this.state.webDefaultPassword) {
      if (existing) existing.remove();
      return;
    }
    if (existing) return;
    var banner = document.createElement('div');
    banner.id = 'default-password-banner';
    banner.className = 'default-password-banner';
    banner.innerHTML = '⚠ This node still uses the default password <span class="mono">admin / admin</span>. ' +
      '<button class="config-btn" id="default-password-fix">Change it</button>';
    var shell = document.querySelector('.command-center') || document.body;
    shell.insertBefore(banner, shell.firstChild.nextSibling);
    var self = this;
    document.getElementById('default-password-fix').addEventListener('click', function () {
      // The section first: navigate('config') renders whichever is current.
      self.state.configSection = 'security';
      self.navigate('config');
    });
  },

  // ========================================================================
  //  DATA FETCHING
  // ========================================================================

  async fetchJSON(url) {
    try {
      var resp = await fetch(url);
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      return null;
    }
  },

  async refresh() {
    var self = this;
    var results = await Promise.all([
      this.fetchJSON('/api/status'),
      this.fetchJSON('/api/nodes'),
      this.fetchJSON('/api/sharding/status'),
      this.fetchJSON('/api/cluster/resources'),
      this.fetchJSON('/v1/models'),   // federated fleet union (F1) — every node's model
    ]);
    this.state.status = results[0];
    this.state.nodes = results[1]?.nodes || [];
    this.state.shardingStatus = results[2];
    this.state.clusterResources = results[3];
    // Fleet-wide loaded models: ids from the federated /v1/models union, so the
    // chat dropdown + INSTANCES panel see models on ALL nodes, not just local.
    this.state.fleetModels = ((results[4] && results[4].data) || []).map(function (m) { return m.id; });

    // An update in flight is a property of the node, not of the tab that
    // started it: /api/status carries it, and every page polls that.
    this.state.updateRunning = !!(results[0] && results[0].update_running);
    this.renderUpdateBanner();
    this.updateTopBar();
    this.updateClusterHero();
    this.updateChatModelSelect();

    // Don't rebuild the view + right panel out from under an active interaction
    // (dragging a node pill, selecting text to copy, a mid-click) — that 5s flicker
    // was wiping clicks/selection. State is already updated above, so the next idle
    // tick renders fresh.
    if (this._userBusy()) return;

    // Preserve scroll position of the main content area across periodic
    // re-renders — the 5s poll was rebuilding innerHTML and bouncing the
    // user back to the top of long lists.
    var mainEl = document.querySelector('.main-content') || document.querySelector('#main') || document.scrollingElement;
    var savedScroll = mainEl ? mainEl.scrollTop : 0;

    // Redrawn through _redraw: not while someone is typing in the view, and
    // without losing what was typed or produced there.
    this._redraw(document.getElementById('center-stage'),
                 function () { return self._renderCurrentView(false); });

    // Always update right panel
    this.renderInstances();
    this.populateLaunchModels();
    // Keep the distributed/solo hint in sync with the latest cluster state.
    if (typeof this._renderNodeDots === 'function') {
      try { this._renderNodeDots(); } catch (_) {}
    }
    if (typeof this._launchHintUpdater === 'function') {
      try { this._launchHintUpdater(); } catch (_) {}
    }

    // Restore scroll
    if (mainEl && savedScroll) {
      try { mainEl.scrollTop = savedScroll; } catch (_) {}
    }
  },

  // True while the user is actively interacting, so the periodic poll doesn't
  // rebuild the DOM mid-action (drag / text-selection / just-clicked).
  // Draw the current view. ``entering``: the operator just navigated here —
  // views that are drawn once (config, downloads) are drawn now.
  _renderCurrentView(entering) {
    var done;
    if (entering) {
      if (this.state.currentView === 'config') this._configViewInitialized = false;
      if (this.state.currentView === 'downloads') this._downloadsViewInitialized = false;
    }
    switch (this.state.currentView) {
      case 'dashboard':
        done = this.renderDashboard();
        this.renderUpdateBanner();
        break;
      case 'downloads':
        // Don't rebuild the downloads view during periodic refresh — just update the queue
        // in place. Only do a full render when the user lands on the view.
        if (this._downloadsViewInitialized) {
          done = this.renderDownloadsQueue();
          this.updateNavDownloadBadge();
        } else {
          this._downloadsViewInitialized = true;
          done = this.renderDownloads();
        }
        break;
      case 'training':
        done = this.renderTraining();
        break;
      case 'images':
        done = this.renderImages();
        break;
      case 'config':
        // Rendered on entry and on a section switch, NOT on every poll.
        // Rebuilding the section's innerHTML every few seconds destroyed
        // whatever was half-typed into it — filling in the MQTT broker,
        // username and password was close to impossible, and the same applied
        // to every other form in here. Nothing on these pages ticks; a live
        // status is worth less than being able to type.
        if (!this._configViewInitialized) {
          this._configViewInitialized = true;
          done = this.renderConfig();
        }
        break;
      case 'server':
        done = this.renderServer();
        break;
      case 'profiles':
        done = this.renderProfiles();
        break;
    }

    return done;
  },

  // An editable field inside ``root`` has the focus: someone is typing there.
  _typingIn(root) {
    var el = document.activeElement;
    if (!root || !el || el === document.body || !root.contains(el)) return false;
    var tag = (el.tagName || '').toLowerCase();
    if (el.isContentEditable || tag === 'textarea' || tag === 'select') return true;
    if (tag !== 'input') return false;
    return ['button', 'submit', 'reset', 'checkbox', 'radio', 'range', 'file',
            'image', 'color'].indexOf(String(el.type || '').toLowerCase()) === -1;
  },

  // What has been typed or ticked inside ``root`` and not saved: every field
  // with an id whose value differs from the one the page was drawn with.
  _snapshotEdits(root) {
    var edits = [];
    if (!root) return edits;
    root.querySelectorAll('input[id], textarea[id], select[id]').forEach(function (el) {
      var type = String(el.type || '').toLowerCase();
      if (type === 'checkbox' || type === 'radio') {
        if (el.checked !== el.defaultChecked) edits.push({ id: el.id, checked: el.checked });
      } else if (el.tagName === 'SELECT') {
        var changed = Array.prototype.some.call(el.options, function (o) {
          return o.selected !== o.defaultSelected;
        });
        if (changed) edits.push({ id: el.id, value: el.value });
      } else if (type !== 'file' && el.value !== el.defaultValue) {
        edits.push({ id: el.id, value: el.value });
      }
    });
    return edits;
  },

  _restoreEdits(edits) {
    (edits || []).forEach(function (edit) {
      var el = document.getElementById(edit.id);
      if (!el) return;
      if ('checked' in edit) {
        // Only where the redraw left the default: a value the server just
        // changed wins over one typed before it changed.
        if (el.checked === el.defaultChecked) el.checked = edit.checked;
      } else if (el.value === el.defaultValue || el.tagName === 'SELECT') {
        el.value = edit.value;
      }
    });
  },

  // Redraw part of the page without taking anything from the person using it.
  //
  // The periodic refresh rebuilt whole views with innerHTML every five
  // seconds. Reported from the Profiles page:
  //
  //     wenn ich unter profiel die aktuelle konfiguration speichern will und
  //     oben einen namen dafür eingeben will wird das feld ständig geleert
  //
  // and it had happened before (the MQTT form, fixed for Config alone). The
  // Server view lost the OpenCode config it had just generated the same way,
  // and the Images, Training and benchmark panels had the same shape. So,
  // for every redraw instead of per page:
  //   * nothing is redrawn while a field inside ``root`` has the focus;
  //   * what was typed into a field (by id) is carried into the new HTML;
  //   * an element marked data-keep (with an id) — a generated result, a
  //     report — is carried over as the same node, listeners and all.
  // ``draw`` may return a promise; the carrying-over waits for it.
  _redraw(root, draw) {
    if (this._typingIn(root)) return false;
    var edits = this._snapshotEdits(root);
    var kept = [];
    if (root) {
      root.querySelectorAll('[data-keep][id]').forEach(function (el) {
        if (el.childNodes.length) kept.push(el);
      });
    }
    var self = this;
    var finish = function () {
      kept.forEach(function (old) {
        var fresh = document.getElementById(old.id);
        if (fresh && fresh !== old && !fresh.childNodes.length) fresh.replaceWith(old);
      });
      self._restoreEdits(edits);
    };
    var done;
    try { done = draw(); } catch (err) { finish(); throw err; }
    if (done && typeof done.then === 'function') {
      done.then(finish, finish);
    } else {
      finish();
    }
    return true;
  },

  _userBusy() {
    if (this._pointerDown) return true;
    if (this._lastInteract && (Date.now() - this._lastInteract) < 1200) return true;
    try { if (String(window.getSelection())) return true; } catch (_) {}
    return false;
  },

  startPolling() {
    var self = this;
    // Track active interaction so the poll won't rebuild the DOM out from under it.
    if (!this._interactionWired) {
      this._interactionWired = true;
      document.addEventListener('pointerdown', function () { self._pointerDown = true; self._lastInteract = Date.now(); }, true);
      document.addEventListener('pointerup', function () { self._pointerDown = false; self._lastInteract = Date.now(); }, true);
    }
    // Status + nodes every 5s
    this.state.pollInterval = setInterval(function () { self.refresh(); }, 5000);
    // Metrics every 3s
    this.state.metricsInterval = setInterval(function () { self.pollMetrics(); }, 3000);
    // Version check every 30 minutes
    this.checkVersion();
    this.state.versionInterval = setInterval(function () { self.checkVersion(); }, 30 * 60 * 1000);
    // Are we behind our own fork's branch? Hourly, which is often enough for
    // something that moves a few times a day and rare enough to stay well
    // inside GitHub's unauthenticated rate limit with three nodes behind one
    // address.
    this.checkForSourceUpdate(false);
    this.state.sourceUpdateInterval = setInterval(
      function () { self.checkForSourceUpdate(false); }, 60 * 60 * 1000);
    // Initial fetch
    this.refresh();
  },

  checkVersion() {
    var self = this;
    fetch('/api/version/check')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        self.state.versionInfo = data;
        self.renderVersionBadge();
      })
      .catch(function () {});
  },

  initClusterUpdateBtn() {
    var self = this;
    var btn = document.getElementById('cluster-update-all-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var nodeCount = (self.state.nodes || []).length || 1;
      if (!confirm('Update all ' + nodeCount + ' node(s) to the latest AINode image?\n\nEach node will docker pull + restart. The master updates last.')) return;
      self.runClusterUpdate();
    });
  },

  runClusterUpdate() {
    var self = this;
    var btn = document.getElementById('cluster-update-all-btn');
    var panel = document.getElementById('cluster-update-panel');
    if (btn) { btn.disabled = true; btn.textContent = 'Updating...'; }
    if (panel) { panel.style.display = ''; panel.innerHTML = '<div class="cluster-update-row"><span class="cu-spinner">⟳</span> Starting update...</div>'; }

    fetch('/api/cluster/update-all', { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          self.toast('Update failed: ' + data.error, 'error');
          if (btn) { btn.disabled = false; btn.textContent = '⬆ Update all'; }
          return;
        }
        var updateId = data.update_id;
        self._pollClusterUpdate(updateId);
      })
      .catch(function () {
        self.toast('Update request failed', 'error');
        if (btn) { btn.disabled = false; btn.textContent = '⬆ Update all'; }
      });
  },

  _pollClusterUpdate(updateId) {
    var self = this;
    var panel = document.getElementById('cluster-update-panel');
    var btn = document.getElementById('cluster-update-all-btn');

    fetch('/api/cluster/update-status?id=' + updateId)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!panel) return;

        // Render per-node status
        var rows = Object.entries(data.nodes || {}).map(function (entry) {
          var nid = entry[0], n = entry[1];
          var icon = n.status === 'done' ? '✅' :
                     n.status === 'failed' ? '❌' :
                     n.status === 'updating' ? '<span class="cu-spinner">⟳</span>' : '⏳';
          return '<div class="cluster-update-row">' + icon + ' <b>' + self.esc(n.node_name || nid) + '</b> — ' + self.esc(n.status) +
                 (n.log ? ' <span class="cu-log">' + self.esc(n.log.slice(0, 80)) + '</span>' : '') + '</div>';
        }).join('');

        panel.innerHTML = rows;

        if (data.status === 'complete') {
          var allOk = Object.values(data.nodes).every(function (n) { return n.status === 'done'; });
          if (btn) { btn.disabled = false; btn.textContent = '⬆ Update all'; }
          self.toast(allOk ? 'All nodes updated ✅' : 'Update complete — some nodes failed', allOk ? 'success' : 'error');
          // Clear panel after 10s
          setTimeout(function () {
            if (panel) panel.style.display = 'none';
            self.checkVersion(); // refresh version badge
          }, 10000);
        } else {
          // Still running — poll again in 3s
          setTimeout(function () { self._pollClusterUpdate(updateId); }, 3000);
        }
      })
      .catch(function () {
        setTimeout(function () { self._pollClusterUpdate(updateId); }, 5000);
      });
  },

  renderVersionBadge() {
    var info = this.state.versionInfo;
    var self = this;

    // Show/hide the cluster "Update all" button based on update availability
    var clusterBtn = document.getElementById('cluster-update-all-btn');
    if (clusterBtn) {
      if (info && info.update_available) {
        clusterBtn.style.display = '';
        clusterBtn.title = 'Update all nodes to v' + info.latest;
        clusterBtn.textContent = '⬆ Update all  v' + info.latest;
      } else {
        clusterBtn.style.display = 'none';
      }
    }

    // Remove existing top-bar badge
    var existing = document.getElementById('update-badge');
    if (existing) existing.remove();

    if (!info || !info.update_available) return;

    // Inject update badge into top bar
    var topBar = document.querySelector('.top-bar') || document.querySelector('nav');
    if (!topBar) return;

    var badge = document.createElement('button');
    badge.id = 'update-badge';
    badge.className = 'update-badge';
    badge.innerHTML = '⬆ Update available: v' + info.latest;
    badge.title = 'Click to update to v' + info.latest;
    badge.addEventListener('click', function () {
      // Fleet-wide update: POST /api/cluster/update-all (resolves one target tag
      // and threads it to every peer, master last) instead of the head-only
      // /api/engine/update. F4.
      var nodeCount = (self.state.nodes || []).length || 1;
      var nodeWord = nodeCount === 1 ? 'node' : 'nodes';
      if (!confirm('Update all ' + nodeCount + ' ' + nodeWord + ' to v' + info.latest + '? Services restart one by one.')) return;
      badge.textContent = 'Updating...';
      badge.disabled = true;
      fetch('/api/cluster/update-all', { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data && data.error) {
            badge.textContent = '⬆ Update available: v' + info.latest;
            badge.disabled = false;
            self.toast('Update failed: ' + data.error, 'error');
            return;
          }
          self.toast('Fleet update started — nodes restart one by one', 'success');
          badge.textContent = 'Updating fleet...';
          // Surface live per-node progress in the cluster panel if it's mounted.
          if (data && data.update_id) self._pollClusterUpdate(data.update_id);
          // Re-check version in 90 seconds (bumped from 60s — a rolling fleet
          // restart takes longer than a single-node one).
          setTimeout(function () { self.checkVersion(); }, 90000);
        })
        .catch(function () {
          badge.textContent = '⬆ Update available: v' + info.latest;
          badge.disabled = false;
          self.toast('Update failed — try ainode update from terminal', 'error');
        });
    });
    topBar.appendChild(badge);
  },

  async pollMetrics() {
    var data = await this.fetchJSON('/api/metrics');
    if (data) this.state.metrics = data;
  },

  // ========================================================================
  //  TOP BAR STATUS
  // ========================================================================

  updateTopBar() {
    var nodes = this.state.nodes;
    var s = this.state.status;
    var onlineCount = 0;
    if (s && s.engine_ready) onlineCount = 1;
    onlineCount = Math.max(onlineCount, nodes.filter(function (n) {
      return n.status === 'online' || n.status === 'serving' || n.engine_ready;
    }).length);

    var pill = document.querySelector('.top-bar-status');
    var countEl = document.getElementById('top-node-count');
    var labelEl = document.querySelector('.top-bar-status .node-label');

    if (!pill) return;

    if (onlineCount > 0) {
      pill.classList.remove('offline');
      pill.classList.add('online');
      if (countEl) countEl.textContent = onlineCount;
      if (labelEl) labelEl.textContent = onlineCount === 1 ? 'node online' : 'nodes online';
    } else {
      pill.classList.remove('online');
      pill.classList.add('offline');
      if (countEl) countEl.textContent = '0';
      if (labelEl) labelEl.textContent = 'offline';
    }
  },

  // ========================================================================
  //  CLUSTER HERO PILL (aggregated VRAM/GPUs)
  // ========================================================================

  updateClusterHero() {
    var r = this.state.clusterResources;
    var pill = document.getElementById('cluster-hero-pill');
    if (!pill) return;
    if (!r || !r.total_nodes) { pill.style.display = 'none'; return; }
    pill.style.display = '';
    var nodesEl = document.getElementById('hero-nodes');
    var vramEl = document.getElementById('hero-vram');
    var gpusEl = document.getElementById('hero-gpus');
    if (nodesEl) nodesEl.textContent = r.total_nodes + (r.total_nodes === 1 ? ' node' : ' nodes');
    if (vramEl) vramEl.textContent = Math.round(r.total_vram_gb) + ' GB VRAM';
    if (gpusEl) gpusEl.textContent = r.total_gpus + (r.total_gpus === 1 ? ' GPU' : ' GPUs');

    // Also surface on the Server view top bar (if the element exists).
    var srvEl = document.getElementById('server-cluster-summary');
    if (srvEl) {
      srvEl.textContent = 'Cluster: ' + r.total_nodes + ' nodes · '
        + Math.round(r.total_vram_gb) + ' GB VRAM · ' + r.total_gpus + ' GPUs';
    }
  },

  // ========================================================================
  //  DASHBOARD (center = topology canvas)
  // ========================================================================

  renderDashboard() {
    var nodes = this.state.nodes;
    var s = this.state.status;
    if (!s) return;

    // Update topology
    if (this.state.topology) {
      var topoNodes = nodes.length > 0 ? nodes : [{
        node_id: s.node_id || 'local',
        node_name: s.node_name || 'This Node',
        gpu_name: s.gpu?.name || 'GPU',
        gpu_memory_gb: s.gpu?.memory_total_mb ? (s.gpu.memory_total_mb / 1024).toFixed(1) : 0,
        model: s.model || '',
        status: s.engine_ready ? 'online' : (s.model ? 'starting' : 'online'),
        version: s.version || '',
        api_port: s.api_port || 8000,
        engine_ready: s.engine_ready,
      }];

      // MEMBERSHIP — stamp model + TP=N onto participating nodes from the
      // authoritative distributed_instance (cluster/resources). Keep any
      // distinct per-node model; never overwrite it. Solo mode (di null) no-op.
      var di = this.state.clusterResources && this.state.clusterResources.distributed_instance;
      if (di && di.model) {
        // Members are EXACTLY the head + this instance's peers (resolved to
        // node_ids server-side). Don't mark every 'member'-mode node — that
        // over-counts idle members not in this instance.
        var memberIds = di.peer_node_ids || di.peer_ips || [];
        topoNodes = topoNodes.map(function (n) {
          var participates =
            n.node_id === di.head_node_id ||
            memberIds.indexOf(n.node_id) !== -1;
          if (participates) {
            n.model = n.model || di.model;
            n.tp_size = di.tensor_parallel_size;
          }
          return n;
        });
      }

      // VRAM — merge this node's live GPU metrics so the ring shows real %.
      // Only mutates the local node; remote nodes keep their static totals.
      var gm = this.state.metrics && this.state.metrics.gpu;
      if (gm && !gm.error && gm.memory_total_mb > 0) {
        var localId = s.node_id;
        topoNodes.forEach(function (n) {
          if (n.node_id === localId || topoNodes.length === 1) {
            n.gpu_memory_used_pct = Math.round((gm.memory_used_mb / gm.memory_total_mb) * 100);
            n.gpu_utilization = gm.utilization_percent;
            n.gpu_temp = gm.temperature_c;
          }
        });
      }
      // Pass engine_ready so the topology can drive the loading → real transition.
      // If no model is configured, the server is ready (no engine to wait for).
      var engineReady = !!(s && (s.engine_ready || !s.model));
      this.state.topology.update(topoNodes, engineReady);
    }
  },

  // ========================================================================
  //  RIGHT PANEL — INSTANCES
  // ========================================================================

  // ========================================================================
  //  ERROR ASSISTANT
  // ========================================================================
  // The raw error keeps its place on the card. This adds, underneath it, what
  // a model that is ALREADY RUNNING makes of it — with the launch settings,
  // the cluster state and a filtered engine log as context, because no public
  // model has heard of AINode or of this hardware. Nothing is downloaded and
  // no helper model is bundled: if nothing is serving, there is no button.

  // Where a load's minutes went. The phases were always detected; what was
  // missing was a clock on them — so "it takes five minutes" could never be
  // answered with anything but "yes". Shown while loading (the running phase
  // counts up) and kept afterwards, because the question is usually asked
  // once the model is already up.
  loadTimelineBlock(inst) {
    var timeline = inst.timeline || [];
    if (!timeline.length) return '';
    var parts = timeline.map(function (entry) {
      return this.esc(this.phaseLabel(entry.phase)) + ' ' +
             this.formatSeconds(entry.seconds);
    }, this).join(' · ');
    var total = inst.loadSeconds || timeline.reduce(function (sum, e) {
      return sum + (e.seconds || 0);
    }, 0);
    var lead = inst.status === 'READY'
      ? 'loaded in ' + this.formatSeconds(total)
      : this.formatSeconds(total) + ' so far';
    return '<div class="load-timeline"><strong>' + this.esc(lead) + '</strong> · ' +
           parts + '</div>';
  },

  phaseLabel(phase) {
    return ({
      starting: 'container + engine start',
      distributing: 'copying weights out',
      loading_weights: 'reading weights',
      distributed_init: 'forming the cluster',
      profiling: 'compiling + sizing the cache',
      ready: 'serving',
    })[phase] || phase;
  },

  formatSeconds(seconds) {
    return AINodeLib.formatSeconds(seconds);
  },

  assistBlock(inst, errorText, readyModels) {
    var model = inst.model || '';
    this._assistErrors = this._assistErrors || {};
    this._assistErrors[model] = errorText;
    this._assistNodes = this._assistNodes || {};
    this._assistNodes[model] = (inst.nodes && inst.nodes[0]) || '';

    var entry = (this.state.assist || {})[model];
    var helpers = (readyModels || []).filter(function (m) { return m !== model; });

    if (!entry) {
      if (!helpers.length) return '';
      return '<div class="assist-row">' +
        '<button class="btn-ghost server-btn-sm" data-assist="' + this.esc(model) +
        '" title="Sends this error plus the launch settings, the cluster state ' +
        'and the filtered engine log to a model that is already running.">' +
        'EXPLAIN THIS ERROR</button>' +
        '<span class="assist-hint">asks ' + this.esc(helpers[0]) + '</span></div>';
    }
    if (entry.status === 'loading') {
      return '<div class="assist-row"><span class="assist-hint">Asking ' +
        this.esc(entry.helper || 'a loaded model') + '…</span></div>';
    }
    if (entry.status === 'error') {
      return '<div class="assist-row"><span class="assist-hint warn">' +
        this.esc(entry.error) + '</span>' +
        '<button class="btn-ghost server-btn-sm" data-assist="' + this.esc(model) +
        '">TRY AGAIN</button></div>';
    }
    return '<div class="assist-answer">' +
      '<div class="assist-caption">Suggested by ' + this.esc(entry.helper) +
      ' — it was given this error, the launch settings, the state of every ' +
      'node and ' + (entry.logLines || 0) + ' lines of engine log. ' +
      'It can be wrong; the error above is the record.</div>' +
      '<div class="assist-text">' + this.esc(entry.answer) + '</div>' +
      '<div class="assist-row">' +
      '<button class="btn-ghost server-btn-sm" data-assist="' + this.esc(model) +
      '">ASK AGAIN</button>' +
      '<button class="btn-ghost server-btn-sm" data-assist-dismiss="' +
      this.esc(model) + '">DISMISS</button></div></div>';
  },

  async askAssistant(model) {
    if (!model) return;
    this.state.assist = this.state.assist || {};
    this.state.assist[model] = { status: 'loading' };
    this.renderInstances();
    var nodeLabel = (this._assistNodes || {})[model] || '';
    var body = {
      model: model,
      node_id: this._nodeIdForLabel(nodeLabel),
      error: (this._assistErrors || {})[model] || '',
    };
    try {
      var resp = await fetch('/api/assist/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok || data.error) {
        this.state.assist[model] = {
          status: 'error',
          error: data.error || ('the assistant answered ' + resp.status),
        };
      } else {
        this.state.assist[model] = {
          status: 'done',
          answer: data.answer || '',
          helper: data.helper_model || '',
          logLines: (data.context_sent || {}).log_lines || 0,
        };
      }
    } catch (err) {
      this.state.assist[model] = { status: 'error', error: err.message };
    }
    this.renderInstances();
  },

  renderInstances() {
    var container = document.getElementById('instances-list');
    if (!container) return;
    var self = this;
    var s = this.state.status;
    var live = !!(s && s.engine_ready);
    var phase = (s && s.load_phase) || 'idle';
    // Both come from this node's status, so they describe the same launch.
    var loadError = (s && s.load_error) || '';
    // What a silent pre-launch step is doing. The engine-image pull and the
    // weight copy to a peer both run before the launcher writes a line, so
    // without this the card sits at "starting" for minutes with an empty log.
    var loadDetail = (s && s.load_detail) || '';
    // Coarse phase → [label, percent] for the launching card (3c).
    // Any phase missing here renders as the `idle` fallback — a flat 8% that
    // reads as a hang. Keep in step with LOAD_PHASE_ORDER in
    // ainode/engine/load_phase.py.
    var instances = [];

    // Distributed instance (authoritative from /api/cluster/resources) —
    // when the head broadcasts one, render it before anything else and
    // skip adding its model as a "single" duplicate.
    var cr = this.state.clusterResources;
    var distModel = null;
    if (cr && cr.distributed_instance && cr.distributed_instance.model) {
      var di = cr.distributed_instance;
      distModel = di.model;
      instances.push({
        model: di.model,
        strategy: 'distributed',
        tp_size: di.tensor_parallel_size,
        nodes: di.member_names || [di.head_node_name || di.head_node_id].concat(di.peer_node_ids || di.peer_ips || []),
        // The instance's own status, not this browser's node. `live` is
        // s.engine_ready — the readiness of whichever node the UI is pointed
        // at — so a model serving on a sub-node read STARTING because the
        // HEAD had no engine running, and a phase of "idle" drew 8%. An
        // instance from an older build carries no status and defaults to
        // serving, which is how those behaved before.
        status: (di.status || 'serving') === 'serving' ? 'READY' : 'STARTING',
        phase: di.status === 'failed' ? 'failed' : (di.load_phase || ''),
        error: di.load_error || '',
        detail: di.load_detail || '',
        // A single-node serve is SOLO even when the head advertises a
        // distributed_instance (it's configured with peers) — only badge
        // DISTRIBUTED when the model is actually split across nodes, on any
        // axis. parallel_label comes from the server ("TP=4", "PP=3"); the
        // fallback covers a head still running an older build.
        badge: (self.instanceWorldSize(di) > 1)
          ? ((di.degraded ? 'DEGRADED · ' : 'DISTRIBUTED · ')
             + (di.parallel_label || ('TP=' + di.tensor_parallel_size)))
          : 'SOLO · TP=1',
        degraded: !!di.degraded,
        missingPeers: di.missing_peer_ips || [],
        survivingNodeIds: di.surviving_node_ids || [],
      });
    }

    // Fleet-wide: one card per node serving a solo model, across ALL nodes
    // (not just local) — sourced from cluster /api/nodes so the panel matches
    // the federated /v1/models the chat dropdown uses. Skip the distributed one.
    var seen = {};
    (this.state.nodes || []).forEach(function (n) {
      // Show the friendly node NAME (Spark-2-DGX), not the raw node-id hex.
      // node_name is the id→name mapping carried on every /api/nodes entry;
      // fall back to hostname, then the id, when the name is unknown (F1).
      var host = n.node_name || n.hostname || n.node_id;
      var modelName = n.model;
      if (modelName && !(distModel && modelName === distModel)) {
        var key = modelName + '@' + (n.node_id || n.hostname);
        if (!seen[key]) {
          seen[key] = true;
          instances.push({
            model: modelName,
            strategy: 'single',
            nodes: [host],
            status: n.engine_ready ? 'READY' : 'STARTING',
            badge: 'SINGLE',
            phase: n.node_id === (self.state.status && self.state.status.node_id)
              ? phase : '',
            error: n.node_id === (self.state.status && self.state.status.node_id)
              ? loadError : '',
          });
        }
      }
      // Stacked instances (2nd+ model on this node, ports 8001+) — invisible
      // before D5. Render a card per stacked instance (model, port, status).
      (n.instances || []).forEach(function (inst) {
        var im = inst.model;
        if (!im || (distModel && im === distModel)) return;
        // The node's own main port is the primary, whatever n.model says. A
        // node blanks n.model until its engine answers (api/server.py, so a
        // dead engine never advertises a phantom model), so during a load the
        // "same model on the main port" test could not match and the primary
        // was drawn as a stacked card: "STACKED · :8000" on a node running
        // exactly one model. The port is the thing that actually decides it.
        var isPrimary = inst.api_port == null || inst.api_port === n.api_port;
        // Skip the primary only when the SINGLE card above already drew it.
        if (isPrimary && im === n.model) return;
        var skey = im + '@' + (n.node_id || n.hostname) + ':' + (inst.api_port || '');
        if (seen[skey]) return;
        seen[skey] = true;
        instances.push({
          model: im,
          strategy: isPrimary ? 'single' : 'stacked',
          nodes: [isPrimary ? host
            : host + (inst.api_port ? ':' + inst.api_port : '')],
          // Use the stacked instance's OWN status, not the node's primary
          // readiness. n.engine_ready reflects the PRIMARY engine (and for the
          // local node is hardcoded online → always true), so OR-ing it in made
          // a still-loading or dead stacked model read READY the moment the
          // primary was up. inst.status is the per-instance truth (the head
          // only advertises live instances, flipped to `serving`).
          status: inst.status === 'serving' ? 'READY' : 'STARTING',
          badge: isPrimary ? 'SINGLE'
            : 'STACKED' + (inst.api_port ? ' · :' + inst.api_port : ''),
          // Per-instance, from the record that crossed the wire. The node's
          // load_error is ONE value per node and used to be painted on every
          // card: two models failing on two different machines showed the same
          // message, down to the process id and the second.
          phase: inst.status === 'failed' ? 'failed' : (inst.load_phase || ''),
          error: inst.load_error || '',
          detail: inst.load_detail || '',
          timeline: inst.load_timeline || [],
          loadSeconds: inst.load_seconds || 0,
          kind: inst.kind || 'llm',
          api_port: inst.api_port || 0,
        });
      });
    });

    // Collect from sharding status (legacy — keep for pipeline/tensor runs
    // that don't come through the cluster/resources distributed_instance)
    var sharding = this.state.shardingStatus;
    if (sharding && sharding.active_sharding && sharding.active_sharding.model) {
      var sh = sharding.active_sharding;
      var shardNodes = sh.shard_map ? Object.keys(sh.shard_map) : [];
      var already = instances.find(function (inst) { return inst.model === sh.model; });
      if (!already) {
        instances.push({
          model: sh.model,
          strategy: sh.strategy || 'pipeline',
          nodes: shardNodes,
          status: live ? 'READY' : 'STARTING',
          badge: (sh.strategy || 'pipeline').toUpperCase(),
        });
      } else {
        already.strategy = sh.strategy || 'pipeline';
        already.nodes = shardNodes.length > 0 ? shardNodes : already.nodes;
      }
    }

    // Launching card (3c): a model is configured and the engine is spinning up
    // but hasn't registered as an instance yet — show its load phase so the
    // multi-minute launch doesn't read as "nothing running".
    if (instances.length === 0 && s && s.model && !live) {
      instances.push({
        model: s.model,
        strategy: 'launching',
        nodes: [s.node_id || 'local'],
        status: 'STARTING',
        badge: 'LAUNCHING',
        // This card has no instance record yet, so its timing comes from the
        // node's own status — which is the same engine.
        timeline: s.load_timeline || [],
        loadSeconds: s.load_seconds || 0,
      });
    }

    if (instances.length === 0) {
      container.innerHTML = '<div class="instances-empty">No running instances</div>';
      return;
    }

    // Who could explain a failure: any OTHER model that is answering right
    // now. No model loaded means no assistant and no button — AINode ships no
    // helper model of its own, and a cluster whose models all failed to start
    // is exactly the case where a bundled one would have failed too.
    var readyModels = instances.filter(function (i) {
      return i.status === 'READY' && i.model;
    }).map(function (i) { return i.model; });


    // The card says four things: what it is, how it is doing, and two
    // buttons. Everything else — the load timing, the full error text, the
    // assistant, the kernel cache, the relaunch, the launch parameters — is
    // one click away in Details. It had all accumulated on the card, because
    // every new piece of information needed somewhere to go, and the panel
    // that is looked at most often had become the least readable one.
    // Each card's own phase, error and detail — never the node's. A card
    // that carries the fields at all (every remote and stacked one) uses
    // ONLY its own, even when they are empty: falling back to the node's
    // painted a model still loading on another machine as READY, because the
    // head's own engine happened to be ready, and painted one node-level
    // error message on every card in the panel.
    instances.forEach(function (inst) {
      var hasOwnState = inst.phase !== undefined;
      inst.instPhase = hasOwnState
        ? (inst.phase || (inst.status === 'READY' ? 'ready' : 'starting'))
        : phase;
      inst.instError = hasOwnState ? (inst.error || '') : loadError;
      inst.instDetail = hasOwnState ? (inst.detail || '') : loadDetail;
      inst.readyModels = readyModels;
    });

    container.innerHTML = instances.map(function (inst, idx) {
      var state = self.instanceState(inst, inst.instPhase);
      return '<div class="instance-card" data-idx="' + idx + '">' +
        '<div class="instance-head">' +
        '<span class="instance-model" title="' + self.esc(inst.model) + '">' +
          self.esc(inst.model) + '</span>' +
        '<span class="instance-state ' + state.cls + '">' + self.esc(state.label) +
          '</span>' +
        '</div>' +
        '<div class="instance-footer">' +
        '<button class="btn-ghost server-btn-sm" data-details="' +
          self.esc(inst.model) + '">DETAILS</button>' +
        '<button class="instance-delete" data-model="' + self.esc(inst.model) +
          '">UNLOAD</button>' +
        '</div></div>';
    }).join('');

    container.querySelectorAll('[data-details]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.openInstanceDetails(btn.getAttribute('data-details'));
      });
    });
    container.querySelectorAll('.instance-delete').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.deleteInstance(btn.dataset.model);
      });
    });

    // Keep an open dialog current: a load's phases move, the assistant
    // answers late, and a dialog frozen at the moment it was opened would be
    // the one place in the UI that lies.
    this._instanceCache = {};
    instances.forEach(function (inst) { self._instanceCache[inst.model] = inst; });
    if (this._detailsFor && this._instanceCache[this._detailsFor]) {
      this.renderInstanceDetails(this._instanceCache[this._detailsFor]);
    }
  },

  // One word and one colour. The phase bar belongs to a load in progress;
  // everything else is a steady state.
  instanceState(inst, phase) {
    if (inst.degraded) return { label: 'DEGRADED', cls: 'degraded' };
    if (phase === 'failed') return { label: 'FAILED', cls: 'failed' };
    if (inst.status === 'READY') return { label: 'READY', cls: 'ready' };
    // Reached ready, but not answering any more: it stopped after it had
    // served. Drawing the phase here painted a full green bar labelled with
    // whatever the load was last doing — "SIZING THE KV CACHE · 100%" on a
    // model that had died — which reads as progress instead of as loss.
    if (phase === 'ready') return { label: 'STOPPED ANSWERING', cls: 'stopped' };
    var info = (INSTANCE_PHASE_INFO[phase] || ['starting', 10]);
    return { label: 'LOADING · ' + info[1] + '%', cls: 'loading' };
  },

  openInstanceDetails(model) {
    this._detailsFor = model;
    var inst = (this._instanceCache || {})[model];
    if (inst) this.renderInstanceDetails(inst);
  },

  closeInstanceDetails() {
    this._detailsFor = null;
    var modal = document.getElementById('instance-detail-modal');
    if (modal) modal.remove();
  },

  renderInstanceDetails(inst) {
    var self = this;
    var phase = inst.instPhase || '';
    var error = inst.instError || '';
    var state = this.instanceState(inst, phase);
    var nodes = (inst.nodes || []).map(function (n) { return self.esc(n); })
      .join(', ') || 'this node';

    var rows = [
      ['Status', state.label + (inst.instDetail ? ' — ' + inst.instDetail : '')],
      ['Nodes', nodes],
      ['Split', inst.badge || inst.strategy || 'single'],
    ];
    if (inst.api_port) rows.push(['Port', String(inst.api_port)]);

    var body = '<table class="instance-detail-table">' + rows.map(function (row) {
      return '<tr><th>' + self.esc(row[0]) + '</th><td>' + self.esc(row[1]) +
        '</td></tr>';
    }).join('') + '</table>';

    body += this.loadTimelineBlock(inst);

    if (phase === 'failed' && error) {
      // The raw error, verbatim and first. The trigger that used to draw a
      // second clear-cache button is kept as a sentence instead: the dialog
      // offers the clear below in every case, so what was missing is not the
      // button but the hint that THIS error is the kind it fixes.
      var kernelFault =
        /compile cache|illegal (instruction|memory access|address)|compiled kernel|cudaError/i.test(error);
      body += '<div class="instance-detail-section"><h4>Error</h4>' +
        '<div class="instance-failed-note">' + this.esc(error) +
        (kernelFault
          ? '<div class="assist-hint warn" style="margin-top:8px">This is the ' +
            'kind of fault a stale compiled kernel produces — Clear kernel ' +
            'cache below is the first thing to try.</div>'
          : '') +
        this.assistBlock(inst, error, inst.readyModels || []) + '</div></div>';
    }
    if (inst.degraded) {
      body += '<div class="instance-degraded">Lost ' +
        this.esc((inst.missingPeers || []).join(', ')) +
        ' — this instance cannot serve until it is relaunched on the ' +
        ((inst.survivingNodeIds || []).length || 1) + ' node(s) still online.' +
        '</div>';
    }

    var actions = '<div class="assist-row">';
    if (inst.degraded) {
      actions += '<button class="instance-relaunch btn-ghost server-btn-sm" ' +
        'data-model="' + this.esc(inst.model) + '">RELAUNCH</button>';
    }
    actions += '<button class="instance-cache-clear btn-ghost server-btn-sm" ' +
      'data-nodes="' + this.esc((inst.nodes || []).join(',')) +
      '">CLEAR KERNEL CACHE</button>' +
      '<button class="instance-delete btn-ghost server-btn-sm" data-model="' +
      this.esc(inst.model) + '">UNLOAD</button></div>';
    body += '<div class="instance-detail-section">' + actions + '</div>';

    var modal = document.getElementById('instance-detail-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'instance-detail-modal';
      modal.className = 'model-detail-modal-overlay';
      document.body.appendChild(modal);
      modal.addEventListener('click', function (e) {
        if (e.target === modal) self.closeInstanceDetails();
      });
    }
    modal.innerHTML = '<div class="model-detail-modal">' +
      '<div class="md-header"><div class="md-header-left">' +
      '<div class="md-title">' + this.esc(inst.model) + '</div></div>' +
      '<button class="md-close">×</button></div>' +
      '<div class="md-description">' + body + '</div></div>';
    modal.querySelector('.md-close').addEventListener('click', function () {
      self.closeInstanceDetails();
    });
    this.bindInstanceActions(modal);
  },

  // The actions that used to sit on the card. Bound against whatever root
  // they are drawn in, so the dialog and any future home for them agree.
  bindInstanceActions(root) {
    var self = this;
    root.querySelectorAll('[data-assist]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.askAssistant(btn.getAttribute('data-assist'));
      });
    });
    root.querySelectorAll('[data-assist-dismiss]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        delete (self.state.assist || {})[btn.getAttribute('data-assist-dismiss')];
        self.renderInstances();
      });
    });
    root.querySelectorAll('.instance-delete').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.closeInstanceDetails();
        self.deleteInstance(btn.dataset.model);
      });
    });
    root.querySelectorAll('.instance-relaunch').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.relaunchInstance(btn.dataset.model, btn);
      });
    });
    root.querySelectorAll('.instance-cache-clear').forEach(function (btn) {
      btn.addEventListener('click', async function (e) {
        e.stopPropagation();
        // Every node the instance runs on: the kernels are compiled per node,
        // and a label can carry a port the node id does not have.
        var labels = (btn.getAttribute('data-nodes') || '').split(',')
          .map(function (l) { return l.split(':')[0].trim(); })
          .filter(Boolean);
        if (!confirm('Clear the compiled-kernel cache on ' +
                     (labels.join(', ') || 'this node') + '?\n\n' +
                     'Unload the model first — the cache is read at launch. ' +
                     'The next launch recompiles, which takes a few minutes once.')) return;
        var label = btn.textContent;
        btn.textContent = 'Clearing…';
        btn.disabled = true;
        var freed = 0;
        var failed = [];
        for (var i = 0; i < (labels.length || 1); i++) {
          var nodeId = self._nodeIdForLabel(labels[i]);
          var resp = await fetch('/api/cluster/compile-cache', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(nodeId ? { node_id: nodeId } : {}),
          });
          var data = await resp.json().catch(function () { return {}; });
          if (!resp.ok || data.error) failed.push(labels[i] || 'this node');
          else freed += data.freed_mb || 0;
        }
        btn.textContent = label;
        btn.disabled = false;
        if (failed.length) {
          self.toast('Could not clear the cache on ' + failed.join(', '), 'error');
          return;
        }
        self.toast('Cleared ' + freed + ' MB on ' + (labels.length || 1) +
                   ' node(s) — relaunch the model', 'success');
      });
    });
  },

  // Re-run a degraded instance on the nodes still online. The server re-plans
  // the split for the smaller node set (a TP=4 instance down to 3 nodes comes
  // back as PP=3, not an impossible TP=3) and refuses with a reason when the
  // weights no longer fit — that refusal is the useful answer, so show it in
  // full rather than a generic failure.
  async relaunchInstance(model, btn) {
    if (btn) { btn.disabled = true; btn.textContent = 'RELAUNCHING...'; }
    try {
      var resp = await fetch('/api/sharding/relaunch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model }),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (resp.ok) {
        var how = (data.parallel_plan && data.parallel_plan.label) || '';
        this.toast('Relaunching ' + model + (how ? ' as ' + how : ''), 'success');
      } else {
        this.toast(data.error || ('Relaunch failed (' + resp.status + ')'), 'error');
      }
    } catch (err) {
      this.toast('Relaunch failed: ' + err, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'RELAUNCH'; }
      this.refresh();
    }
  },

  async deleteInstance(model) {
    try {
      var resp = await fetch('/api/models/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model }),
      });
      if (resp.ok) {
        this.toast('Instance stopped: ' + model, 'success');
        this.invalidate();
        this.refresh();
      } else {
        var data = await resp.json().catch(function () { return {}; });
        this.toast(data.error || 'Failed to stop instance', 'error');
      }
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  // ========================================================================
  //  UTILITIES
  // ========================================================================

  _embedNodeSelect() {
    // Where an embedding model should run. Defaults to this node, which is
    // what happened implicitly before and stays the least surprising choice.
    var own = (this.state.status && this.state.status.node_id) || '';
    var options = ['<option value="">this node</option>'];
    (this.state.nodes || []).forEach(function (n) {
      if (!n.node_id || n.node_id === own) return;
      var label = n.node_name || n.hostname || n.node_id;
      options.push('<option value="' + n.node_id + '">' + label + '</option>');
    });
    return '<select id="embed-node" class="mono" ' +
      'style="padding:5px 6px;background:var(--bg-input,#111);color:inherit;' +
      'border:1px solid var(--border,#333);border-radius:4px">' +
      options.join('') + '</select>';
  },

  _nodeIdForLabel(label) {
    // Cards show the friendly name (and sometimes name:port); the API wants
    // the id. An unknown label means the local node, which is the right
    // default for a card with no node information.
    if (!label) return '';
    var name = String(label).split(':')[0];
    var match = (this.state.nodes || []).find(function (n) {
      return n.node_name === name || n.hostname === name || n.node_id === name;
    });
    return match ? match.node_id : '';
  },

  esc(str) {
    return AINodeLib.esc(str);
  },

  formatUptime(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
  },

  formatMarkdown(text) {
    if (!text) return '';
    var self = this;
    var placeholders = [];
    var codeBlockIndex = 0;

    // 1. Extract fenced code blocks first (so inner content isn't re-parsed)
    var html = text.replace(/```(\w*)\n?([\s\S]*?)```/g, function (match, lang, code) {
      var language = (lang || 'plaintext').toLowerCase();
      var escapedCode = self.esc(code.replace(/\n$/, ''));
      var id = 'code-' + Date.now() + '-' + (codeBlockIndex++);
      var block =
        '<div class="code-block-wrapper">' +
          '<div class="code-block-header">' +
            '<span class="code-block-lang">' + self.esc(language) + '</span>' +
            '<button class="code-copy-btn" data-code-id="' + id + '" title="Copy code">' +
              '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">' +
                '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>' +
                '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>' +
              '</svg>' +
              '<span>Copy</span>' +
            '</button>' +
          '</div>' +
          '<pre class="code-block"><code id="' + id + '" class="language-' + self.esc(language) + '">' + escapedCode + '</code></pre>' +
        '</div>';
      placeholders.push(block);
      return '\u0001CB' + (placeholders.length - 1) + '\u0001';
    });

    // 2. Extract inline code
    html = html.replace(/`([^`\n]+)`/g, function (m, code) {
      placeholders.push('<code class="inline-code">' + self.esc(code) + '</code>');
      return '\u0001IC' + (placeholders.length - 1) + '\u0001';
    });

    // 3. Escape remaining HTML
    html = self.esc(html);

    // 4. Markdown inline formatting
    html = html
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/~~(.+?)~~/g, '<del>$1</del>');

    // 5. Auto-link URLs
    html = html.replace(
      /(https?:\/\/[^\s<]+[^\s<.,;:?!\)])/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>'
    );

    // 6. Markdown-style links [text](url)
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>'
    );

    // 7. Headers (#, ##, ###)
    html = html
      .replace(/^### (.+)$/gm, '<h4 class="chat-h">$1</h4>')
      .replace(/^## (.+)$/gm, '<h3 class="chat-h">$1</h3>')
      .replace(/^# (.+)$/gm, '<h2 class="chat-h">$1</h2>');

    // 8. Lists
    html = html.replace(/^(\s*)[-*] (.+)$/gm, '$1• $2');

    // 9. Newlines to <br> (but not inside code blocks which are already placeholders)
    html = html.replace(/\n/g, '<br>');

    // 10. Restore code blocks / inline code
    html = html.replace(/\u0001CB(\d+)\u0001/g, function (_, idx) {
      return placeholders[parseInt(idx)];
    });
    html = html.replace(/\u0001IC(\d+)\u0001/g, function (_, idx) {
      return placeholders[parseInt(idx)];
    });

    return html;
  },

  skeletonCards(n) {
    return Array(n).fill('<div class="skeleton-card"><div class="skeleton" style="height:48px;margin-bottom:8px"></div><div class="skeleton" style="height:14px;width:60%"></div></div>').join('');
  },

  gaugeColor(pct) {
    if (pct > 90) return '#ef4444';
    if (pct > 70) return '#f59e0b';
    return '#76b900';
  },

  tempColor(celsius) {
    if (celsius > 85) return '#ef4444';
    if (celsius > 70) return '#f59e0b';
    return '#76b900';
  },

};

// ========================================================================
//  BOOT
// ========================================================================

document.addEventListener('DOMContentLoaded', function () { AINode.init(); });
