/* AINode Command Center — the Server view.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  SERVER VIEW (LM Studio-style console)
  // ========================================================================

  _serverState: {
    logsSince: 0,
    logsPoll: null,
    autoScroll: true,
    selectedModelId: null,
    lastStatus: null,
    endpoints: null,
    endpointTab: 'openai',
    logs: [],
  },

  async renderServer() {
    var mount = document.getElementById('server-content');
    if (!mount) return;
    var self = this;

    // Three requests that do not depend on each other, so they go at once.
    // They used to run one after another — status (which itself probed every
    // local instance serially), then the endpoint catalog, then the model
    // catalog — and the page drew nothing until the last one landed.
    var needCatalog = !this._serverState.modelsCatalog;
    var results = await Promise.all([
      this.fetchJSON('/api/server/status'),
      needCatalog ? this.fetchJSON('/api/models') : Promise.resolve(null),
    ]);
    var status = results[0];
    this._serverState.lastStatus = status;
    // The endpoint catalog is a constant and now travels with the status it
    // belongs to; the separate request remains for older nodes.
    this._serverState.endpoints = (status && status.endpoints)
      || this._serverState.endpoints
      || await this.fetchJSON('/api/server/endpoints');
    // Raw catalog (size_gb / local_size_gb / architecture) for MODEL INFO — the
    // loaded-model object lacks size, and this.state.catalog (live-catalog view)
    // isn't loaded here and remaps the fields.
    if (needCatalog) {
      this._serverState.modelsCatalog = (results[1] && results[1].models) || [];
    }

    var s = status || { status: 'stopped', reachable_at: [], loaded_models: [] };
    var primaryUrl = (s.reachable_at && (s.reachable_at[1] || s.reachable_at[0])) || '—';

    var html = '';

    // --- Top status bar ---
    html += '<div class="server-status-bar">';
    html += '  <div class="server-status-left">';
    html += '    <span class="server-status-indicator"><span class="server-status-dot running"></span> Running</span>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-toggle">Stop</button>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-settings-btn">Server Settings</button>';
    html += '    <button class="btn-ghost server-btn-sm" id="server-mcp-btn">mcp.json</button>';
    html += '  </div>';
    html += '  <div class="server-status-center">';
    html += '    <span class="server-reachable-label">Reachable at:</span>';
    html += '    <span class="server-reachable-url mono" id="server-reachable-url">' + this.esc(primaryUrl) + '</span>';
    html += '    <button class="server-copy-btn" data-copy="' + this.esc(primaryUrl) + '" title="Copy">⧉</button>';
    html += '    <span class="server-cluster-summary mono" id="server-cluster-summary" style="margin-left:16px;color:#76B900"></span>';
    html += '  </div>';
    html += '  <div class="server-status-right">';
    html += '    <button class="btn-nvidia server-btn-sm" id="server-load-model">+ Load Model</button>';
    html += '  </div>';
    html += '</div>';

    // --- Loaded Models ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Loaded Models</h3>';
    html += '    <span class="server-section-meta">' + ((s.loaded_models || []).length) + ' loaded · ' + this.esc(this.formatUptime(Math.floor(s.uptime_seconds || 0))) + ' uptime</span>';
    html += '  </div>';
    if (!s.loaded_models || s.loaded_models.length === 0) {
      html += '  <div class="server-empty">No models loaded — click <strong>+ Load Model</strong> to start one</div>';
    } else {
      html += '  <div class="server-loaded-list" id="server-loaded-list">';
      s.loaded_models.forEach(function (m, idx) {
        html += self._renderLoadedCard(m, idx);
      });
      html += '  </div>';
    }
    html += '</section>';

    // --- Supported Endpoints ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Supported endpoints</h3>';
    html += '    <span class="badge-new">NEW REST API v1</span>';
    html += '  </div>';
    html += '  <div class="server-tab-pills" id="server-endpoint-tabs">';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'openai' ? ' active' : '') + '" data-tab="openai">OpenAI-compatible</button>';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'lmstudio' ? ' active' : '') + '" data-tab="lmstudio">LM Studio API</button>';
    html += '    <button class="server-tab-pill' + (this._serverState.endpointTab === 'anthropic' ? ' active' : '') + '" data-tab="anthropic">Anthropic-compatible</button>';
    html += '  </div>';
    html += '  <div class="server-endpoints-list" id="server-endpoints-list">';
    html += this._renderEndpointRows(primaryUrl);
    html += '  </div>';
    // A ready-made client config. The three settings that decide whether a
    // coding session works — does the model reason, does it take images, and
    // how much context was it LAUNCHED with — are per-instance, and getting
    // any of them from the model's advertised figures produces a config that
    // works until it quietly does not.
    html += '  <div style="margin-top:14px;display:flex;gap:10px;align-items:center">';
    html += '    <button class="btn-ghost server-btn-sm" id="opencode-config">' +
            'Generate OpenCode config</button>';
    html += '    <span style="font-size:11px;color:var(--text-muted)">' +
            'for every model the cluster is serving right now</span>';
    html += '  </div>';
    html += '  <div id="opencode-stale">' + ((this._opencodeDrift || {}).html || '') + '</div>';
    html += '  <div id="opencode-config-out" data-keep></div>';
    html += '</section>';

    // --- Throughput benchmark ---
    // Capacity on this hardware is not something to derive from parameter
    // counts: a 26B MoE measured 70 tok/s alone and 544 across 16 streams,
    // while a 230B across two nodes managed 20.5 where its catalog entry
    // claimed 42. The cluster answers "how many users" better than any
    // estimate, so the measurement belongs here rather than in a script.
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Throughput benchmark</h3>';
    html += '    <span style="font-size:12px;color:var(--text-muted)">' +
            'measures what a client actually gets, over /v1/chat/completions</span>';
    html += '  </div>';
    html += '  <div id="bench-panel">' + this._renderBenchPanel() + '</div>';
    html += '</section>';

    // --- Developer Logs ---
    html += '<section class="server-section">';
    html += '  <div class="server-section-header">';
    html += '    <h3 class="server-section-title">Developer Logs</h3>';
    html += '    <div class="server-log-controls">';
    html += '      <label class="server-log-toggle"><input type="checkbox" id="server-log-autoscroll"' + (this._serverState.autoScroll ? ' checked' : '') + '> auto-scroll</label>';
    html += '      <button class="btn-ghost server-btn-sm" id="server-log-clear">Clear</button>';
    html += '    </div>';
    html += '  </div>';
    html += '  <div class="server-log-panel" id="server-log-panel">';
    html += this._renderLogEntries(this._serverState.logs);
    html += '  </div>';
    html += '</section>';

    mount.innerHTML = html;

    this._bindServerEvents();
    this._renderServerRightPanel(this._serverState.selectedModelId);
  },

  _renderLoadedCard(m, idx) {
    var id = m.id || 'unknown';
    var nodeHost = m.node_hostname || m.node_id || 'local';
    // Stacked instances live on ports 8001+ — show the port so two models on
    // one node are distinguishable (F2). Primary instances omit it.
    var portStr = (m.port && m.port !== 8000) ? ' · :' + m.port : '';
    var type = m.type || 'llm';
    var isEmbed = type === 'embed';
    var ready = m.ready !== false;
    // Remote instances (loaded on a peer) can't be ejected from here — the eject
    // endpoint only targets this node's local InstanceManager (F2).
    var ejectable = m.ejectable !== false;
    var sizeStr = m.size_bytes > 0 ? this.formatBytes(m.size_bytes) : '—';
    var parallel = m.parallel || 1;
    var selected = (this._serverState.selectedModelId === id) ? ' selected' : '';
    var typeTagStyle = isEmbed
      ? ' style="color:var(--cyan);border-color:var(--cyan)"'
      : '';
    var dimsMeta = isEmbed && m.dimensions
      ? '  <span class="server-meta">· ' + m.dimensions + 'd</span>'
      : '';
    var primaryIconBtn = isEmbed
      ? '  <button class="server-icon-btn" data-action="show-info" data-model="' + this.esc(id) + '" title="Embedding info">ℹ</button>'
      : '  <button class="server-icon-btn" data-action="open-chat" data-model="' + this.esc(id) + '" title="Open in Chat">🔍</button>';
    var statusBadge = ready
      ? '<span class="server-badge ready">READY</span>'
      : '<span class="server-badge">STARTING</span>';
    var ejectBtn = ejectable
      ? '  <button class="server-eject-btn" data-action="eject" data-model="' + this.esc(id) + '">Eject</button>'
      : '';
    return '<div class="server-loaded-card' + selected + '" data-model-id="' + this.esc(id) + '" data-idx="' + idx + '">' +
      '<div class="server-loaded-left">' +
      '  ' + statusBadge +
      '  <span class="server-node-pill">' + this.esc(nodeHost + portStr) + '</span>' +
      '  <span class="server-type-tag"' + typeTagStyle + '>' + this.esc(type) + '</span>' +
      '  <span class="server-model-id mono" data-copy="' + this.esc(id) + '" title="Click to copy">' + this.esc(id) + '</span>' +
      '</div>' +
      '<div class="server-loaded-right">' +
      '  <span class="server-meta">' + this.esc(sizeStr) + '</span>' +
      '  <span class="server-meta">· ' + parallel + 'x</span>' +
      dimsMeta +
      '  <button class="server-icon-btn" data-action="preview" title="Preview">👁</button>' +
      primaryIconBtn +
      '  <button class="server-icon-btn" data-action="copy-curl" data-model="' + this.esc(id) + '" data-type="' + this.esc(type) + '" title="Copy curl">⎘</button>' +
      ejectBtn +
      '</div>' +
      '</div>';
  },

  // ---- Throughput benchmark -------------------------------------------

  //: Context sizes worth measuring. Throughput at 1K and at 64K are
  //: different numbers, and the second is what RAG and coding workloads look
  //: like — so the prompt is padded to the chosen size rather than assumed.
  BENCH_CONTEXTS: [
    { label: 'short prompt', tokens: 0 },
    { label: '4K context', tokens: 4096 },
    { label: '16K context', tokens: 16384 },
    { label: '32K context', tokens: 32768 },
    { label: '64K context', tokens: 65536 },
    { label: '128K context', tokens: 131072 },
  ],

  BENCH_LEVELS: ['1', '1,4,8', '1,4,8,16', '1,2,4,8,16,32'],

  _benchState: { models: [], status: null, model: '', levels: '1,4,8',
                 maxTokens: 256, promptTokens: 0, style: 'chat' },

  _renderBenchPanel() {
    var self = this;
    var state = this._benchState;
    var status = state.status || {};
    var running = !!status.running;

    var options = (state.models || []).map(function (id) {
      return '<option value="' + self.esc(id) + '"' +
        (id === state.model ? ' selected' : '') + '>' + self.esc(id) + '</option>';
    }).join('');
    if (!options) {
      options = '<option value="">no model is serving</option>';
    }

    var contexts = this.BENCH_CONTEXTS.map(function (c) {
      return '<option value="' + c.tokens + '"' +
        (c.tokens === state.promptTokens ? ' selected' : '') + '>' +
        c.label + '</option>';
    }).join('');

    var levels = this.BENCH_LEVELS.map(function (l) {
      return '<option value="' + l + '"' + (l === state.levels ? ' selected' : '') +
        '>' + l.split(',').length + ' Stufen (' + l + ')</option>';
    }).join('');

    var styles = ['chat', 'code', 'rag'].map(function (st) {
      return '<option value="' + st + '"' + (st === state.style ? ' selected' : '') +
        '>' + st + '</option>';
    }).join('');

    var field = 'padding:6px 8px;background:var(--bg-input,#111);color:inherit;' +
      'border:1px solid var(--border,#333);border-radius:4px';
    var label = 'font-size:11px;color:var(--text-muted);display:block;margin-bottom:3px';

    var html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;' +
      'margin-bottom:14px">' +
      '<div style="flex:1;min-width:220px"><label style="' + label + '">Model</label>' +
      '<select id="bench-model" class="mono" style="' + field + ';width:100%"' +
      (running ? ' disabled' : '') + '>' + options + '</select></div>' +
      '<div><label style="' + label + '">Context</label>' +
      '<select id="bench-context" style="' + field + '"' + (running ? ' disabled' : '') +
      '>' + contexts + '</select></div>' +
      '<div><label style="' + label + '">Concurrency</label>' +
      '<select id="bench-levels" style="' + field + '"' + (running ? ' disabled' : '') +
      '>' + levels + '</select></div>' +
      '<div><label style="' + label + '">Output tokens</label>' +
      '<input id="bench-max-tokens" type="number" min="1" max="4096" value="' +
      state.maxTokens + '" style="' + field + ';width:100px"' +
      (running ? ' disabled' : '') + '></div>' +
      '<div><label style="' + label + '">Prompt</label>' +
      '<select id="bench-style" style="' + field + '"' + (running ? ' disabled' : '') +
      '>' + styles + '</select></div>' +
      '<div>' + (running
        ? '<button class="btn-ghost server-btn-sm" id="bench-cancel">Stop</button>'
        : '<button class="btn-nvidia server-btn-sm" id="bench-run">Run</button>') +
      '</div></div>';

    // A benchmark is a load generator pointed at a production cluster. Say so
    // where the button is, not in a manual nobody opens.
    html += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:12px">' +
      'This occupies KV slots and bandwidth while it runs — real requests will ' +
      'be slower. The highest concurrency level is the heaviest part.</div>';

    if (running) {
      html += '<div class="server-empty">Running' +
        (status.current ? ' at ' + status.current + ' parallel…' : '…') +
        ' ' + ((status.results || []).length) + ' of ' +
        ((status.concurrency || []).length) + ' levels done.</div>';
    }
    if (status.error) {
      html += '<div class="instance-failed-note">' + this.esc(status.error) + '</div>';
    }

    var results = status.results || [];
    if (results.length) {
      html += '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;' +
        'font-size:12px">' +
        '<thead><tr style="text-align:left;color:var(--text-muted)">' +
        '<th style="padding:4px 8px">parallel</th>' +
        '<th style="padding:4px 8px">total tok/s</th>' +
        '<th style="padding:4px 8px">per stream</th>' +
        '<th style="padding:4px 8px">prompt</th>' +
        '<th style="padding:4px 8px">slowest</th>' +
        '<th style="padding:4px 8px">failed</th></tr></thead><tbody>';
      results.forEach(function (r) {
        html += '<tr style="border-top:1px solid var(--border,#333)">' +
          '<td style="padding:4px 8px" class="mono">' + r.concurrency + '</td>' +
          '<td style="padding:4px 8px" class="mono">' + (r.total_tokens_per_second || 0) + '</td>' +
          '<td style="padding:4px 8px" class="mono">' + (r.per_stream_tokens_per_second || 0) + '</td>' +
          '<td style="padding:4px 8px" class="mono">' + (r.prompt_tokens || 0) + '</td>' +
          '<td style="padding:4px 8px" class="mono">' + (r.slowest_seconds || 0) + 's</td>' +
          '<td style="padding:4px 8px" class="mono">' +
          (r.failed ? '<span style="color:var(--red,#e05)">' + r.failed + '</span>' : '0') +
          '</td></tr>';
        if (r.error) {
          html += '<tr><td colspan="6" style="padding:2px 8px;color:var(--text-muted)">' +
            self.esc(r.error) + '</td></tr>';
        }
      });
      html += '</tbody></table></div>';

      // The reading, not just the numbers. "Total keeps climbing while per
      // stream holds" and "both fall" mean different things and call for
      // different remedies, and that is the whole point of running this.
      var reading = self._benchReading(results);
      if (reading) {
        html += '<div style="margin-top:10px;font-size:12px;color:var(--text-muted)">' +
          self.esc(reading) + '</div>';
      }
    }
    return html;
  },

  _benchReading(results) {
    var usable = results.filter(function (r) { return r.ok > 0; });
    if (usable.length < 2) return '';
    var first = usable[0];
    var last = usable[usable.length - 1];
    if (!first.per_stream_tokens_per_second || !last.total_tokens_per_second) return '';
    var gain = last.total_tokens_per_second / (first.total_tokens_per_second || 1);
    var kept = last.per_stream_tokens_per_second / first.per_stream_tokens_per_second;
    var scale = gain.toFixed(1) + 'x total throughput at ' + last.concurrency +
      ' parallel, each stream at ' + Math.round(kept * 100) + '% of its solo rate. ';
    if (kept > 0.7) {
      return scale + 'Still scaling — try a higher concurrency level to find the knee.';
    }
    if (gain > 1.3) {
      return scale + 'Past the knee: more concurrency still buys throughput, ' +
        'but each user waits noticeably longer.';
    }
    return scale + 'Saturated — more concurrency costs latency without buying ' +
      'throughput. This is the ceiling for this model on these nodes.';
  },

  async _loadBenchOptions() {
    try {
      var data = await this.fetchJSON('/api/bench/options');
      this._benchState.models = (data && data.models) || [];
      if (!this._benchState.model && this._benchState.models.length) {
        this._benchState.model = this._benchState.models[0];
      }
    } catch (err) {
      this._benchState.models = [];
    }
  },

  _bindBenchPanel(root) {
    var self = this;
    var panel = root.querySelector('#bench-panel');
    if (!panel) return;

    var read = function () {
      var model = panel.querySelector('#bench-model');
      var context = panel.querySelector('#bench-context');
      var levels = panel.querySelector('#bench-levels');
      var maxTokens = panel.querySelector('#bench-max-tokens');
      var style = panel.querySelector('#bench-style');
      if (model) self._benchState.model = model.value;
      if (context) self._benchState.promptTokens = parseInt(context.value, 10) || 0;
      if (levels) self._benchState.levels = levels.value;
      if (maxTokens) self._benchState.maxTokens = parseInt(maxTokens.value, 10) || 256;
      if (style) self._benchState.style = style.value;
    };
    // Remember the selection across the poll-driven re-render, or a choice
    // made while a run is in flight is lost a second later.
    ['#bench-model', '#bench-context', '#bench-levels', '#bench-max-tokens',
     '#bench-style'].forEach(function (sel) {
      var el = panel.querySelector(sel);
      if (el) el.addEventListener('change', read);
    });

    var runBtn = panel.querySelector('#bench-run');
    if (runBtn) {
      runBtn.addEventListener('click', async function () {
        read();
        if (!self._benchState.model) {
          self.toast('No model is serving', 'error');
          return;
        }
        runBtn.disabled = true;
        try {
          var resp = await fetch('/api/bench/run', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: self._benchState.model,
              concurrency: self._benchState.levels.split(',').map(Number),
              max_tokens: self._benchState.maxTokens,
              prompt_tokens: self._benchState.promptTokens,
              style: self._benchState.style,
            }),
          });
          var payload = await resp.json().catch(function () { return {}; });
          if (!resp.ok) {
            self.toast(payload.error || 'Could not start the benchmark', 'error');
            runBtn.disabled = false;
            return;
          }
          self._pollBench();
        } catch (err) {
          self.toast('Could not start the benchmark: ' + err.message, 'error');
          runBtn.disabled = false;
        }
      });
    }

    var cancelBtn = panel.querySelector('#bench-cancel');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', async function () {
        cancelBtn.disabled = true;
        await fetch('/api/bench/cancel', { method: 'POST' }).catch(function () {});
        self._pollBench();
      });
    }
  },

  async _pollBench() {
    var self = this;
    if (this._benchPoll) return;
    var tick = async function () {
      try {
        self._benchState.status = await self.fetchJSON('/api/bench/status');
      } catch (err) {
        self._benchState.status = null;
      }
      var panel = document.querySelector('#bench-panel');
      if (panel) {
        self._redraw(panel, function () {
          panel.innerHTML = self._renderBenchPanel();
          self._bindBenchPanel(document);
        });
      }
      if (!(self._benchState.status && self._benchState.status.running)) {
        clearInterval(self._benchPoll);
        self._benchPoll = null;
      }
    };
    await tick();
    if (this._benchState.status && this._benchState.status.running) {
      this._benchPoll = setInterval(tick, 2000);
    }
  },

  _renderEndpointRows(baseUrl) {
    var self = this;
    var tab = this._serverState.endpointTab;
    var catalog = this._serverState.endpoints || {};
    var rows = catalog[tab] || [];
    if (!rows.length) return '<div class="server-empty">No endpoints</div>';
    return rows.map(function (ep) {
      var planned = ep.status === 'planned';
      var methodClass = 'server-method-' + ep.method.toLowerCase();
      var dim = planned ? ' planned' : '';
      return '<div class="server-endpoint-row' + dim + '">' +
        '<span class="server-method-badge ' + methodClass + '">' + self.esc(ep.method) + '</span>' +
        '<span class="server-endpoint-path mono">' + self.esc(ep.path) + '</span>' +
        '<span class="server-endpoint-desc">' + self.esc(ep.description || '') + '</span>' +
        (planned ? '<span class="server-endpoint-planned">planned</span>' : '') +
        '<button class="server-copy-btn" data-copy="curl -X ' + self.esc(ep.method) + ' ' + self.esc(baseUrl) + self.esc(ep.path) + '" title="Copy curl">⧉</button>' +
        '</div>';
    }).join('');
  },

  _renderLogEntries(entries) {
    var self = this;
    if (!entries || entries.length === 0) {
      return '<div class="server-log-empty">Waiting for API requests…</div>';
    }
    return entries.map(function (e) {
      var ts = new Date((e.timestamp || 0) * 1000);
      var tsStr = ts.toISOString().replace('T', ' ').replace('Z', '').slice(0, 19);
      var levelCls = 'log-' + (e.level || 'INFO').toLowerCase();
      var modelStr = e.model ? ' [' + self.esc(e.model) + ']' : '';
      var sizeStr = e.content_length ? ' ' + self.formatBytes(e.content_length) : '';
      return '<div class="server-log-entry ' + levelCls + '">' +
        '<span class="log-ts mono">' + tsStr + '</span> ' +
        '<span class="log-level">[' + self.esc(e.level || 'INFO') + ']</span>' +
        modelStr +
        ' <span class="log-method mono">' + self.esc(e.method) + '</span> ' +
        '<span class="log-path mono">' + self.esc(e.path) + '</span> ' +
        '<span class="log-status mono">' + (e.status || 0) + '</span> ' +
        '<span class="log-dur mono">' + (e.duration_ms || 0) + 'ms</span>' +
        '<span class="log-size mono">' + sizeStr + '</span>' +
        '</div>';
    }).join('');
  },

  _bindServerEvents() {
    var self = this;
    var root = document.getElementById('server-content');
    if (!root) return;

    // Copy buttons
    root.querySelectorAll('[data-copy]').forEach(function (el) {
      el.addEventListener('click', function (e) {
        e.stopPropagation();
        var text = el.dataset.copy || '';
        if (!text) return;
        navigator.clipboard.writeText(text).then(function () {
          self.toast('Copied to clipboard', 'success');
        }).catch(function () { self.toast('Copy failed', 'error'); });
      });
    });

    this.checkOpencodeDrift();

    var openCodeBtn = root.querySelector('#opencode-config');
    if (openCodeBtn) {
      openCodeBtn.addEventListener('click', async function () {
        var out = document.querySelector('#opencode-config-out');
        openCodeBtn.disabled = true;
        var label = openCodeBtn.textContent;
        openCodeBtn.textContent = 'Reading the cluster…';
        try {
          // The head's own address, so the pasted config points where the
          // operator is already talking to rather than at localhost.
          var base = location.protocol + '//' + location.host;
          var data = await self.fetchJSON(
            '/api/clients/opencode?base_url=' + encodeURIComponent(base));
          // The view may have been redrawn while this was in flight; write
          // into the element that is on the page now, not the one clicked on.
          out = document.querySelector('#opencode-config-out') || out;
          var text = JSON.stringify(data.config, null, 2);
          var notes = (data.notes || []).map(function (n) {
            return '<div style="color:var(--text-muted);font-size:11px;' +
              'margin-bottom:4px">' + self.esc(n) + '</div>';
          }).join('');
          out.innerHTML = notes +
            '<div style="display:flex;gap:8px;margin:8px 0">' +
            '<button class="btn-nvidia server-btn-sm" data-copy="' +
              self.esc(text) + '">Copy</button>' +
            '<span style="font-size:11px;color:var(--text-muted);align-self:center">' +
              'save as ~/.config/opencode/opencode.json</span></div>' +
            '<pre style="max-height:320px;overflow:auto;background:var(--bg-input,#111);' +
            'padding:10px;border-radius:4px;font-size:11px">' +
            self.esc(text) + '</pre>';
          out.querySelectorAll('[data-copy]').forEach(function (b) {
            b.addEventListener('click', function () {
              navigator.clipboard.writeText(b.dataset.copy).then(function () {
                self.toast('Config copied', 'success');
                self.rememberOpencodeConfig(data);
              }).catch(function () { self.toast('Copy failed', 'error'); });
            });
          });
        } catch (err) {
          out.innerHTML = '<div class="server-empty">Could not build it: ' +
            self.esc(err.message) + '</div>';
        }
        openCodeBtn.disabled = false;
        openCodeBtn.textContent = label;
      });
    }

    // Throughput benchmark: options once, then bind. A run already in
    // flight — started here, or from another browser — resumes its poll, so
    // reloading the page does not lose a measurement in progress.
    this._bindBenchPanel(root);
    if (!this._benchState.models.length) {
      this._loadBenchOptions().then(function () {
        var panel = document.querySelector('#bench-panel');
        if (panel) {
          panel.innerHTML = self._renderBenchPanel();
          self._bindBenchPanel(document);
        }
      });
    }
    this._pollBench();

    // Endpoint tab pills
    root.querySelectorAll('.server-tab-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self._serverState.endpointTab = btn.dataset.tab;
        var s = self._serverState.lastStatus || {};
        var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || '';
        var list = document.getElementById('server-endpoints-list');
        if (list) list.innerHTML = self._renderEndpointRows(primary);
        root.querySelectorAll('.server-tab-pill').forEach(function (b) {
          b.classList.toggle('active', b.dataset.tab === self._serverState.endpointTab);
        });
        // Rebind copy buttons in newly rendered rows
        self._bindServerEvents();
      });
    });

    // Loaded card click → select + update right panel
    root.querySelectorAll('.server-loaded-card').forEach(function (card) {
      card.addEventListener('click', function () {
        var id = card.dataset.modelId;
        self._serverState.selectedModelId = id;
        root.querySelectorAll('.server-loaded-card').forEach(function (c) { c.classList.remove('selected'); });
        card.classList.add('selected');
        self._renderServerRightPanel(id);
      });
    });

    // Loaded card actions
    root.querySelectorAll('.server-loaded-card [data-action]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var action = btn.dataset.action;
        var model = btn.dataset.model;
        if (action === 'eject') self._serverEjectModel(model);
        else if (action === 'open-chat') {
          self.state.chatModel = model;
          self.navigate('chat');
        }
        else if (action === 'show-info') self._serverShowEmbedInfo(model);
        else if (action === 'copy-curl') self._serverShowCurl(model, btn.dataset.type || 'llm');
        else if (action === 'preview') self.toast('Preview coming soon', 'info');
      });
    });

    // Top-bar buttons
    var toggleBtn = document.getElementById('server-toggle');
    if (toggleBtn) toggleBtn.addEventListener('click', function () { self.toast('Server toggle not yet implemented', 'info'); });
    var settingsBtn = document.getElementById('server-settings-btn');
    if (settingsBtn) settingsBtn.addEventListener('click', function () { self.navigate('config'); });
    var mcpBtn = document.getElementById('server-mcp-btn');
    if (mcpBtn) mcpBtn.addEventListener('click', function () { self.toast('mcp.json export coming soon', 'info'); });
    var loadBtn = document.getElementById('server-load-model');
    if (loadBtn) loadBtn.addEventListener('click', function () { self._openLoadModelModal(); });

    // Logs
    var clearBtn = document.getElementById('server-log-clear');
    if (clearBtn) clearBtn.addEventListener('click', function () { self._serverClearLogs(); });
    var autoScroll = document.getElementById('server-log-autoscroll');
    if (autoScroll) autoScroll.addEventListener('change', function () { self._serverState.autoScroll = autoScroll.checked; });
  },

  async _serverEjectModel(modelId) {
    if (!modelId) return;
    if (!confirm('Eject model ' + modelId + '?')) return;
    try {
      var resp = await fetch('/api/server/models/' + encodeURIComponent(modelId) + '/eject', { method: 'POST' });
      var body = await resp.json().catch(function () { return {}; });
      if (resp.ok) this.toast(body.message || 'Model ejected', 'success');
      else this.toast(body.message || 'Eject not available', 'info');
      this.renderServer();
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  _serverShowCurl(modelId, modelType) {
    var s = this._serverState.lastStatus || {};
    var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || 'http://localhost:3000';
    var curl;
    if (modelType === 'embed') {
      curl = 'curl ' + primary + '/v1/embeddings \\\n' +
        '  -H "Content-Type: application/json" \\\n' +
        '  -d \'{"model":"' + modelId + '","input":"The quick brown fox"}\'';
    } else {
      curl = 'curl ' + primary + '/v1/chat/completions \\\n' +
        '  -H "Content-Type: application/json" \\\n' +
        '  -d \'{"model":"' + modelId + '","messages":[{"role":"user","content":"Hello!"}]}\'';
    }
    navigator.clipboard.writeText(curl).then(function () {
      AINode.toast('curl example copied', 'success');
    }).catch(function () { AINode.toast('Copy failed', 'error'); });
  },

  _serverShowEmbedInfo(modelId) {
    var s = this._serverState.lastStatus || {};
    var models = s.loaded_models || [];
    var m = models.find(function (x) { return x.id === modelId; }) || { id: modelId };
    var primary = (s.reachable_at && s.reachable_at.length > 1) ? s.reachable_at[1] : (s.reachable_at && s.reachable_at[0]) || 'http://localhost:3000';
    var self = this;

    var existing = document.getElementById('embed-info-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'embed-info-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:560px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon" style="color:var(--cyan);border-color:var(--cyan)">ℹ</div>' +
            '<div class="md-title">Embedding Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description">' +
          '<div style="margin-bottom:8px" class="mono">' + self.esc(m.id) + '</div>' +
          '<div style="color:var(--text-muted);font-size:13px;margin-bottom:12px">' +
          'Embedding models turn text into vectors. They are an API surface for external apps ' +
          '(RAG, semantic search, clustering) — not used directly by the chat UI.' +
          '</div>' +
          '<div style="display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-size:13px">' +
            '<div style="color:var(--text-muted)">Dimensions</div><div>' + (m.dimensions || '—') + '</div>' +
            '<div style="color:var(--text-muted)">Max seq length</div><div>' + (m.max_seq_length || '—') + '</div>' +
            '<div style="color:var(--text-muted)">Endpoint</div><div class="mono">' + self.esc(primary) + '/v1/embeddings</div>' +
          '</div>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="embed-info-close" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Close</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#embed-info-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
  },

  _openLoadModelModal() {
    var self = this;
    var existing = document.getElementById('load-model-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'load-model-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:720px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon">+</div>' +
            '<div class="md-title">Load Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description" style="padding-bottom:0">' +
          '<div class="server-tab-pills" id="load-model-tabs">' +
            '<button class="server-tab-pill active" data-tab="llms">LLMs</button>' +
            '<button class="server-tab-pill" data-tab="embeddings">Embeddings</button>' +
          '</div>' +
          '<div id="load-model-body" style="margin-top:12px;max-height:420px;overflow:auto"></div>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="load-model-close" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Close</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#load-model-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });

    var currentTab = 'llms';
    var render = function () { self._renderLoadModelTab(modal, currentTab); };
    modal.querySelectorAll('#load-model-tabs .server-tab-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        currentTab = btn.dataset.tab;
        modal.querySelectorAll('#load-model-tabs .server-tab-pill').forEach(function (b) {
          b.classList.toggle('active', b.dataset.tab === currentTab);
        });
        render();
      });
    });
    render();
  },

  async _renderLoadModelTab(modal, tab) {
    var self = this;
    var body = modal.querySelector('#load-model-body');
    if (!body) return;
    body.innerHTML = '<div class="server-empty">Loading…</div>';

    if (tab === 'llms') {
      try {
        var data = await this.fetchJSON('/api/models');
        var models = (data && data.models) || [];
        var downloaded = models.filter(function (m) { return m.downloaded; });
        if (!downloaded.length) {
          body.innerHTML = '<div class="server-empty">No LLMs downloaded yet. Use the Downloads view to get one.</div>';
          return;
        }
        var html = '<div class="server-loaded-list">';
        downloaded.forEach(function (m) {
          var id = m.hf_repo || m.id || m.name || 'unknown';
          html += '<div class="server-loaded-card">' +
            '<div class="server-loaded-left">' +
            '  <span class="server-type-tag">llm</span>' +
            '  <span class="server-model-id mono">' + self.esc(id) + '</span>' +
            '</div>' +
            '<div class="server-loaded-right">' +
            '  <button class="btn-nvidia server-btn-sm" data-action="llm-load" data-model="' + self.esc(id) + '">Load</button>' +
            '</div>' +
            '</div>';
        });
        html += '</div>';
        body.innerHTML = html;
        body.querySelectorAll('[data-action="llm-load"]').forEach(function (btn) {
          btn.addEventListener('click', async function () {
            var mid = btn.dataset.model;
            btn.disabled = true;
            btn.textContent = 'Loading…';
            try {
              var resp = await fetch('/api/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: mid }),
              });
              var payload = await resp.json().catch(function () { return {}; });
              if (resp.ok && !payload.error) {
                self.toast('Loaded ' + mid, 'success');
                self.renderServer();
              } else {
                self.toast((payload.error && payload.error.message) || payload.error || 'Load failed', 'error');
                btn.disabled = false;
                btn.textContent = 'Load';
              }
            } catch (err) {
              self.toast('Load failed: ' + err.message, 'error');
              btn.disabled = false;
              btn.textContent = 'Load';
            }
          });
        });
      } catch (err) {
        body.innerHTML = '<div class="server-empty">Failed to load models: ' + self.esc(err.message) + '</div>';
      }
      return;
    }

    // Embeddings tab
    try {
      var edata = await this.fetchJSON('/api/embeddings/models');
      var emodels = (edata && edata.models) || [];
      // No early return on an empty catalog: the free-text field below is the
      // way to load anything at all, and hiding it behind a non-empty list
      // would leave an operator with an empty tab and no next step.
      // Any sentence-transformers repo, not only the four in the catalog.
      // The load endpoint has always accepted an arbitrary repo id — it hands
      // the string straight to SentenceTransformer — but the only way to
      // reach that was curl, because this tab renders a fixed list. Someone
      // looking for nomic-embed-text-v1 found nothing and reasonably
      // concluded AINode could not serve it. The model search next door is no
      // help either: it filters on pipeline_tag=text-generation, so an
      // embedding model cannot appear there by construction.
      var ehtml =
        '<div class="server-loaded-card" style="margin-bottom:12px">' +
        '  <div class="server-loaded-left" style="flex-direction:column;align-items:flex-start;gap:6px;flex:1">' +
        '    <div style="font-size:12px;color:var(--text-muted)">' +
        '      Any Hugging Face embedding model (sentence-transformers)' +
        '    </div>' +
        '    <input id="embed-any-repo" class="mono" placeholder="nomic-ai/nomic-embed-text-v1.5" ' +
        '           style="width:100%;max-width:420px;padding:6px 8px;background:var(--bg-input,#111);' +
        '                  color:inherit;border:1px solid var(--border,#333);border-radius:4px">' +
        '  </div>' +
        '  <div class="server-loaded-right" style="gap:8px">' +
        // Which node runs it. An embedding model loads on whichever node's API
        // is asked, so before this the only way to put one on node 3 was to
        // open node 3's own UI — and a profile saved on the head then brought
        // it back on the head.
        '    ' + self._embedNodeSelect() +
        '    <button class="btn-nvidia server-btn-sm" id="embed-any-load">Load</button>' +
        '  </div>' +
        '</div>' +
        '<div class="server-loaded-list">';
      emodels.forEach(function (m) {
        var loaded = !!m.loaded;
        ehtml += '<div class="server-loaded-card">' +
          '<div class="server-loaded-left" style="flex-direction:column;align-items:flex-start;gap:4px">' +
          '  <div>' +
          '    <span class="server-type-tag" style="color:var(--cyan);border-color:var(--cyan)">embed</span> ' +
          '    <span class="server-model-id mono">' + self.esc(m.id) + '</span>' +
          '  </div>' +
          '  <div style="font-size:12px;color:var(--text-muted)">' +
          (m.dimensions || '?') + 'd · ' + (m.max_seq_length || '?') + ' ctx · ' + (m.size_mb || '?') + ' MB' +
          '  </div>' +
          '  <div style="font-size:12px;color:var(--text-muted);max-width:480px">' + self.esc(m.description || '') + '</div>' +
          '</div>' +
          '<div class="server-loaded-right">' +
          (loaded
            ? '  <span class="server-badge ready">LOADED' +
              (m.node_name ? ' · ' + self.esc(m.node_name) : '') + '</span>'
            : '  <button class="btn-nvidia server-btn-sm" data-action="embed-load" data-model="' + self.esc(m.id) + '">Load</button>'
          ) +
          '</div>' +
          '</div>';
      });
      if (!emodels.length) {
        ehtml += '<div class="server-empty">No curated embedding models — ' +
                 'name any repo above.</div>';
      }
      ehtml += '</div>';
      body.innerHTML = ehtml;

      var anyInput = body.querySelector('#embed-any-repo');
      var anyButton = body.querySelector('#embed-any-load');
      var loadEmbedding = async function (id, btn, restoreLabel) {
        btn.disabled = true;
        btn.textContent = 'Loading…';
        // Always through the cluster route, even for this node: one path, and
        // an empty node_id means "here". Two paths is how the placement got
        // lost in the first place.
        var sel = body.querySelector('#embed-node');
        var nodeId = sel ? sel.value : '';
        try {
          var resp = await fetch('/api/cluster/embeddings/load', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: id, node_id: nodeId }),
          });
          var payload = await resp.json().catch(function () { return {}; });
          if (resp.ok) {
            self.toast('Loaded ' + id, 'success');
            self._renderLoadModelTab(modal, 'embeddings');
            self.renderServer();
            return;
          }
          self.toast((payload.error && payload.error.message) || 'Load failed', 'error');
        } catch (err) {
          self.toast('Load failed: ' + err.message, 'error');
        }
        btn.disabled = false;
        btn.textContent = restoreLabel;
      };
      if (anyButton && anyInput) {
        var loadTyped = function () {
          var id = (anyInput.value || '').trim();
          if (!id) return;
          if (id.indexOf('/') === -1) {
            // A bare name reaches the Hub as a repo id and 404s minutes later.
            self.toast('Use the full repo id, owner/name', 'error');
            return;
          }
          loadEmbedding(id, anyButton, 'Load');
        };
        anyButton.addEventListener('click', loadTyped);
        anyInput.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') loadTyped();
        });
      }

      body.querySelectorAll('[data-action="embed-load"]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          loadEmbedding(btn.dataset.model, btn, 'Load');
        });
      });
    } catch (err) {
      body.innerHTML = '<div class="server-empty">Failed to load embeddings: ' + self.esc(err.message) + '</div>';
    }
  },

  async _serverClearLogs() {
    try {
      await fetch('/api/server/logs', { method: 'DELETE' });
      this._serverState.logs = [];
      this._serverState.logsSince = 0;
      var panel = document.getElementById('server-log-panel');
      if (panel) panel.innerHTML = this._renderLogEntries([]);
      this.toast('Logs cleared', 'success');
    } catch (err) {
      this.toast('Failed to clear logs', 'error');
    }
  },

  startServerLogPolling() {
    var self = this;
    if (this._serverState.logsPoll) return;
    this._serverState.logsPoll = setInterval(function () { self._pollServerLogs(); }, 2000);
    this._pollServerLogs();
  },

  stopServerLogPolling() {
    if (this._serverState.logsPoll) {
      clearInterval(this._serverState.logsPoll);
      this._serverState.logsPoll = null;
    }
  },

  async _pollServerLogs() {
    var since = this._serverState.logsSince || 0;
    var data = await this.fetchJSON('/api/server/logs?since=' + since);
    if (!data || !data.entries) return;
    if (data.entries.length > 0) {
      this._serverState.logs = this._serverState.logs.concat(data.entries).slice(-500);
      this._serverState.logsSince = data.entries[data.entries.length - 1].timestamp || data.now;
      var panel = document.getElementById('server-log-panel');
      if (panel) {
        panel.innerHTML = this._renderLogEntries(this._serverState.logs);
        if (this._serverState.autoScroll) panel.scrollTop = panel.scrollHeight;
      }
    } else if (!this._serverState.logsSince) {
      this._serverState.logsSince = data.now || Date.now() / 1000;
    }
  },

  _renderServerRightPanel(modelId) {
    var mount = document.getElementById('right-panel-server');
    if (!mount) return;
    var s = this._serverState.lastStatus || {};
    var models = s.loaded_models || [];
    var model = null;
    if (modelId) model = models.find(function (m) { return m.id === modelId; });
    if (!model && models.length > 0) {
      model = models[0];
      this._serverState.selectedModelId = model.id;
    }
    if (!model) {
      mount.innerHTML = '<div class="panel-section"><h3 class="panel-title">MODEL INFO</h3>' +
        '<div class="server-empty" style="margin:12px 16px">No model selected</div></div>';
      return;
    }
    var self = this;
    var primary = (s.reachable_at && (s.reachable_at[1] || s.reachable_at[0])) || '—';
    // Pull derivable fields from the catalog (size/arch/quant) — the served
    // model object lacks them. Real arch (e.g. LlamaForCausalLM) needs the
    // per-model config.json (owned follow-up); family is a truthful stand-in.
    var _cat = (this._serverState.modelsCatalog || []).find(function (c) { return (c.hf_repo || c.id) === model.id; }) || {};
    var arch = model.architecture || _cat.architecture || _cat.family || AINode._modelFamily(model.id) || '—';
    var fileName = (model.id || '').split('/').pop();
    var _catSize = _cat.local_size_gb || _cat.size_gb;
    var sizeStr = model.size_bytes > 0 ? this.formatBytes(model.size_bytes)
      : (_catSize ? Math.round(_catSize) + ' GB' : '—');

    var html = '';
    html += '<div class="panel-section">';
    html += '  <h3 class="panel-title">MODEL INFO</h3>';
    html += '  <div class="server-info-host">';
    html += '    <span class="label">Hosted on</span>';
    html += '    <span class="mono">' + this.esc(model.node_hostname || '') + '</span>';
    html += '    <button class="server-copy-btn" data-copy="' + this.esc(model.node_hostname || '') + '">⧉</button>';
    html += '  </div>';
    html += '  <div class="server-tab-pills server-info-tabs" id="server-info-tabs">';
    html += '    <button class="server-tab-pill active" data-info-tab="info">Info</button>';
    html += '    <button class="server-tab-pill" data-info-tab="load">Load</button>';
    html += '    <button class="server-tab-pill" data-info-tab="inference">Inference</button>';
    html += '  </div>';
    html += '  <div id="server-info-body">';
    html += this._renderServerInfoTab('info', model, arch, fileName, sizeStr);
    html += '  </div>';
    html += '</div>';

    html += '<div class="panel-section">';
    html += '  <h3 class="panel-title">API USAGE</h3>';
    html += '  <div class="server-info-section">';
    html += '    <div class="server-info-row"><span class="label">Model ID</span><span class="mono val">' + this.esc(model.id) + '</span><button class="server-copy-btn" data-copy="' + this.esc(model.id) + '">⧉</button></div>';
    html += '    <div class="server-info-row"><span class="label">Reachable at</span><span class="mono val">' + this.esc(primary) + '</span><button class="server-copy-btn" data-copy="' + this.esc(primary) + '">⧉</button></div>';
    html += '  </div>';
    html += '</div>';

    mount.innerHTML = html;

    mount.querySelectorAll('[data-copy]').forEach(function (el) {
      el.addEventListener('click', function () {
        var text = el.dataset.copy || '';
        if (!text) return;
        navigator.clipboard.writeText(text).then(function () { self.toast('Copied', 'success'); });
      });
    });
    mount.querySelectorAll('[data-info-tab]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        mount.querySelectorAll('[data-info-tab]').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var body = document.getElementById('server-info-body');
        if (body) body.innerHTML = self._renderServerInfoTab(btn.dataset.infoTab, model, arch, fileName, sizeStr);
      });
    });
  },

  _modelQuant(id) {
    var m = (id || '').toUpperCase().match(/NVFP4|MXFP4|AWQ|GPTQ|FP8|INT4|INT8/);
    return m ? m[0] : '';
  },

  _modelFamily(id) {
    var n = (id || '').split('/').pop().toLowerCase();
    if (n.indexOf('llama') >= 0) return 'Llama';
    if (n.indexOf('qwen') >= 0) return 'Qwen';
    if (n.indexOf('glm') >= 0) return 'GLM';
    if (n.indexOf('mixtral') >= 0 || n.indexOf('mistral') >= 0) return 'Mistral';
    if (n.indexOf('deepseek') >= 0) return 'DeepSeek';
    if (n.indexOf('phi') >= 0) return 'Phi';
    if (n.indexOf('gemma') >= 0) return 'Gemma';
    return '';
  },

  // One row, or an em-dash that does not pretend. Never a literal.
  _loadRow(label, value) {
    var shown = (value === '' || value === null || value === undefined)
      ? '—' : String(value);
    return '<div class="server-info-row"><span class="label">' +
      this.esc(label) + '</span><span class="mono">' + this.esc(shown) +
      '</span></div>';
  },

  _renderServerInfoTab(tab, model, arch, fileName, sizeStr) {
    if (tab === 'load') {
      // These were three literals in this template — 4096 and -1 typed into
      // the HTML, under a real model, claiming to describe it. A number
      // nobody measured is worse than a blank: a blank does not get pasted
      // into a client config. The layer-offload row is gone rather than
      // zeroed; it is a llama.cpp knob that never applied to a vLLM.
      var known = (model.max_model_len || model.kv_cache_dtype ||
                   model.gpu_memory_utilization);
      var split = (model.pipeline_parallel_size > 1)
        ? 'pipeline x' + model.pipeline_parallel_size
        : (model.tensor_parallel_size > 1
            ? 'tensor x' + model.tensor_parallel_size : 'single node');
      return '<div class="server-info-section">' +
        this._loadRow('Context length', model.max_model_len
          ? this.formatNumber(model.max_model_len) + ' tokens' : '') +
        this._loadRow('KV cache dtype', model.kv_cache_dtype || '') +
        this._loadRow('Memory share', model.gpu_memory_utilization
          ? Math.round(model.gpu_memory_utilization * 100) + '% of total' : '') +
        this._loadRow('Concurrent sequences', model.max_num_seqs || '') +
        this._loadRow('Split', known ? split : '') +
        this._loadRow('Trust remote code',
          known ? (model.trust_remote_code ? 'yes' : 'no') : '') +
        '<div class="server-hint">' + (known
          ? 'What this instance was launched with. Read-only — change it by ' +
            'reloading the model.'
          : 'This node did not report its launch parameters. The engine log ' +
            'banner has them: <span class="mono">grep &quot;serve command&quot; ' +
            '~/.ainode/logs/*.log</span>') +
        '</div>' +
        '</div>';
    }
    if (tab === 'inference') {
      return '<div class="server-info-section">' +
        '<div class="server-hint">Sampling is per request, not per instance — ' +
        'this engine holds no temperature, top-p or top-k of its own. Send ' +
        'them with the request; the OpenAI-compatible defaults apply when you ' +
        'do not.</div>' +
        '</div>';
    }
    // Info tab
    var caps = (model.capabilities || []).map(function (c) { return '<span class="server-cap-badge">' + AINode.esc(c) + '</span>'; }).join('');
    return '<div class="server-info-section">' +
      '<div class="server-info-row"><span class="label">Model</span><span class="mono val">' + AINode.esc(model.id) + '</span></div>' +
      '<div class="server-info-row"><span class="label">File</span><span class="mono val">' + AINode.esc(fileName) + '</span></div>' +
      '<div class="server-info-row"><span class="label">Format</span><span class="val">' + AINode.esc(model.format || 'SafeTensors') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Quantization</span><span class="val">' + AINode.esc(model.quantization || AINode._modelQuant(model.id) || 'none') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Arch</span><span class="val">' + AINode.esc(arch) + '</span></div>' +
      '<div class="server-info-row"><span class="label">Capabilities</span><span class="val">' + (caps || '—') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Domain</span><span class="val">' + AINode.esc(model.type || 'llm') + '</span></div>' +
      '<div class="server-info-row"><span class="label">Size on disk</span><span class="val">' + AINode.esc(sizeStr) + '</span></div>' +
      '</div>';
  },
});
