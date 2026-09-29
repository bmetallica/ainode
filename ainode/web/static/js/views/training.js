/* AINode Command Center — the Training view, its wizard and dataset modals, the guide and the loss chart.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  TRAINING VIEW  (overview / autodata / datasets / runs / templates)
  // ========================================================================

  // ---- Sidebar ------------------------------------------------------------
  async renderTrainingSidebar() {
    var mount = document.getElementById('left-panel-training');
    if (!mount) return;
    var self = this;

    var jobs = this.state.trainingJobs || [];
    // Pull fresh jobs if we don't have them yet
    if (!jobs.length) {
      var data = await this.fetchJSON('/api/training/jobs');
      this.state.trainingJobs = (data && data.jobs) || [];
      jobs = this.state.trainingJobs;
    }

    var tab = this.state.trainingTab || 'overview';
    var items = [
      { id: 'overview',   label: 'Overview',   icon: '&#9711;' },
      { id: 'autodata',   label: 'AutoData',   icon: '&#9883;' },
      { id: 'datasets',   label: 'Datasets',   icon: '&#8864;' },
      { id: 'runs',       label: 'Runs',       icon: '&#9873;' },
      { id: 'templates',  label: 'Templates',  icon: '&#9641;' },
    ];
    var guides = [
      { id: 'beginners',    label: "Beginner's Guide" },
      { id: 'distributed',  label: 'Distributed Training' },
      { id: 'pipeline',     label: 'Training Pipeline' },
      { id: 'standalone',   label: 'Standalone Training' },
    ];

    var recent = jobs.slice().sort(function (a, b) {
      return (b.start_time || 0) - (a.start_time || 0);
    }).slice(0, 5);

    var subnavHtml = items.map(function (it) {
      var active = it.id === tab ? ' active' : '';
      return '<button class="training-subnav-item' + active + '" data-tab="' + it.id + '">' +
        '<span class="sub-icon">' + it.icon + '</span>' + self.esc(it.label) + '</button>';
    }).join('');

    var recentHtml = recent.length
      ? recent.map(function (j) {
          var s = self._statusInfo(j.status);
          var model = (j.config && j.config.base_model ? j.config.base_model.split('/').pop() : '—');
          var name = (j.config && j.config.run_name) || model;
          return '<div class="training-recent-item" data-job-id="' + self.esc(j.job_id) + '">' +
            '<div class="training-recent-top">' +
              '<span class="training-recent-name">' + self.esc(name) + '</span>' +
              '<span class="job-status-badge ' + s.cls + '">' + s.label + '</span>' +
            '</div>' +
            '<div class="training-recent-meta">' + self.esc(j.job_id) + '</div>' +
            '</div>';
        }).join('')
      : '<div class="conv-empty" style="padding:10px 14px;font-size:12px;color:var(--text-muted)">No runs yet</div>';

    var guidesHtml = guides.map(function (g) {
      return '<a class="training-guide-link" data-guide="' + g.id + '">&#9733; ' + self.esc(g.label) + '</a>';
    }).join('');

    mount.innerHTML =
      '<button class="btn-nvidia" id="training-new-run-btn">+ NEW RUN</button>' +
      '<div class="training-sidebar-title">Training</div>' +
      '<div class="training-sidebar-section training-subnav">' + subnavHtml + '</div>' +
      '<div class="training-sidebar-section">' +
        '<div class="training-sidebar-title" style="padding:0 8px 8px">Recent Runs</div>' +
        recentHtml +
      '</div>' +
      '<div class="training-sidebar-section">' +
        '<div class="training-sidebar-title" style="padding:0 8px 8px">Guides</div>' +
        guidesHtml +
      '</div>';

    // Bindings
    var newBtn = document.getElementById('training-new-run-btn');
    if (newBtn) newBtn.addEventListener('click', function () { self.showNewRunWizard(); });

    mount.querySelectorAll('.training-subnav-item').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.trainingTab = btn.dataset.tab;
        self.state.trainingView = 'list';
        self.state.trainingDetailId = null;
        self._autodataViewInitialized = false;  // full-render the AutoData panel on (re)entry
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
    mount.querySelectorAll('.training-recent-item').forEach(function (el) {
      el.addEventListener('click', function () {
        self.state.trainingTab = 'runs';
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = el.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
    mount.querySelectorAll('.training-guide-link').forEach(function (el) {
      el.addEventListener('click', function () {
        self.showGuideModal(el.dataset.guide);
      });
    });
  },

  _statusInfo(status) {
    var m = {
      pending:   { cls: 'status-pending',   label: 'Pending' },
      running:   { cls: 'status-running',   label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed:    { cls: 'status-failed',    label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    return m[status] || { cls: '', label: status || '—' };
  },

  async renderTraining() {
    var container = document.getElementById('training-content');
    if (!container) return;

    // If we're drilling into a specific job detail, handle that first
    if (this.state.trainingView === 'detail' && this.state.trainingDetailId) {
      var data = await this.fetchJSON('/api/training/jobs');
      this.state.trainingJobs = (data && data.jobs) || [];
      await this.renderTrainingDetail(container);
      return;
    }

    var tab = this.state.trainingTab || 'overview';
    // Fetch jobs + datasets + stats in parallel
    var results = await Promise.all([
      this.fetchJSON('/api/training/jobs'),
      this.fetchJSON('/api/training/stats'),
      this.fetchJSON('/api/datasets'),
      this.fetchJSON('/api/training/templates'),
    ]);
    this.state.trainingJobs = (results[0] && results[0].jobs) || [];
    this.state.trainingStats = results[1] || null;
    this.state.datasets = (results[2] && results[2].datasets) || [];
    this.state.trainingTemplates = (results[3] && results[3].templates) || [];

    // Refresh sidebar so Recent Runs shows latest
    this.renderTrainingSidebar();

    switch (tab) {
      case 'overview':   this.renderTrainingOverview(container); break;
      case 'autodata':
        // Mirror the downloads-view guard: full render on entry, then the 5s poll
        // leaves the panel alone (the run has its own poller) so the form never flickers.
        if (this._autodataViewInitialized) { /* in place — _pollAutoData owns #autodata-status */ }
        else { this._autodataViewInitialized = true; this.renderTrainingAutoData(container); }
        break;
      case 'datasets':   this.renderTrainingDatasets(container); break;
      case 'runs':       this.renderTrainingRuns(container); break;
      case 'templates':  this.renderTrainingTemplates(container); break;
      default:           this.renderTrainingOverview(container);
    }
  },

  // ---- Overview ----------------------------------------------------------
  renderTrainingOverview(container) {
    var self = this;
    var stats = this.state.trainingStats || {};
    var jobs = this.state.trainingJobs || [];
    var recent = jobs.slice().sort(function (a, b) {
      return (b.start_time || 0) - (a.start_time || 0);
    }).slice(0, 4);

    var tilesHtml =
      '<div class="training-stat-tile accent"><div class="stat-label">Active Runs</div><div class="stat-value">' + (stats.running || 0) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">Completed Today</div><div class="stat-value">' + (stats.completed_today || 0) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">Total Runs</div><div class="stat-value">' + (stats.total || jobs.length) + '</div></div>' +
      '<div class="training-stat-tile"><div class="stat-label">GPU Hours</div><div class="stat-value">' + (stats.total_gpu_hours != null ? stats.total_gpu_hours.toFixed(1) : '0.0') + '</div></div>';

    var quickstart = [
      { id: 'lora', icon: 'L', title: 'Fine-tune with LoRA', desc: 'Small adapter weights. Recommended for most users — trains fast, stays memory-efficient.' },
      { id: 'distributed', icon: 'D', title: 'Distributed Training', desc: 'Scale across multiple DGX Spark nodes with DDP + NCCL. Great for large datasets.' },
      { id: 'full', icon: 'F', title: 'Full Fine-tune', desc: 'Updates all weights. Highest quality, needs the most memory — use for single large-memory nodes.' },
    ].map(function (q) {
      return '<div class="quickstart-tile" data-qs="' + q.id + '">' +
        '<div class="quickstart-icon">' + q.icon + '</div>' +
        '<h4>' + self.esc(q.title) + '</h4>' +
        '<p>' + self.esc(q.desc) + '</p>' +
        '<button class="btn-nvidia" data-qs-btn="' + q.id + '">Start &rsaquo;</button>' +
        '</div>';
    }).join('');

    var recentHtml = recent.length
      ? recent.map(function (j) { return self.renderJobCard(j); }).join('')
      : '<div class="training-empty"><h3>No runs yet</h3><p>Kick off your first fine-tune from the quick-start tiles above.</p></div>';

    container.innerHTML =
      '<div class="training-overview">' +
        '<div class="training-stat-tiles">' + tilesHtml + '</div>' +
        '<div>' +
          '<div class="training-section-head"><h3>Quick Start</h3><span class="muted">Jump straight in</span></div>' +
          '<div class="quickstart-grid">' + quickstart + '</div>' +
        '</div>' +
        '<div>' +
          '<div class="training-section-head"><h3>Quantize a Model</h3><span class="muted">Shrink to AWQ / NVFP4 — runs on a free node, lands in Installed</span></div>' +
          '<div class="quantize-panel">' +
            '<div class="form-grid">' +
              '<div class="form-group"><label class="form-label">Base model (HF repo or installed)</label><input type="text" id="q-base" class="form-input" placeholder="Qwen/Qwen3.5-4B"></div>' +
              '<div class="form-group"><label class="form-label">Scheme</label><select id="q-scheme" class="form-input"><option value="awq">AWQ (W4A16 — proven on GB10)</option><option value="nvfp4">NVFP4 (Blackwell-native)</option></select></div>' +
              '<div class="form-group"><label class="form-label">Calibration samples</label><input type="number" id="q-calib" class="form-input" value="256" min="16" max="1024"></div>' +
            '</div>' +
            '<label class="form-label" style="display:block;margin-top:8px"><input type="checkbox" id="q-push"> Push result to Hugging Face</label>' +
            '<input type="text" id="q-repo" class="form-input" placeholder="HF repo name (optional; namespace from your write token)" style="margin-top:6px">' +
            '<div class="form-hint">Target node must be idle — quantization needs the full GPU memory (unload models first).</div>' +
            '<button class="btn-nvidia" id="q-submit" style="margin-top:10px">Quantize &rsaquo;</button>' +
          '</div>' +
        '</div>' +
        '<div>' +
          '<div class="training-section-head"><h3>Recent Activity</h3><span class="muted">' + recent.length + ' run' + (recent.length === 1 ? '' : 's') + '</span></div>' +
          '<div class="recent-activity-list">' + recentHtml + '</div>' +
        '</div>' +
      '</div>';

    container.querySelectorAll('[data-qs-btn]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.showNewRunWizard({ method: btn.dataset.qsBtn === 'full' ? 'full' : 'lora', distributed: btn.dataset.qsBtn === 'distributed' });
      });
    });

    var qBtn = container.querySelector('#q-submit');
    if (qBtn) qBtn.addEventListener('click', async function () {
      var base = (container.querySelector('#q-base').value || '').trim();
      if (!base) { self.toast('Enter a base model to quantize', 'error'); return; }
      var payload = {
        method: 'quantize',
        base_model: base,
        scheme: container.querySelector('#q-scheme').value,
        calib_samples: parseInt(container.querySelector('#q-calib').value, 10) || 256,
        push_to_hf: container.querySelector('#q-push').checked,
      };
      var repo = (container.querySelector('#q-repo').value || '').trim();
      if (repo) payload.hf_repo = repo;
      qBtn.disabled = true; qBtn.textContent = 'Submitting…';
      try {
        var resp = await fetch('/api/training/jobs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        var result = await resp.json();
        if (!resp.ok) {
          self.toast('Error: ' + (result.error || 'Failed to start quantize job'), 'error');
          qBtn.disabled = false; qBtn.textContent = 'Quantize ›'; return;
        }
        self.toast('Quantize job started', 'success');
        self.state.trainingTab = 'runs';
        if (self.renderTrainingSidebar) self.renderTrainingSidebar();
        self.renderTraining();
      } catch (e) {
        self.toast('Network error: ' + e.message, 'error');
        qBtn.disabled = false; qBtn.textContent = 'Quantize ›';
      }
    });
    container.querySelectorAll('.training-job-card').forEach(function (card) {
      card.addEventListener('click', function () {
        self.state.trainingTab = 'runs';
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = card.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    });
  },

  // ---- AutoData ----------------------------------------------------------
  // Δ-filtered synthetic-data generation over four AInode-served models. The run
  // executes server-side and calls back into the federated /v1 proxy, so every
  // role endpoint is this server's own loopback + the picked served-model-name.
  renderTrainingAutoData(container) {
    var self = this;
    var models = this.state.fleetModels || [];
    var opts = models.map(function (m) {
      return '<option value="' + self.esc(m) + '">' + self.esc(m) + '</option>';
    }).join('');
    var role = function (key, label, hint) {
      return '<div class="form-group"><label class="form-label">' + label + '</label>' +
        '<select id="ad-' + key + '" class="form-input">' + opts + '</select>' +
        (hint ? '<div class="form-hint">' + hint + '</div>' : '') + '</div>';
    };

    container.innerHTML =
      '<div class="training-overview">' +
        '<div class="dataset-toolbar"><div>' +
          '<h2 style="font-size:18px;font-weight:700">AutoData</h2>' +
          '<p style="font-size:12.5px;color:var(--text-muted);margin-top:4px">Generate synthetic training data from your loaded models. The Challenger writes tasks, a weak and a strong solver answer them, and the Judge keeps only what the strong model gets right and the weak one misses — the result lands in Datasets, ready to train on.</p>' +
        '</div></div>' +
        (models.length ? '' : '<div class="form-hint" style="margin-bottom:10px">No models loaded — load at least one model (Models tab) so the four roles have an endpoint to call.</div>') +
        '<div class="quantize-panel">' +
          '<div class="form-grid">' +
            role('challenger', 'Challenger (writes tasks)', 'A strong, creative model') +
            role('weak', 'Weak solver', 'Smaller/faster — should miss the hard ones') +
            role('strong', 'Strong solver', 'The teacher — its answers become the data') +
            role('judge', 'Judge (grades answers)', '') +
          '</div>' +
          '<div class="form-group" style="margin-top:8px"><label class="form-label">Task spec</label>' +
            '<textarea id="ad-task-spec" class="form-input" rows="2" placeholder="e.g. Challenging grade-school math word problems with a single numeric answer."></textarea></div>' +
          '<div class="form-group"><label class="form-label">Generator prompt (Challenger system prompt)</label>' +
            '<textarea id="ad-gen-prompt" class="form-input" rows="2" placeholder="e.g. You are an expert problem author. Write diverse, unambiguous tasks."></textarea></div>' +
          '<div class="form-grid">' +
            '<div class="form-group"><label class="form-label"># Tasks</label><input type="number" id="ad-n-tasks" class="form-input" value="20" min="1" max="500"></div>' +
            '<div class="form-group"><label class="form-label">Judge mode</label><select id="ad-judge-mode" class="form-input">' +
              '<option value="rubric">Rubric (LLM grades)</option>' +
              '<option value="verify">Verify (math/numeric, rubric fallback)</option>' +
              '<option value="exact">Exact (substring match)</option>' +
            '</select></div>' +
            '<div class="form-group"><label class="form-label">Dataset name</label><input type="text" id="ad-name" class="form-input" placeholder="optional"></div>' +
          '</div>' +
          '<label class="form-label" style="display:block;margin-top:8px"><input type="checkbox" id="ad-meta"> Meta-optimize — rewrite the generator prompt over several rounds to lift yield</label>' +
          '<button class="btn-nvidia" id="ad-run" style="margin-top:10px"' + (models.length ? '' : ' disabled') + '>Run AutoData &rsaquo;</button>' +
        '</div>' +
        '<div id="autodata-status" style="margin-top:14px"></div>' +
      '</div>';

    this._renderAutoDataStatus();  // restore an in-flight/last run if we re-entered the tab

    var runBtn = container.querySelector('#ad-run');
    if (runBtn) runBtn.addEventListener('click', function () { self._startAutoDataRun(container); });
  },

  async _startAutoDataRun(container) {
    var self = this;
    var val = function (sel) { var el = container.querySelector(sel); return el ? el.value.trim() : ''; };
    var pick = function (sel) { var el = container.querySelector(sel); return el ? el.value : ''; };
    var taskSpec = val('#ad-task-spec'), genPrompt = val('#ad-gen-prompt');
    if (!taskSpec || !genPrompt) { self.toast('Task spec and generator prompt are required', 'error'); return; }
    if (!pick('#ad-challenger')) { self.toast('Load at least one model first', 'error'); return; }
    // ponytail: the federated /v1 proxy is this same server; loopback + the served-model-name
    // routes to whichever node serves it. Assumes the default api_port when the port is blank.
    var base = 'http://localhost:' + (window.location.port || '8000') + '/v1';
    var ep = function (sel) { return { url: base, model: pick(sel) }; };
    var body = {
      config: {
        task_spec: taskSpec, gen_prompt: genPrompt,
        challenger: ep('#ad-challenger'), weak: ep('#ad-weak'),
        strong: ep('#ad-strong'), judge: ep('#ad-judge'),
        n_tasks: parseInt(val('#ad-n-tasks'), 10) || 20,
        judge_mode: pick('#ad-judge-mode'),
      },
      meta: container.querySelector('#ad-meta').checked,
    };
    var name = val('#ad-name');
    if (name) body.name = name;

    var runBtn = container.querySelector('#ad-run');
    runBtn.disabled = true; runBtn.textContent = 'Starting…';
    try {
      var resp = await fetch('/api/training/autodata', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      var data = await resp.json();
      if (!resp.ok) {
        self.toast('Error: ' + (data.error || 'Failed to start AutoData'), 'error');
        runBtn.disabled = false; runBtn.textContent = 'Run AutoData ›'; return;
      }
      self.state.autodataRun = { run_id: data.run_id, status: 'running', meta: body.meta };
      self.toast('AutoData run started', 'success');
      self._renderAutoDataStatus();
      self._pollAutoData();
    } catch (e) {
      self.toast('Network error: ' + e.message, 'error');
      runBtn.disabled = false; runBtn.textContent = 'Run AutoData ›';
    }
  },

  _pollAutoData() {
    var self = this;
    if (this._autodataPollTimer) clearTimeout(this._autodataPollTimer);
    var tick = async function () {
      var cur = self.state.autodataRun;
      if (!cur || !cur.run_id) return;
      var data = await self.fetchJSON('/api/training/autodata/' + encodeURIComponent(cur.run_id));
      if (data) {
        self.state.autodataRun = data;
        self._renderAutoDataStatus();
        if (data.status === 'completed' || data.status === 'failed') {
          var btn = document.getElementById('ad-run');
          if (btn) { btn.disabled = false; btn.textContent = 'Run AutoData ›'; }
          if (data.status === 'completed' && data.dataset_id) {
            // pull the new dataset into state so it shows up the moment they open Datasets
            var ds = await self.fetchJSON('/api/datasets');
            if (ds && ds.datasets) self.state.datasets = ds.datasets;
            self._renderAutoDataStatus();
          }
          return;  // terminal — stop polling
        }
      }
      self._autodataPollTimer = setTimeout(tick, 3000);
    };
    tick();
  },

  _renderAutoDataStatus() {
    var el = document.getElementById('autodata-status');
    if (!el) return;
    var self = this;
    var run = this.state.autodataRun;
    if (!run) { el.innerHTML = ''; return; }
    var card = function (inner) {
      return '<div style="padding:14px 16px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;font-size:13px;line-height:1.6">' + inner + '</div>';
    };
    if (run.status === 'running') {
      el.innerHTML = card('<span class="job-status-badge status-running">Running</span> Generating &amp; Δ-filtering tasks…' + (run.meta ? ' (meta-optimizing the generator prompt)' : ''));
      return;
    }
    if (run.status === 'failed') {
      el.innerHTML = card('<span class="job-status-badge status-failed">Failed</span> ' + self.esc(run.error || 'Unknown error'));
      return;
    }
    if (run.status === 'completed') {
      var r = run.report || {};
      var stats = run.meta
        ? 'Kept <strong>' + (r.kept != null ? r.kept : '—') + '</strong> &middot; best yield <strong>' +
            (r.best_yield != null ? r.best_yield + '%' : '—') + '</strong> &middot; <strong>' + (r.rounds != null ? r.rounds : '—') + '</strong> rounds'
        : 'Kept <strong>' + (r.kept != null ? r.kept : '—') + '</strong> of <strong>' + (r.total != null ? r.total : '—') +
            '</strong> &middot; yield <strong>' + (r.yield_pct != null ? r.yield_pct + '%' : '—') + '</strong> ' +
            '<span class="muted">(too-easy ' + (r.too_easy || 0) + ' &middot; too-hard ' + (r.too_hard || 0) + ' &middot; errors ' + (r.errors || 0) + ')</span>';
      var tail = run.dataset_id
        ? '<div style="margin-top:10px"><button class="btn-nvidia" id="ad-goto-dataset">Open in Datasets &rsaquo;</button></div>'
        : '<div style="margin-top:10px" class="muted">No rows kept — nothing to save. Try more tasks, or pick a wider weak/strong gap.</div>';
      el.innerHTML = card('<span class="job-status-badge status-completed">Completed</span> ' + stats + tail);
      var goto = document.getElementById('ad-goto-dataset');
      if (goto) goto.addEventListener('click', function () {
        self.state.trainingTab = 'datasets';
        self._autodataViewInitialized = false;
        self.renderTrainingSidebar();
        self.renderTraining();
      });
    }
  },

  // ---- Datasets ----------------------------------------------------------
  renderTrainingDatasets(container) {
    var self = this;
    var datasets = this.state.datasets || [];
    var list = datasets.length
      ? '<div class="dataset-grid">' + datasets.map(function (d) { return self.renderDatasetCard(d); }).join('') + '</div>'
      : '<div class="training-empty"><h3>No datasets yet</h3><p>Upload a file, reference a HuggingFace dataset, or point at a local path to get started.</p></div>';

    container.innerHTML =
      '<div class="training-overview">' +
        '<div class="dataset-toolbar">' +
          '<div><h2 style="font-size:18px;font-weight:700">Datasets</h2>' +
          '<p style="font-size:12.5px;color:var(--text-muted);margin-top:4px">Training data you can attach to any run.</p></div>' +
          '<button class="btn-nvidia" id="dataset-add-btn">+ Add Dataset</button>' +
        '</div>' +
        list +
      '</div>';

    var addBtn = document.getElementById('dataset-add-btn');
    if (addBtn) addBtn.addEventListener('click', function () { self.showDatasetAddModal(); });

    container.querySelectorAll('[data-ds-delete]').forEach(function (b) {
      b.addEventListener('click', async function (e) {
        e.stopPropagation();
        if (!confirm('Delete this dataset?')) return;
        var resp = await fetch('/api/datasets/' + encodeURIComponent(b.dataset.dsDelete), { method: 'DELETE' });
        if (resp.ok) { self.toast('Dataset deleted', 'success'); self.renderTraining(); }
        else { self.toast('Failed to delete dataset', 'error'); }
      });
    });
    container.querySelectorAll('[data-ds-use]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        self.showNewRunWizard({ dataset_id: b.dataset.dsUse });
      });
    });
    container.querySelectorAll('[data-ds-preview]').forEach(function (b) {
      b.addEventListener('click', async function (e) {
        e.stopPropagation();
        var id = b.dataset.dsPreview;
        var target = document.getElementById('ds-preview-' + id);
        if (!target) return;
        if (target.style.display !== 'none' && target.dataset.loaded === '1') {
          target.style.display = 'none';
          target.dataset.loaded = '0';
          return;
        }
        target.style.display = '';
        target.textContent = 'Loading preview…';
        var data = await self.fetchJSON('/api/datasets/' + encodeURIComponent(id) + '/preview');
        if (!data) { target.textContent = 'Preview unavailable.'; return; }
        target.textContent = JSON.stringify(data.samples, null, 2);
        target.dataset.loaded = '1';
      });
    });
  },

  renderDatasetCard(d) {
    var name = this.esc(d.name || d.id);
    var badge = '<span class="dataset-badge src-' + this.esc(d.source) + '">' + this.esc(d.source) + '</span>';
    var fmt = d.format ? '<span class="dataset-badge">' + this.esc(d.format) + '</span>' : '';
    var sz = this.formatBytes(d.size_bytes || 0);
    var samples = (d.samples || 0).toLocaleString();
    var created = d.created_at ? new Date(d.created_at * 1000).toLocaleDateString() : '—';

    return '<div class="dataset-card">' +
      '<div class="dataset-card-top">' +
        '<div><div class="dataset-card-name">' + name + '</div>' +
        '<div class="dataset-card-id">' + this.esc(d.id) + '</div></div>' +
        '<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">' + badge + fmt + '</div>' +
      '</div>' +
      '<div class="dataset-card-stats">' +
        '<span>' + samples + ' samples</span>' +
        '<span>' + sz + '</span>' +
        '<span>' + this.esc(created) + '</span>' +
      '</div>' +
      '<div class="dataset-card-path" title="' + this.esc(d.path || '') + '">' + this.esc(d.path || '') + '</div>' +
      '<div class="dataset-card-actions">' +
        '<button class="btn-nvidia btn-sm" data-ds-use="' + this.esc(d.id) + '">Use in Run</button>' +
        '<button class="btn-ghost btn-sm" data-ds-preview="' + this.esc(d.id) + '">Preview</button>' +
        '<button class="btn-danger btn-sm" data-ds-delete="' + this.esc(d.id) + '">Delete</button>' +
      '</div>' +
      '<div class="dataset-preview-samples" id="ds-preview-' + this.esc(d.id) + '" style="display:none" data-loaded="0"></div>' +
    '</div>';
  },

  // ---- Runs --------------------------------------------------------------
  renderTrainingRuns(container) {
    var self = this;
    var jobs = this.state.trainingJobs || [];
    var filter = this.state.runsFilter || 'all';
    if (filter !== 'all') jobs = jobs.filter(function (j) { return j.status === filter; });
    jobs = jobs.slice().sort(function (a, b) { return (b.start_time || 0) - (a.start_time || 0); });

    var filters = ['all', 'running', 'completed', 'failed'];
    var filterBar = '<div class="runs-filter-bar">' + filters.map(function (f) {
      var a = f === self.state.runsFilter ? ' active' : '';
      return '<button class="pill' + a + '" data-filter="' + f + '">' + f.charAt(0).toUpperCase() + f.slice(1) + '</button>';
    }).join('') + '</div>';

    var body;
    if (jobs.length === 0) {
      body = '<div class="training-empty"><h3>No runs found</h3><p>Try a different filter or start a new run from the sidebar.</p></div>';
    } else {
      var rows = jobs.map(function (j) {
        var s = self._statusInfo(j.status);
        var cfg = j.config || {};
        var modelShort = (cfg.base_model || '').split('/').pop() || '—';
        var runName = cfg.run_name || modelShort;
        var started = j.start_time ? new Date(j.start_time * 1000).toLocaleString() : '—';
        var elapsed = j.elapsed_seconds ? self.formatUptime(Math.round(j.elapsed_seconds)) : '—';
        var method = (cfg.method || 'lora').toUpperCase();
        return '<tr class="run-row" data-job-id="' + self.esc(j.job_id) + '">' +
          '<td><strong>' + self.esc(runName) + '</strong><div class="mono" style="color:var(--text-muted);font-size:10.5px">' + self.esc(j.job_id) + '</div></td>' +
          '<td class="mono">' + self.esc(modelShort) + '</td>' +
          '<td>' + method + '</td>' +
          '<td><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></td>' +
          '<td>' + self.esc(started) + '</td>' +
          '<td class="mono">' + self.esc(elapsed) + '</td>' +
          '<td class="mono">' + (j.current_loss != null ? j.current_loss.toFixed(4) : '—') + '</td>' +
          '</tr>';
      }).join('');

      body = '<table class="runs-table">' +
        '<thead><tr><th>Run</th><th>Model</th><th>Method</th><th>Status</th><th>Started</th><th>Duration</th><th>Loss</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
        '</table>';
    }

    container.innerHTML = '<div class="training-overview">' +
      '<div class="training-section-head"><h3>Runs</h3><span class="muted">' + (this.state.trainingJobs || []).length + ' total</span></div>' +
      filterBar + body +
    '</div>';

    container.querySelectorAll('[data-filter]').forEach(function (p) {
      p.addEventListener('click', function () {
        self.state.runsFilter = p.dataset.filter;
        self.renderTrainingRuns(container);
      });
    });
    container.querySelectorAll('.run-row').forEach(function (row) {
      row.addEventListener('click', function () {
        self.state.trainingView = 'detail';
        self.state.trainingDetailId = row.dataset.jobId;
        self.state.trainingLossData = [];
        self.renderTraining();
      });
    });
  },

  // ---- Templates ---------------------------------------------------------
  renderTrainingTemplates(container) {
    var self = this;
    var templates = this.state.trainingTemplates || [];
    var cards = templates.length ? templates.map(function (t) {
      var shape = typeof t.sample_shape === 'object'
        ? JSON.stringify(t.sample_shape, null, 2)
        : String(t.sample_shape || '');
      return '<div class="template-card" data-template-id="' + self.esc(t.id) + '">' +
        '<h4>' + self.esc(t.name) + '</h4>' +
        '<p>' + self.esc(t.description || '') + '</p>' +
        '<div class="template-shape">' + self.esc(shape) + '</div>' +
        '<div class="template-meta-row">' +
          '<span>&#9830; ' + self.esc((t.method || 'lora').toUpperCase()) + '</span>' +
          '<span>&#9200; ' + self.esc(t.estimated_time || '—') + '</span>' +
          (t.distributed ? '<span>&#9733; Distributed</span>' : '') +
        '</div>' +
        '<button class="btn-nvidia btn-sm" data-use-template="' + self.esc(t.id) + '">Use this template</button>' +
      '</div>';
    }).join('') : '<div class="training-empty"><h3>No templates available</h3></div>';

    container.innerHTML = '<div class="training-overview">' +
      '<div class="training-section-head"><h3>Templates</h3><span class="muted">Starter recipes</span></div>' +
      '<div class="templates-gallery">' + cards + '</div>' +
    '</div>';

    container.querySelectorAll('[data-use-template]').forEach(function (b) {
      b.addEventListener('click', function () {
        var tpl = (self.state.trainingTemplates || []).find(function (t) { return t.id === b.dataset.useTemplate; });
        if (!tpl) return;
        self.showNewRunWizard({
          template_id: tpl.id,
          method: tpl.method || 'lora',
          distributed: !!tpl.distributed,
          num_epochs: tpl.recommended_epochs,
          batch_size: tpl.recommended_batch_size,
          learning_rate: tpl.recommended_lr,
        });
      });
    });
  },

  renderJobCard(job) {
    var statusMap = {
      pending: { cls: 'status-pending', label: 'Pending' },
      running: { cls: 'status-running', label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed: { cls: 'status-failed', label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    var s = statusMap[job.status] || { cls: '', label: job.status };
    var modelShort = job.config?.base_model?.split('/').pop() || 'Unknown';
    var methodLabel = (job.config?.method || 'lora').toUpperCase();
    var started = job.start_time ? new Date(job.start_time * 1000).toLocaleString() : 'Not started';
    var elapsed = job.elapsed_seconds ? this.formatUptime(Math.round(job.elapsed_seconds)) : '--';

    var html = '<div class="training-job-card" data-job-id="' + this.esc(job.job_id) + '">' +
      '<div class="job-card-top">' +
      '<div><div class="job-card-model">' + this.esc(modelShort) + '</div>' +
      '<div class="job-card-meta"><span class="job-method-badge">' + methodLabel + '</span> ' + this.esc(job.job_id) + '</div></div>' +
      '<span class="job-status-badge ' + s.cls + '">' + s.label + '</span>' +
      '</div>';

    if (job.status === 'running') {
      html += '<div class="job-card-progress">' +
        '<div class="progress-bar"><div class="progress-fill" style="width:' + (job.progress || 0) + '%"></div></div>' +
        '<div class="job-progress-text">' + (job.progress || 0).toFixed(1) + '% &middot; Epoch ' + (job.current_epoch || 0) + '/' + (job.config?.num_epochs || '?') +
        (job.current_loss != null ? ' &middot; Loss: ' + job.current_loss.toFixed(4) : '') + '</div></div>';
    }

    html += '<div class="job-card-footer"><span>Started: ' + started + '</span><span>Duration: ' + elapsed + '</span></div></div>';
    return html;
  },

  // =======================================================================
  //  NEW RUN WIZARD (modal, multi-step)
  // =======================================================================

  async showNewRunWizard(prefill) {
    var self = this;
    // Gather fresh data
    var dsets = this.state.datasets.length
      ? this.state.datasets
      : (((await this.fetchJSON('/api/datasets')) || {}).datasets || []);
    this.state.datasets = dsets;

    // Only offer models that are actually on disk — a run against a
    // not-downloaded model would fail. When none are downloaded we leave the
    // list empty and show an empty-state hint (the free-text HF-id input in
    // step 1 still works). The backend only ever sets `downloaded`.
    var modelsResp = await this.fetchJSON('/api/models');
    var modelList = [];
    if (modelsResp && Array.isArray(modelsResp.models)) {
      modelList = modelsResp.models
        .filter(function (m) { return m.downloaded; })
        // base_model = the canonical HF repo id (cards know it), NOT the on-disk
        // slug — the containerized runner hands base_model to
        // AutoTokenizer.from_pretrained, which rejects a '--' slug (HFValidationError).
        .map(function (m) {
          var repo = m.hf_repo || m.id || m.name || m.repo_id;
          return { id: repo, name: m.name || repo, size: m.size || m.size_gb || '' };
        });
    }

    var initial = Object.assign({
      base_model: modelList[0] ? modelList[0].id : '',
      dataset_id: '',
      method: 'lora',
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 2e-4,
      max_seq_length: 2048,
      lora_rank: 16,
      lora_alpha: 32,
      gradient_accumulation_steps: 1,
      warmup_steps: 0,
      weight_decay: 0.0,
      use_gradient_checkpointing: false,
      distributed: false,
      num_nodes: 1,
      run_name: '',
      template_id: null,
      step: 1,
      models: modelList,
      datasets: dsets,
    }, prefill || {});
    this.state.wizardState = initial;

    // Build overlay
    var existing = document.querySelector('.model-detail-modal-overlay.wizard-overlay');
    if (existing) existing.remove();

    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay wizard-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal wizard-modal">' +
        '<div class="md-header">' +
          '<div class="md-header-left"><div class="md-icon">&#9881;</div><div><div style="font-weight:700;font-size:15px">New Training Run</div><div style="font-size:11.5px;color:var(--text-muted)">Configure and launch a fine-tune</div></div></div>' +
          '<button class="btn-ghost btn-sm" id="wizard-close">Close</button>' +
        '</div>' +
        '<div class="wizard-body">' +
          '<div class="wizard-steps" id="wizard-steps"></div>' +
          '<div class="wizard-panel" id="wizard-panel"></div>' +
        '</div>' +
        '<div class="wizard-footer">' +
          '<button class="btn-ghost" id="wizard-prev">&larr; Back</button>' +
          '<span class="estimate-pill" id="wizard-estimate"></span>' +
          '<div class="spacer"></div>' +
          '<button class="btn-nvidia" id="wizard-next">Next &rarr;</button>' +
        '</div>' +
      '</div>';

    document.body.appendChild(overlay);

    overlay.addEventListener('click', function (e) { if (e.target === overlay) self.closeWizard(); });
    document.getElementById('wizard-close').addEventListener('click', function () { self.closeWizard(); });
    document.getElementById('wizard-prev').addEventListener('click', function () { self.wizardGo(-1); });
    document.getElementById('wizard-next').addEventListener('click', function () { self.wizardGo(+1); });

    this.renderWizardStep();
  },

  closeWizard() {
    var o = document.querySelector('.wizard-overlay');
    if (o) o.remove();
    this.state.wizardState = null;
  },

  wizardGo(delta) {
    var s = this.state.wizardState;
    if (!s) return;
    var next = s.step + delta;
    if (next < 1) return;
    if (next > 5) { this.submitWizard(); return; }
    // Validate current step before moving forward
    if (delta > 0 && !this.validateWizardStep()) return;
    s.step = next;
    this.renderWizardStep();
  },

  validateWizardStep() {
    var s = this.state.wizardState;
    if (s.step === 1 && !s.base_model) { this.toast('Pick a base model to continue', 'error'); return false; }
    if (s.step === 2 && !s.dataset_id && !s.dataset_path_inline) { this.toast('Select or add a dataset', 'error'); return false; }
    return true;
  },

  renderWizardStep() {
    var s = this.state.wizardState;
    if (!s) return;
    var self = this;
    var stepDefs = [
      { n: 1, label: 'Base Model' },
      { n: 2, label: 'Dataset' },
      { n: 3, label: 'Method' },
      { n: 4, label: 'Distribution' },
      { n: 5, label: 'Review' },
    ];

    document.getElementById('wizard-steps').innerHTML = stepDefs.map(function (d) {
      var cls = d.n === s.step ? ' active' : (d.n < s.step ? ' done' : '');
      return '<div class="wizard-step-nav' + cls + '"><span class="step-num">' + d.n + '</span>' + d.label + '</div>';
    }).join('');

    var panel = document.getElementById('wizard-panel');
    var nextBtn = document.getElementById('wizard-next');
    if (s.step === 5) nextBtn.textContent = 'Launch Training Run';
    else nextBtn.textContent = 'Next →';
    document.getElementById('wizard-prev').style.visibility = s.step === 1 ? 'hidden' : '';

    switch (s.step) {
      case 1: panel.innerHTML = this._renderWizardStep1(); break;
      case 2: panel.innerHTML = this._renderWizardStep2(); break;
      case 3: panel.innerHTML = this._renderWizardStep3(); break;
      case 4: panel.innerHTML = this._renderWizardStep4(); break;
      case 5: panel.innerHTML = this._renderWizardStep5(); break;
    }
    this._bindWizardStep();
    this._updateEstimate();
  },

  _renderWizardStep1() {
    var s = this.state.wizardState;
    var models = s.models || [];
    var grid;
    if (models.length) {
      grid = '<div class="wizard-option-grid">' + models.map(function (m) {
        var active = m.id === s.base_model ? ' active' : '';
        return '<div class="wizard-option' + active + '" data-model="' + AINode.esc(m.id) + '">' +
          '<div class="wizard-option-title">' + AINode.esc(m.name || m.id) + '</div>' +
          '<div class="wizard-option-sub">' + AINode.esc(m.id) + (m.size ? ' · ' + AINode.esc(m.size) : '') + '</div>' +
          '</div>';
      }).join('') + '</div>';
    } else {
      grid = '<div class="wizard-empty" style="padding:16px;color:var(--text-muted);font-size:13px">' +
        'No downloaded models — download one from the Models page, or enter an HF id below.</div>';
    }
    return '<h3>Choose a Base Model</h3>' +
      '<p class="panel-sub">Downloaded models are shown. Your fine-tune will build on top of this one.</p>' +
      grid +
      '<div class="form-group" style="margin-top:16px"><label class="form-label">Or enter a HuggingFace model ID</label>' +
      '<input type="text" id="wz-custom-model" class="form-input" placeholder="org/model" value="' + AINode.esc(s.custom_model || '') + '"></div>';
  },

  _renderWizardStep2() {
    var s = this.state.wizardState;
    var datasets = s.datasets || [];
    var options = datasets.map(function (d) {
      var active = d.id === s.dataset_id ? ' active' : '';
      return '<div class="wizard-option' + active + '" data-dataset="' + AINode.esc(d.id) + '">' +
        '<div class="wizard-option-title">' + AINode.esc(d.name) + '</div>' +
        '<div class="wizard-option-sub">' + (d.samples || 0).toLocaleString() + ' samples · ' + AINode.esc(d.source) + ' · ' + AINode.esc(d.format || '') + '</div>' +
        '</div>';
    }).join('');
    var body = datasets.length
      ? '<div class="wizard-option-grid">' + options + '</div>'
      : '<div class="training-empty" style="padding:24px"><h3>No datasets yet</h3><p>Add one below or via the Datasets tab.</p></div>';

    return '<h3>Select Training Dataset</h3>' +
      '<p class="panel-sub">Reference a saved dataset, or paste a path / HuggingFace ID inline.</p>' +
      body +
      '<div class="form-group" style="margin-top:16px">' +
        '<label class="form-label">Or inline dataset path</label>' +
        '<input type="text" id="wz-inline-ds" class="form-input" placeholder="~/.ainode/datasets/file.jsonl or alpaca.jsonl" value="' + AINode.esc(s.dataset_path_inline || '') + '">' +
        '<div class="form-hint">Fields accepted: <code>text</code>, or <code>instruction</code>+<code>output</code>, or <code>prompt</code>+<code>completion</code>.</div>' +
      '</div>';
  },

  _renderWizardStep3() {
    var s = this.state.wizardState;
    var methods = [
      { id: 'lora',  title: 'LoRA',      sub: 'Train small adapter weights. Fast, low memory.' },
      { id: 'qlora', title: 'QLoRA',     sub: '4-bit quantized LoRA. Fits huge models in limited memory.' },
      { id: 'full',  title: 'Full',      sub: 'Update all weights. Highest quality, highest memory.' },
    ];
    var methodCards = methods.map(function (m) {
      var active = m.id === s.method ? ' active' : '';
      return '<div class="wizard-option' + active + '" data-method="' + m.id + '">' +
        '<div class="wizard-option-title">' + m.title + '</div>' +
        '<div class="wizard-option-sub">' + m.sub + '</div>' +
        '</div>';
    }).join('');

    var loraBlock = (s.method === 'lora' || s.method === 'qlora') ?
      '<div class="form-grid">' +
        '<div class="form-group"><label class="form-label">LoRA Rank</label><input type="number" id="wz-lora-rank" class="form-input" value="' + s.lora_rank + '" min="1" max="256"></div>' +
        '<div class="form-group"><label class="form-label">LoRA Alpha</label><input type="number" id="wz-lora-alpha" class="form-input" value="' + s.lora_alpha + '" min="1" max="512"></div>' +
      '</div>' : '';

    return '<h3>Training Method</h3>' +
      '<p class="panel-sub">LoRA is recommended. Full fine-tune needs significant GPU memory.</p>' +
      '<div class="wizard-option-grid">' + methodCards + '</div>' +
      '<div style="margin-top:20px">' +
        '<div class="form-grid">' +
          '<div class="form-group"><label class="form-label">Epochs</label><input type="number" id="wz-epochs" class="form-input" value="' + s.num_epochs + '" min="1" max="100"></div>' +
          '<div class="form-group"><label class="form-label">Batch Size</label><input type="number" id="wz-batch" class="form-input" value="' + s.batch_size + '" min="1" max="128"></div>' +
          '<div class="form-group"><label class="form-label">Learning Rate</label><input type="text" id="wz-lr" class="form-input" value="' + s.learning_rate + '"></div>' +
          '<div class="form-group"><label class="form-label">Max Seq Length</label><input type="number" id="wz-seq" class="form-input" value="' + s.max_seq_length + '" min="128" step="128"></div>' +
          '<div class="form-group"><label class="form-label">Grad Accumulation</label><input type="number" id="wz-ga" class="form-input" value="' + s.gradient_accumulation_steps + '" min="1"></div>' +
          '<div class="form-group"><label class="form-label">Warmup Steps</label><input type="number" id="wz-warmup" class="form-input" value="' + s.warmup_steps + '" min="0"></div>' +
        '</div>' +
        loraBlock +
        '<div style="margin-top:12px"><button type="button" class="btn-ghost btn-sm" id="wz-smart">&#9733; Smart Defaults</button></div>' +
      '</div>';
  },

  _renderWizardStep4() {
    var s = this.state.wizardState;
    var nodeCount = (this.state.nodes || []).length || 1;
    var dropdownMax = Math.max(1, nodeCount);
    var nodesInput = '';
    for (var i = 1; i <= Math.max(3, dropdownMax); i++) {
      nodesInput += '<button type="button" class="node-dot' + (i === s.num_nodes ? ' active' : '') + '" data-nodes="' + i + '">' + i + '</button>';
    }
    return '<h3>Distribution</h3>' +
      '<p class="panel-sub">Single node is the default. Enable distributed DDP to shard across multiple cluster nodes.</p>' +
      '<div class="wizard-option-grid">' +
        '<div class="wizard-option' + (!s.distributed ? ' active' : '') + '" data-dist="0">' +
          '<div class="wizard-option-title">Single Node</div>' +
          '<div class="wizard-option-sub">Run on this node\'s GPU(s).</div>' +
        '</div>' +
        '<div class="wizard-option' + (s.distributed ? ' active' : '') + '" data-dist="1">' +
          '<div class="wizard-option-title">Multi-node DDP</div>' +
          '<div class="wizard-option-sub">Data-parallel across ' + nodeCount + ' online node' + (nodeCount === 1 ? '' : 's') + '.</div>' +
        '</div>' +
      '</div>' +
      (s.distributed ? '<div style="margin-top:16px"><label class="form-label">Number of nodes</label><div class="node-selector">' + nodesInput + '</div></div>' : '') +
      '<div class="form-group" style="margin-top:16px"><label class="form-label">Run name (optional)</label><input type="text" id="wz-run-name" class="form-input" placeholder="alpaca-3b-epoch3" value="' + AINode.esc(s.run_name || '') + '"></div>';
  },

  _renderWizardStep5() {
    var s = this.state.wizardState;
    var self = this;
    var summaryRow = function (k, v) { return '<div class="k">' + k + '</div><div class="v">' + AINode.esc(String(v == null ? '—' : v)) + '</div>'; };
    var dsname = '—';
    if (s.dataset_id) {
      var match = (s.datasets || []).find(function (d) { return d.id === s.dataset_id; });
      if (match) dsname = match.name + ' (' + match.id + ')';
    } else if (s.dataset_path_inline) {
      dsname = s.dataset_path_inline;
    }
    return '<h3>Review & Launch</h3>' +
      '<p class="panel-sub">Double-check the config, then launch. You can cancel any time from the run detail view.</p>' +
      '<div class="wizard-summary">' +
        summaryRow('Run name', s.run_name || 'auto') +
        summaryRow('Base model', s.base_model) +
        summaryRow('Dataset', dsname) +
        summaryRow('Method', (s.method || 'lora').toUpperCase()) +
        summaryRow('Epochs', s.num_epochs) +
        summaryRow('Batch size', s.batch_size) +
        summaryRow('Learning rate', s.learning_rate) +
        summaryRow('Max seq len', s.max_seq_length) +
        summaryRow('Grad accum', s.gradient_accumulation_steps) +
        summaryRow('Distributed', s.distributed ? (s.num_nodes + ' nodes') : 'single node') +
      '</div>' +
      '<div class="form-group" style="margin-top:16px"><label style="display:flex;align-items:center;gap:8px;color:var(--text-secondary);font-size:12.5px"><input type="checkbox" id="wz-save-template"> Save as a custom template</label></div>';
  },

  _bindWizardStep() {
    var self = this;
    var s = this.state.wizardState;
    // Step 1
    document.querySelectorAll('.wizard-option[data-model]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.base_model = el.dataset.model;
        s.custom_model = '';
        self.renderWizardStep();
      });
    });
    var custom = document.getElementById('wz-custom-model');
    if (custom) custom.addEventListener('change', function () {
      s.custom_model = custom.value.trim();
      if (s.custom_model) s.base_model = s.custom_model;
    });

    // Step 2
    document.querySelectorAll('.wizard-option[data-dataset]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.dataset_id = el.dataset.dataset;
        s.dataset_path_inline = '';
        self.renderWizardStep();
      });
    });
    var inlineDs = document.getElementById('wz-inline-ds');
    if (inlineDs) inlineDs.addEventListener('change', function () {
      s.dataset_path_inline = inlineDs.value.trim();
      if (s.dataset_path_inline) s.dataset_id = '';
    });

    // Step 3
    document.querySelectorAll('.wizard-option[data-method]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.method = el.dataset.method;
        self.renderWizardStep();
      });
    });
    var hook = function (id, key, parser) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('change', function () {
        var v = parser ? parser(el.value) : el.value;
        if (!isNaN(v) || typeof v === 'string') s[key] = v;
        self._updateEstimate();
      });
    };
    hook('wz-epochs', 'num_epochs', parseInt);
    hook('wz-batch', 'batch_size', parseInt);
    hook('wz-lr', 'learning_rate', parseFloat);
    hook('wz-seq', 'max_seq_length', parseInt);
    hook('wz-ga', 'gradient_accumulation_steps', parseInt);
    hook('wz-warmup', 'warmup_steps', parseInt);
    hook('wz-lora-rank', 'lora_rank', parseInt);
    hook('wz-lora-alpha', 'lora_alpha', parseInt);

    var smart = document.getElementById('wz-smart');
    if (smart) smart.addEventListener('click', function () {
      var model = (s.base_model || '').toLowerCase();
      var big = /70b|72b|405b/.test(model);
      s.batch_size = big ? 1 : 4;
      s.gradient_accumulation_steps = big ? 8 : 2;
      s.learning_rate = s.method === 'full' ? 5e-5 : 2e-4;
      s.warmup_steps = 50;
      s.max_seq_length = 2048;
      self.toast('Smart defaults applied', 'success');
      self.renderWizardStep();
    });

    // Step 4
    document.querySelectorAll('.wizard-option[data-dist]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.distributed = el.dataset.dist === '1';
        if (!s.distributed) s.num_nodes = 1;
        self.renderWizardStep();
      });
    });
    document.querySelectorAll('.node-dot[data-nodes]').forEach(function (el) {
      el.addEventListener('click', function () {
        s.num_nodes = parseInt(el.dataset.nodes, 10);
        self.renderWizardStep();
      });
    });
    var runName = document.getElementById('wz-run-name');
    if (runName) runName.addEventListener('input', function () { s.run_name = runName.value; });
  },

  async _updateEstimate() {
    var s = this.state.wizardState;
    if (!s) return;
    var pill = document.getElementById('wizard-estimate');
    if (!pill) return;
    try {
      var sampleCount = 0;
      if (s.dataset_id) {
        var match = (s.datasets || []).find(function (d) { return d.id === s.dataset_id; });
        if (match) sampleCount = match.samples || 0;
      }
      var resp = await fetch('/api/training/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model: s.base_model,
          dataset_path: s.dataset_path_inline || 'placeholder',
          dataset_id: s.dataset_id,
          method: s.method,
          num_epochs: s.num_epochs,
          batch_size: s.batch_size,
          max_seq_length: s.max_seq_length,
          gradient_accumulation_steps: s.gradient_accumulation_steps,
          distributed: s.distributed,
          num_nodes: s.num_nodes,
          sample_count: sampleCount,
        }),
      });
      if (!resp.ok) { pill.textContent = ''; return; }
      var data = await resp.json();
      var parts = [];
      if (data.memory_gb_per_node) parts.push('&#9881; ~' + data.memory_gb_per_node.toFixed(0) + ' GB/node');
      if (data.estimated_seconds) parts.push('&#9201; ~' + this.formatUptime(Math.round(data.estimated_seconds)));
      if (data.samples_per_sec) parts.push('&rarr; ' + data.samples_per_sec.toFixed(1) + ' samp/s');
      pill.innerHTML = parts.join(' · ');
    } catch (e) {
      pill.textContent = '';
    }
  },

  async submitWizard() {
    var s = this.state.wizardState;
    if (!s) return;
    var payload = {
      base_model: s.base_model,
      method: s.method,
      num_epochs: s.num_epochs,
      batch_size: s.batch_size,
      learning_rate: parseFloat(s.learning_rate),
      max_seq_length: s.max_seq_length,
      gradient_accumulation_steps: s.gradient_accumulation_steps,
      warmup_steps: s.warmup_steps,
      weight_decay: s.weight_decay,
      distributed: s.distributed,
      num_nodes: s.num_nodes,
    };
    if (s.run_name) payload.run_name = s.run_name;
    if (s.template_id) payload.template_id = s.template_id;
    if (s.method === 'lora' || s.method === 'qlora') {
      payload.lora_rank = s.lora_rank;
      payload.lora_alpha = s.lora_alpha;
    }
    if (s.dataset_id) {
      payload.dataset_id = s.dataset_id;
      payload.dataset_path = 'pending';  // backend will overwrite via dataset_id resolution
    } else {
      payload.dataset_path = s.dataset_path_inline;
    }

    try {
      var resp = await fetch('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      var result = await resp.json();
      if (!resp.ok) {
        this.toast('Error: ' + (result.error || 'Failed to create job'), 'error');
        return;
      }
      this.toast('Training run launched', 'success');
      this.closeWizard();
      this.state.trainingTab = 'runs';
      this.state.trainingView = 'detail';
      this.state.trainingDetailId = result.job_id;
      this.state.trainingLossData = [];
      this.renderTrainingSidebar();
      this.renderTraining();
    } catch (err) {
      this.toast('Network error: ' + err.message, 'error');
    }
  },

  // =======================================================================
  //  DATASET ADD MODAL (Upload / HF / Local / URL)
  // =======================================================================

  showDatasetAddModal() {
    var self = this;
    var existing = document.querySelector('.model-detail-modal-overlay.dataset-overlay');
    if (existing) existing.remove();
    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay dataset-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal" style="width:min(640px,94vw)">' +
        '<div class="md-header">' +
          '<div class="md-header-left"><div class="md-icon">&#8864;</div><div><div style="font-weight:700;font-size:15px">Add Dataset</div></div></div>' +
          '<button class="btn-ghost btn-sm" id="dsm-close">Close</button>' +
        '</div>' +
        '<div style="padding:20px 24px">' +
          '<div class="pill-group">' +
            '<button class="pill active" data-src="upload">Upload</button>' +
            '<button class="pill" data-src="huggingface">HuggingFace</button>' +
            '<button class="pill" data-src="local">Local Path</button>' +
            '<button class="pill" data-src="url">URL</button>' +
          '</div>' +
          '<div id="dsm-panel"></div>' +
        '</div>' +
        '<div class="wizard-footer"><div class="spacer"></div><button class="btn-nvidia" id="dsm-submit">Add Dataset</button></div>' +
      '</div>';
    document.body.appendChild(overlay);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.getElementById('dsm-close').addEventListener('click', function () { overlay.remove(); });

    var currentSrc = 'upload';
    var renderPanel = function () {
      var p = document.getElementById('dsm-panel');
      var nameField = '<div class="form-group"><label class="form-label">Name (optional)</label><input type="text" id="dsm-name" class="form-input" placeholder="my-dataset"></div>';
      if (currentSrc === 'upload') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">File</label>' +
          '<div class="upload-drop-zone" id="dsm-drop"><strong>Drop file here</strong> or click to select<br><small style="color:var(--text-muted)">JSONL · JSON · CSV · TSV · Parquet · TXT — up to 2 GB</small></div>' +
          '<input type="file" id="dsm-file" accept=".json,.jsonl,.csv,.tsv,.parquet,.txt" style="display:none"></div>' +
          '<div id="dsm-file-name" style="font-family:var(--font-mono);font-size:11.5px;color:var(--nvidia-green);margin-top:-8px"></div>';
        var drop = document.getElementById('dsm-drop');
        var fileIn = document.getElementById('dsm-file');
        drop.addEventListener('click', function () { fileIn.click(); });
        fileIn.addEventListener('change', function () {
          if (fileIn.files[0]) document.getElementById('dsm-file-name').textContent = fileIn.files[0].name + ' · ' + self.formatBytes(fileIn.files[0].size);
        });
        drop.addEventListener('dragover', function (e) { e.preventDefault(); drop.classList.add('drag-over'); });
        drop.addEventListener('dragleave', function () { drop.classList.remove('drag-over'); });
        drop.addEventListener('drop', function (e) {
          e.preventDefault();
          drop.classList.remove('drag-over');
          if (e.dataTransfer.files[0]) {
            fileIn.files = e.dataTransfer.files;
            document.getElementById('dsm-file-name').textContent = e.dataTransfer.files[0].name + ' · ' + self.formatBytes(e.dataTransfer.files[0].size);
          }
        });
      } else if (currentSrc === 'huggingface') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">HuggingFace Repo ID</label><input type="text" id="dsm-path" class="form-input" placeholder="tatsu-lab/alpaca"></div>' +
          '<div class="form-grid">' +
            '<div class="form-group"><label class="form-label">Config (optional)</label><input type="text" id="dsm-config" class="form-input" placeholder="default"></div>' +
            '<div class="form-group"><label class="form-label">Split (optional)</label><input type="text" id="dsm-split" class="form-input" placeholder="train"></div>' +
          '</div>';
      } else if (currentSrc === 'local') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">Absolute path</label><input type="text" id="dsm-path" class="form-input" placeholder="/home/user/data/train.jsonl"></div>' +
          '<div class="form-hint">The file must already exist and be readable. We will not copy it — just reference it.</div>';
      } else if (currentSrc === 'url') {
        p.innerHTML = nameField +
          '<div class="form-group"><label class="form-label">URL</label><input type="text" id="dsm-path" class="form-input" placeholder="https://example.com/data.jsonl"></div>' +
          '<div class="form-hint">The file will be downloaded to <code>~/.ainode/datasets/</code>.</div>';
      }
    };
    renderPanel();

    overlay.querySelectorAll('.pill[data-src]').forEach(function (p) {
      p.addEventListener('click', function () {
        overlay.querySelectorAll('.pill[data-src]').forEach(function (x) { x.classList.remove('active'); });
        p.classList.add('active');
        currentSrc = p.dataset.src;
        renderPanel();
      });
    });

    document.getElementById('dsm-submit').addEventListener('click', async function () {
      var btn = this;
      btn.disabled = true;
      btn.textContent = 'Adding…';
      var name = (document.getElementById('dsm-name') || {}).value || '';
      try {
        var resp;
        if (currentSrc === 'upload') {
          var fileIn = document.getElementById('dsm-file');
          if (!fileIn.files[0]) { self.toast('Please choose a file', 'error'); btn.disabled = false; btn.textContent = 'Add Dataset'; return; }
          var fd = new FormData();
          fd.append('file', fileIn.files[0]);
          if (name) fd.append('name', name);
          resp = await fetch('/api/datasets/upload', { method: 'POST', body: fd });
        } else {
          var body = { source: currentSrc, name: name };
          var pathEl = document.getElementById('dsm-path');
          var pathVal = pathEl ? pathEl.value.trim() : '';
          if (currentSrc === 'huggingface') {
            body.repo_id = pathVal;
            body.config = (document.getElementById('dsm-config') || {}).value || null;
            body.split = (document.getElementById('dsm-split') || {}).value || null;
          } else if (currentSrc === 'url') {
            body.url = pathVal;
          } else {
            body.path = pathVal;
          }
          resp = await fetch('/api/datasets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        }
        var data = await resp.json();
        if (!resp.ok) { self.toast('Error: ' + (data.error || 'Failed'), 'error'); btn.disabled = false; btn.textContent = 'Add Dataset'; return; }
        self.toast('Dataset added', 'success');
        overlay.remove();
        self.renderTraining();
      } catch (err) {
        self.toast('Network error: ' + err.message, 'error');
        btn.disabled = false; btn.textContent = 'Add Dataset';
      }
    });
  },

  // =======================================================================
  //  GUIDE MODAL (markdown explainers)
  // =======================================================================

  showGuideModal(id) {
    var guides = this.guideContent || {};
    var g = guides[id] || { title: 'Guide', html: '<p>Guide coming soon.</p>' };
    var existing = document.querySelector('.model-detail-modal-overlay.guide-overlay');
    if (existing) existing.remove();
    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay guide-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal guide-modal">' +
        '<div class="md-header"><div class="md-header-left"><div class="md-icon">&#9733;</div><div style="font-weight:700">' + this.esc(g.title) + '</div></div>' +
          '<button class="btn-ghost btn-sm" id="guide-close">Close</button></div>' +
        '<div style="padding:22px 28px">' + g.html + '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.getElementById('guide-close').addEventListener('click', function () { overlay.remove(); });
  },

  guideContent: {
    beginners: {
      title: "Beginner's Guide to Training",
      html: '<h2>What is training?</h2><p>Training — more precisely <strong>fine-tuning</strong> — takes a base model and teaches it your specific domain. You show it hundreds or thousands of examples and it learns the patterns.</p>' +
        '<h3>What you need</h3><ul><li>A base model (start small: 3B params)</li><li>Training data (JSONL, JSON, CSV — a few hundred examples minimum)</li><li>An NVIDIA GPU (you have one!)</li></ul>' +
        '<h3>Recommended first run</h3><ol><li>Pick <code>Llama 3.2 3B Instruct</code> as base model</li><li>Upload an Alpaca-style JSONL (<code>instruction</code> / <code>output</code>)</li><li>Method: <strong>LoRA</strong></li><li>Epochs: 3, batch size: 4</li><li>Click Launch — monitor from the Runs tab</li></ol>' +
        '<h3>Why LoRA?</h3><p>Full fine-tuning updates every weight in the model — huge memory cost. LoRA trains small adapter matrices on top of the frozen base, giving 95%+ of the quality at a fraction of the memory.</p>',
    },
    distributed: {
      title: 'Distributed Training',
      html: '<h2>Multi-node DDP</h2><p>AINode uses <strong>PyTorch Distributed Data Parallel</strong> (DDP) over NCCL for multi-node training. Each node holds a copy of the model; gradients are all-reduced between nodes each step.</p>' +
        '<h3>Requirements</h3><ul><li>2+ online AINode cluster members</li><li>Fast interconnect (10G+ recommended, 100G+ ideal)</li><li>Same model cached on every node</li><li>Shared dataset path or replicated data</li></ul>' +
        '<h3>When to use it</h3><p>Your training time scales roughly linearly with nodes for data-parallel workloads. Use 2+ nodes when:</p>' +
        '<ul><li>Your dataset is large (10k+ samples)</li><li>You want to cut wall-clock time</li><li>The model fits in a single node (for DDP — otherwise consider sharding)</li></ul>',
    },
    pipeline: {
      title: 'Training Pipeline',
      html: '<h2>End-to-end pipeline</h2>' +
        '<ol><li><strong>Data prep</strong> — convert your source data to JSONL with the expected fields</li>' +
        '<li><strong>Train</strong> — pick a base model, dataset, method; launch the run</li>' +
        '<li><strong>Merge</strong> — if LoRA, merge adapter into base model for deployment</li>' +
        '<li><strong>Evaluate</strong> — test on a held-out set before deploy</li>' +
        '<li><strong>Deploy</strong> — load the merged model as a vLLM instance</li></ol>' +
        '<h3>Checkpointing</h3><p>Long runs save checkpoints every N steps to <code>~/.ainode/training/jobs/&lt;id&gt;/output/</code>. You can resume from a checkpoint if training is interrupted.</p>',
    },
    standalone: {
      title: 'Standalone Training',
      html: '<h2>Single-node fine-tuning</h2>' +
        '<p>For most users, a single DGX Spark with 128 GB unified memory is plenty. You can train up to ~8B models with full fine-tuning, or up to 70B+ with LoRA / QLoRA.</p>' +
        '<h3>Tips</h3><ul>' +
        '<li>Use <code>QLoRA</code> for huge models — 4-bit quantized weights leave room for gradients</li>' +
        '<li>Enable gradient checkpointing if you run out of memory (trades compute for memory)</li>' +
        '<li>Keep <code>max_seq_length</code> as small as your data allows — memory scales quadratically with seq length</li>' +
        '<li>Watch the loss chart — if it plateaus early, try a higher learning rate; if it diverges, lower it</li>' +
        '</ul>',
    },
  },

  async renderTrainingDetail(container) {
    var jobId = this.state.trainingDetailId;
    if (!jobId) { this.state.trainingView = 'list'; this.renderTraining(); return; }

    var jobData = await this.fetchJSON('/api/training/jobs/' + jobId);
    if (!jobData) { container.innerHTML = '<div class="training-empty"><h3>Job not found</h3></div>'; return; }

    var extra = await Promise.all([
      this.fetchJSON('/api/training/jobs/' + jobId + '/logs?tail=200'),
      this.fetchJSON('/api/training/jobs/' + jobId + '/output'),
    ]);
    var logsData = extra[0];
    var logs = logsData?.logs || [];
    var outputFiles = extra[1]?.files || [];

    // Collect loss data
    if (jobData.current_loss != null && jobData.progress > 0) {
      var lp = this.state.trainingLossData[this.state.trainingLossData.length - 1];
      if (!lp || lp.progress !== jobData.progress) {
        this.state.trainingLossData.push({ progress: jobData.progress, loss: jobData.current_loss, epoch: jobData.current_epoch });
      }
    }
    if (this.state.trainingLossData.length === 0) {
      for (var i = 0; i < logs.length; i++) {
        var marker = 'AINODE_PROGRESS:';
        var idx = logs[i].indexOf(marker);
        if (idx !== -1) {
          try {
            var p = JSON.parse(logs[i].slice(idx + marker.length));
            if (p.loss != null && p.progress != null) {
              this.state.trainingLossData.push({ progress: p.progress, loss: p.loss, epoch: p.epoch || 0 });
            }
          } catch (e) { /* skip */ }
        }
      }
    }

    var statusMap = {
      pending: { cls: 'status-pending', label: 'Pending' },
      running: { cls: 'status-running', label: 'Running' },
      completed: { cls: 'status-completed', label: 'Completed' },
      failed: { cls: 'status-failed', label: 'Failed' },
      cancelled: { cls: 'status-cancelled', label: 'Cancelled' },
    };
    var s = statusMap[jobData.status] || { cls: '', label: jobData.status };
    var cfg = jobData.config || {};
    var modelShort = cfg.base_model?.split('/').pop() || 'Unknown';
    var started = jobData.start_time ? new Date(jobData.start_time * 1000).toLocaleString() : 'Not started';
    var ended = jobData.end_time ? new Date(jobData.end_time * 1000).toLocaleString() : '--';
    var elapsed = jobData.elapsed_seconds ? this.formatUptime(Math.round(jobData.elapsed_seconds)) : '--';
    var isActive = jobData.status === 'running' || jobData.status === 'pending';
    var canMerge = jobData.status === 'completed' && (cfg.method === 'lora' || cfg.method === 'qlora');
    var canResume = jobData.status === 'failed' || jobData.status === 'cancelled';

    var html = '<div class="training-detail">' +
      '<div class="training-form-header">' +
      '<button class="btn-ghost" id="training-back-btn">&larr; Back to Jobs</button>' +
      '<div class="training-detail-title"><h2>' + this.esc(modelShort) + '</h2><span class="job-status-badge ' + s.cls + '">' + s.label + '</span></div>' +
      '</div>';

    if (jobData.status === 'running') {
      html += '<div class="training-detail-progress">' +
        '<div class="progress-header"><span>' + (jobData.progress || 0).toFixed(1) + '%</span><span>Epoch ' + (jobData.current_epoch || 0) + ' / ' + (cfg.num_epochs || '?') + '</span></div>' +
        '<div class="progress-bar progress-lg"><div class="progress-fill" style="width:' + (jobData.progress || 0) + '%"></div></div>' +
        '<div class="progress-stats">' +
        (jobData.current_loss != null ? '<span>Loss: <strong>' + jobData.current_loss.toFixed(4) + '</strong></span>' : '') +
        '<span>Elapsed: <strong>' + elapsed + '</strong></span>' +
        '</div></div>';
    }

    html += '<div class="training-detail-grid">' +
      '<div class="training-config-card">' +
      '<h3>Configuration</h3>' +
      '<table class="config-table">' +
      '<tr><td>Model</td><td class="mono">' + this.esc(cfg.base_model || '') + '</td></tr>' +
      '<tr><td>Method</td><td>' + (cfg.method || 'lora').toUpperCase() + '</td></tr>' +
      '<tr><td>Dataset</td><td class="mono">' + this.esc(cfg.dataset_path || '') + '</td></tr>' +
      '<tr><td>Epochs</td><td>' + (cfg.num_epochs || '?') + '</td></tr>' +
      '<tr><td>Batch Size</td><td>' + (cfg.batch_size || '?') + '</td></tr>' +
      '<tr><td>Learning Rate</td><td>' + (cfg.learning_rate || '?') + '</td></tr>' +
      '<tr><td>Max Seq Length</td><td>' + (cfg.max_seq_length || '?') + '</td></tr>' +
      (cfg.method === 'lora' ? '<tr><td>LoRA Rank</td><td>' + (cfg.lora_rank || '?') + '</td></tr><tr><td>LoRA Alpha</td><td>' + (cfg.lora_alpha || '?') + '</td></tr>' : '') +
      '<tr><td>Job ID</td><td class="mono">' + this.esc(jobData.job_id) + '</td></tr>' +
      '<tr><td>Started</td><td>' + started + '</td></tr>' +
      '<tr><td>Ended</td><td>' + ended + '</td></tr>' +
      '<tr><td>Duration</td><td>' + elapsed + '</td></tr>' +
      '</table>' +
      (isActive ? '<div style="margin-top:16px"><button class="btn-danger" id="training-cancel-job-btn">Cancel Job</button></div>' : '') +
      (canMerge || canResume ? '<div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap">' +
        (canMerge ? '<button class="btn-nvidia" id="training-merge-btn">Merge Adapter &rarr; Full Model</button>' : '') +
        (canResume ? '<button class="btn-ghost" id="training-resume-btn">Resume from Checkpoint</button>' : '') +
        '</div>' : '') +
      '</div>' +
      '<div class="training-right-col">' +
      (this.state.trainingLossData.length > 1 ?
        '<div class="training-chart-card"><h3>Training Loss</h3><div class="loss-chart-container"><canvas id="loss-chart" width="460" height="200"></canvas></div></div>' : '') +
      (outputFiles.length > 0 ? '<div class="training-config-card"><h3>Artifacts</h3><table class="config-table">' +
        outputFiles.map(function (f) {
          return '<tr><td><a class="chat-link" href="' + AINode.esc(f.download_url) + '" download>' + AINode.esc(f.name) + '</a></td><td class="mono">' + f.size_mb + ' MB</td></tr>';
        }).join('') +
        '</table></div>' : '') +
      '<div class="training-log-card"><h3>Logs <span class="log-line-count">' + (logsData?.total_lines || 0) + ' lines</span></h3>' +
      '<div class="training-log-viewer" id="training-log-viewer">' +
      (logs.length > 0 ? logs.map(function (l) { return '<div class="log-line">' + AINode.esc(l) + '</div>'; }).join('') : '<div class="log-empty">No logs yet</div>') +
      '</div></div>' +
      '</div></div></div>';

    container.innerHTML = html;

    var self = this;
    var bb = document.getElementById('training-back-btn');
    if (bb) bb.addEventListener('click', function () {
      self.state.trainingView = 'list';
      self.state.trainingDetailId = null;
      self.state.trainingLossData = [];
      self.renderTraining();
    });

    var cjb = document.getElementById('training-cancel-job-btn');
    if (cjb) cjb.addEventListener('click', async function () {
      if (!confirm('Cancel this training job?')) return;
      var resp = await fetch('/api/training/jobs/' + jobId, { method: 'DELETE' });
      if (resp.ok) self.renderTraining();
      else {
        var err = await resp.json().catch(function () { return {}; });
        self.toast(err.error || 'Failed to cancel job', 'error');
      }
    });

    var mb = document.getElementById('training-merge-btn');
    if (mb) mb.addEventListener('click', async function () {
      if (!confirm('Merge this adapter into a full model? This can take several minutes.')) return;
      mb.disabled = true;
      var resp = await fetch('/api/training/jobs/' + jobId + '/merge', { method: 'POST' });
      var result = await resp.json().catch(function () { return {}; });
      if (resp.ok) {
        self.toast('Merge started — output: ' + result.output_dir, 'success');
        self.state.trainingDetailId = result.merge_job_id;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      } else {
        mb.disabled = false;
        self.toast(result.error || 'Failed to start merge', 'error');
      }
    });

    var rb = document.getElementById('training-resume-btn');
    if (rb) rb.addEventListener('click', async function () {
      if (!confirm('Resume training from the latest checkpoint?')) return;
      rb.disabled = true;
      var resp = await fetch('/api/training/jobs/' + jobId + '/resume', { method: 'POST' });
      var result = await resp.json().catch(function () { return {}; });
      if (resp.ok) {
        self.toast('Resumed from checkpoint ' + result.checkpoint, 'success');
        self.state.trainingDetailId = result.resume_job_id;
        self.state.trainingLossData = [];
        self.renderTrainingSidebar();
        self.renderTraining();
      } else {
        rb.disabled = false;
        self.toast(result.error || 'Failed to resume job', 'error');
      }
    });

    var lv = document.getElementById('training-log-viewer');
    if (lv) lv.scrollTop = lv.scrollHeight;

    if (this.state.trainingLossData.length > 1) this.drawLossChart();
  },

  // ========================================================================
  //  TRAINING — LOSS CHART
  // ========================================================================

  drawLossChart() {
    var canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    var w = rect.width, h = rect.height;
    var pad = { top: 12, right: 16, bottom: 28, left: 48 };
    var plotW = w - pad.left - pad.right;
    var plotH = h - pad.top - pad.bottom;
    var data = this.state.trainingLossData;
    if (data.length < 2) return;

    var losses = data.map(function (d) { return d.loss; });
    var maxLoss = Math.max.apply(null, losses) * 1.05;
    var minLoss = Math.min.apply(null, losses) * 0.95;
    var maxProg = Math.max.apply(null, data.map(function (d) { return d.progress; }).concat([1]));

    var toX = function (prog) { return pad.left + (prog / maxProg) * plotW; };
    var toY = function (loss) { return pad.top + ((maxLoss - loss) / (maxLoss - minLoss || 1)) * plotH; };

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(45,55,72,0.5)';
    ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      var val = maxLoss - ((maxLoss - minLoss) / 4) * i;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px Inter,sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(3), pad.left - 6, y + 3);
    }

    // X-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('0%', pad.left, h - 6);
    ctx.fillText(maxProg.toFixed(0) + '%', w - pad.right, h - 6);

    // Loss line (green for command center theme)
    ctx.strokeStyle = '#76b900';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    data.forEach(function (d, i) {
      var x = toX(d.progress), y2 = toY(d.loss);
      if (i === 0) ctx.moveTo(x, y2); else ctx.lineTo(x, y2);
    });
    ctx.stroke();

    // Fill gradient
    var gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    gradient.addColorStop(0, 'rgba(118,185,0,0.15)');
    gradient.addColorStop(1, 'rgba(118,185,0,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    data.forEach(function (d, i) {
      var x = toX(d.progress), y2 = toY(d.loss);
      if (i === 0) ctx.moveTo(x, y2); else ctx.lineTo(x, y2);
    });
    ctx.lineTo(toX(data[data.length - 1].progress), h - pad.bottom);
    ctx.lineTo(toX(data[0].progress), h - pad.bottom);
    ctx.closePath();
    ctx.fill();

    // Endpoint dot
    var last = data[data.length - 1];
    ctx.fillStyle = '#76b900';
    ctx.beginPath();
    ctx.arc(toX(last.progress), toY(last.loss), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0a0e17';
    ctx.lineWidth = 2;
    ctx.stroke();
  },
});
