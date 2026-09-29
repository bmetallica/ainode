/* AINode Command Center — the profile wizard.
 *
 * wizzard.md: plan a whole profile — name, models (LLM, embedding, image),
 * which nodes each runs on, then every model's cache against the nodes it
 * shares, live — and save it, apply it, or open it again later.
 *
 * The arithmetic is the server's (POST /api/planner/household): it knows the
 * checkpoints, the measurements and the nodes. This file keeps a draft,
 * sends it on every change, and draws what comes back. The pure parts —
 * reading a saved profile back into a draft, the starting placement, the node
 * bars — are in lib.js and tested there.
 *
 * Methods of the AINode object, loaded after app.js (see views/*.js).
 */
Object.assign(AINode, {
  _pw: null,

  PW_STEPS: [
    { n: 1, label: 'Name' },
    { n: 2, label: 'Models' },
    { n: 3, label: 'Nodes' },
    { n: 4, label: 'Parameters' },
    { n: 5, label: 'Review' },
  ],

  // -- open / close -----------------------------------------------------------

  async openProfileWizard(editName) {
    var self = this;
    var catalog = await this.fetchJSON('/api/planner/household/models')
      .catch(function () { return null; });
    var list = await this.fetchJSON('/api/profiles').catch(function () { return null; });
    var draft = null;
    var wasDefault = false;
    if (editName) {
      var got = await this.fetchJSON('/api/profiles/' + encodeURIComponent(editName))
        .catch(function () { return null; });
      if (!got || !got.profile) {
        this.toast('Could not read "' + editName + '"', 'error');
        return;
      }
      draft = AINodeLib.wizardDraftFromProfile(got.profile);
      wasDefault = !!got.is_default;
      draft.makeDefault = wasDefault;
    } else {
      var kept = this._pwLoadAutosave();
      if (kept && (kept.models || []).length &&
          confirm('Continue the profile you were planning' +
                  (kept.name ? ' ("' + kept.name + '")' : '') + '?')) {
        draft = kept;
      }
    }
    draft = draft || { name: '', description: '', makeDefault: false, models: [], limits: {} };
    draft.limits = draft.limits || {};

    this._pw = {
      step: 1, draft: draft, editing: editName || null, wasDefault: wasDefault,
      catalog: (catalog && catalog.models) || [],
      existing: ((list && list.profiles) || []).map(function (p) { return p.name; }),
      plan: null, origLimits: {}, seq: 0, timer: null, running: [], busy: false,
    };
    // The nodes' limits as they are now, before the draft overrides any.
    var first = await this._pwRequestPlan({ models: [], limits: {} });
    (first && first.nodes || []).forEach(function (n) {
      self._pw.origLimits[n.node_id] = n.limit_gb || 0;
    });
    this._pw.plan = await this._pwRequestPlan(this._pwPlanBody());

    var existingOverlay = document.querySelector('.pw-overlay');
    if (existingOverlay) existingOverlay.remove();
    var overlay = document.createElement('div');
    overlay.className = 'model-detail-modal-overlay wizard-overlay pw-overlay';
    overlay.innerHTML =
      '<div class="model-detail-modal wizard-modal pw-modal">' +
        '<div class="md-header">' +
          '<div class="md-header-left"><div class="md-icon">&#9638;</div><div>' +
            '<div style="font-weight:700;font-size:15px">' +
              (editName ? 'Edit profile · ' + this.esc(editName) : 'New profile') + '</div>' +
            '<div style="font-size:11.5px;color:var(--text-muted)">Models, nodes and ' +
              'their memory, planned together</div></div></div>' +
          '<button class="btn-ghost btn-sm" id="pw-close">Close</button>' +
        '</div>' +
        '<div class="wizard-body">' +
          '<div class="wizard-steps" id="pw-steps"></div>' +
          '<div class="wizard-panel" id="pw-panel"></div>' +
        '</div>' +
        '<div class="wizard-footer">' +
          '<button class="btn-ghost" id="pw-prev">&larr; Back</button>' +
          '<span class="pw-footer-note" id="pw-note"></span>' +
          '<div class="spacer"></div>' +
          '<button class="btn-ghost" id="pw-save" style="display:none">Save</button>' +
          '<button class="btn-nvidia" id="pw-next">Next &rarr;</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    document.getElementById('pw-close').addEventListener('click', function () {
      self.closeProfileWizard();
    });
    document.getElementById('pw-prev').addEventListener('click', function () { self._pwGo(-1); });
    document.getElementById('pw-next').addEventListener('click', function () {
      if (self._pw.step === 5) self._pwSaveProfile(true);
      else self._pwGo(+1);
    });
    document.getElementById('pw-save').addEventListener('click', function () {
      self._pwSaveProfile(false);
    });
    this._pwRender();
  },

  closeProfileWizard(saved) {
    var pw = this._pw;
    if (!saved && pw && !pw.editing && (pw.draft.models || []).length) {
      // Kept, so a closed tab or a slip of the mouse loses nothing; offered
      // back the next time the wizard opens.
      this._pwAutosave();
    }
    if (saved) this._pwClearAutosave();
    if (pw && pw.timer) clearTimeout(pw.timer);
    var o = document.querySelector('.pw-overlay');
    if (o) o.remove();
    this._pw = null;
  },

  _pwAutosave() {
    try {
      localStorage.setItem('ainode.profileWizardDraft', JSON.stringify(this._pw.draft));
    } catch (e) { /* private window: nothing kept */ }
  },

  _pwLoadAutosave() {
    try { return JSON.parse(localStorage.getItem('ainode.profileWizardDraft') || 'null'); }
    catch (e) { return null; }
  },

  _pwClearAutosave() {
    try { localStorage.removeItem('ainode.profileWizardDraft'); } catch (e) { /* ignore */ }
  },

  // -- planning ---------------------------------------------------------------

  _pwPlanBody() {
    var d = this._pw.draft;
    return {
      models: (d.models || []).map(function (m) {
        return Object.assign({}, m);
      }),
      limits: d.limits || {},
    };
  },

  async _pwRequestPlan(body) {
    try {
      var resp = await fetch('/api/planner/household', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return await resp.json();
    } catch (e) {
      return null;
    }
  },

  _pwSchedulePlan() {
    var self = this;
    var pw = this._pw;
    if (!pw) return;
    if (pw.step !== 1) this._pwAutosaveIfNew();
    clearTimeout(pw.timer);
    pw.timer = setTimeout(async function () {
      if (!self._pw) return;
      var seq = ++self._pw.seq;
      var plan = await self._pwRequestPlan(self._pwPlanBody());
      // An answer to an older draft is not an answer to this one.
      if (!self._pw || seq !== self._pw.seq || !plan) return;
      self._pw.plan = plan;
      self._pwUpdateLive();
    }, 250);
  },

  _pwAutosaveIfNew() {
    if (this._pw && !this._pw.editing) this._pwAutosave();
  },

  _pwColor(id) {
    var models = (this._pw && this._pw.draft.models) || [];
    var index = models.findIndex(function (m) { return m.id === id; });
    return AINodeLib.wizardColor(index < 0 ? 0 : index);
  },

  _pwPlanned(id) {
    var plan = this._pw && this._pw.plan;
    return ((plan && plan.models) || []).find(function (m) { return m.id === id; }) || null;
  },

  _pwNodeName(id) {
    var plan = this._pw && this._pw.plan;
    var node = ((plan && plan.nodes) || []).find(function (n) { return n.node_id === id; });
    return node ? (node.name || id) : id;
  },

  // -- rendering ---------------------------------------------------------------

  _pwRender() {
    var pw = this._pw;
    if (!pw) return;
    var steps = document.getElementById('pw-steps');
    steps.innerHTML = this.PW_STEPS.map(function (d) {
      var cls = d.n === pw.step ? ' active' : (d.n < pw.step ? ' done' : '');
      return '<div class="wizard-step-nav' + cls + '"><span class="step-num">' + d.n +
        '</span>' + d.label + '</div>';
    }).join('');
    var panel = document.getElementById('pw-panel');
    panel.innerHTML = [null, this._pwStepName, this._pwStepModels, this._pwStepNodes,
                       this._pwStepParams, this._pwStepReview][pw.step].call(this);
    document.getElementById('pw-prev').style.visibility = pw.step === 1 ? 'hidden' : '';
    var next = document.getElementById('pw-next');
    next.innerHTML = pw.step === 5 ? 'Save &amp; apply' : 'Next &rarr;';
    document.getElementById('pw-save').style.display = pw.step === 5 ? '' : 'none';
    this._pwBind();
    this._pwUpdateLive();
  },

  _pwStepName() {
    var d = this._pw.draft;
    var editing = !!this._pw.editing;
    return '<h3>Name the profile</h3>' +
      '<div class="panel-sub">A profile is the set of models the cluster should serve. ' +
      'Applying it converges every node it uses: what it does not list there is stopped.</div>' +
      '<div class="pw-field"><label class="form-label">Name</label>' +
        '<input class="form-input" id="pw-name" value="' + this.esc(d.name || '') + '"' +
        (editing ? ' disabled' : '') + ' placeholder="e.g. Endausbau"></div>' +
      (editing ? '<div class="server-hint">A profile keeps its name; to rename it, ' +
        'save a copy under the new name and delete this one.</div>' : '') +
      '<div class="pw-field"><label class="form-label">Description (optional)</label>' +
        '<input class="form-input" id="pw-desc" value="' + this.esc(d.description || '') + '"' +
        ' placeholder="What it is for"></div>' +
      '<label class="config-check"><input type="checkbox" id="pw-default"' +
        (d.makeDefault ? ' checked' : '') + '> Load this profile at startup (default)</label>' +
      '<div class="pw-error" id="pw-step-error"></div>';
  },

  _pwStepModels() {
    var self = this;
    var d = this._pw.draft;
    var groups = { llm: 'Language models', embedding: 'Embedding models', image: 'Image models' };
    var html = '<h3>Choose the models</h3>' +
      '<div class="panel-sub">Only models on this node\'s disk are offered. Add a model ' +
      'twice to run it on two nodes — a replica; requests are spread over both.</div>';
    html += '<div class="pw-chosen" id="pw-chosen">' + this._pwChosenList() + '</div>';
    Object.keys(groups).forEach(function (kind) {
      var items = self._pw.catalog.filter(function (m) { return m.kind === kind; });
      if (!items.length) return;
      html += '<div class="pw-group-title">' + groups[kind] + '</div><div class="pw-catalog">';
      items.forEach(function (m) {
        var count = d.models.filter(function (x) { return x.model === m.model; }).length;
        html += '<div class="pw-catalog-row' + (m.complete ? '' : ' disabled') + '">' +
          '<span class="pw-catalog-name mono" title="' + self.esc(m.model) + '">' +
            self.esc(m.model) + '</span>' +
          '<span class="pw-catalog-size">' + (m.size_gb ? m.size_gb.toFixed(1) + ' GB' : '') +
          '</span>' +
          (m.complete
            ? '<button class="btn-ghost server-btn-sm" data-pw-add="' + self.esc(m.model) + '">' +
              (count ? 'Add replica' : 'Add') + '</button>'
            : '<span class="pw-catalog-note" title="' + self.esc(m.incomplete_reason) +
              '">incomplete</span>') +
          '</div>';
      });
      html += '</div>';
    });
    if (!this._pw.catalog.length) {
      html += '<div class="server-empty">No models on disk. Download some first.</div>';
    }
    return html + '<div class="pw-error" id="pw-step-error"></div>';
  },

  _pwChosenList() {
    var self = this;
    var models = this._pw.draft.models;
    if (!models.length) return '<div class="server-hint">Nothing chosen yet.</div>';
    return models.map(function (m) {
      return '<span class="pw-chip"><span class="pw-dot" style="background:' +
        self._pwColor(m.id) + '"></span>' + self.esc(m.model) +
        ' <span class="pw-kind">' + self.esc(m.kind) + '</span>' +
        '<button class="pw-chip-x" data-pw-remove="' + self.esc(m.id) + '" ' +
        'title="Remove">&times;</button></span>';
    }).join('');
  },

  _pwNodeCards(editable) {
    var self = this;
    var plan = this._pw.plan || {};
    var head = plan.head_node_id;
    return '<div class="pw-nodes">' + (plan.nodes || []).map(function (n, i) {
      var limit = self._pw.draft.limits[n.node_id];
      if (limit === undefined) limit = n.limit_gb || 0;
      return '<div class="pw-node' + (n.online ? '' : ' offline') + '">' +
        '<div class="pw-node-head"><strong>' + self.esc(n.name) + '</strong>' +
          (n.node_id === head ? ' <span class="pw-kind">head</span>' : '') +
          (n.online ? '' : ' <span class="pw-kind">offline</span>') + '</div>' +
        '<div class="pw-node-meta">' + n.total_gb.toFixed(0) + ' GB · idle use ' +
          n.baseline_gb.toFixed(1) + ' GB' + (n.baseline_estimated ? ' (estimate)' : '') +
        '</div>' +
        (editable
          ? '<label class="pw-limit">Limit <input class="form-input" type="number" min="0" ' +
            'step="1" data-pw-limit="' + self.esc(n.node_id) + '" value="' +
            (limit ? limit : '') + '" placeholder="none"> GB</label>'
          : (limit ? '<div class="pw-node-meta">Limit ' + limit + ' GB</div>' : '')) +
        '<div id="pw-bar-' + i + '"></div>' +
        '</div>';
    }).join('') + '</div>';
  },

  _pwStepNodes() {
    var self = this;
    var plan = this._pw.plan || {};
    var nodes = plan.nodes || [];
    var html = '<h3>Place the models</h3>' +
      '<div class="panel-sub">Tick the nodes each model runs on. A language model on ' +
      'several nodes is split across them and has to include the head; image and ' +
      'embedding models run on one node. The limit caps what this node may be filled to ' +
      '— for every launch, not only this profile.</div>';
    html += this._pwNodeCards(true);
    html += '<div class="pw-toolbar"><button class="btn-ghost server-btn-sm" ' +
      'id="pw-autoassign">Place the rest automatically</button></div>';
    html += '<table class="pw-table"><thead><tr><th>Model</th>' +
      nodes.map(function (n) { return '<th>' + self.esc(n.name) + '</th>'; }).join('') +
      '<th>Split</th></tr></thead><tbody>';
    this._pw.draft.models.forEach(function (m, i) {
      html += '<tr><td><span class="pw-dot" style="background:' + self._pwColor(m.id) +
        '"></span><span class="mono">' + self.esc(m.model) + '</span> <span class="pw-kind">' +
        self.esc(m.kind) + '</span><div class="pw-error small" id="pw-err-' + i + '"></div></td>';
      nodes.forEach(function (n) {
        var on = (m.node_ids || []).indexOf(n.node_id) !== -1;
        html += '<td class="pw-cell"><input type="checkbox" data-pw-place="' + i + '" ' +
          'data-node="' + self.esc(n.node_id) + '"' + (on ? ' checked' : '') + '></td>';
      });
      var split = '';
      if (m.kind === 'llm' && (m.node_ids || []).length > 1) {
        split = '<select class="form-input pw-small" data-pw-strategy="' + i + '">' +
          ['tensor', 'pipeline'].map(function (s) {
            return '<option value="' + s + '"' + ((m.strategy || 'tensor') === s ? ' selected' : '') +
              '>' + s + '</option>';
          }).join('') + '</select>';
      }
      html += '<td>' + (split || '<span class="pw-kind">—</span>') + '</td></tr>';
    });
    html += '</tbody></table><div class="pw-conflicts" id="pw-conflicts"></div>' +
      '<div class="pw-error" id="pw-step-error"></div>';
    return html;
  },

  _pwStepParams() {
    var self = this;
    var html = '<h3>Memory and parameters</h3>' +
      '<div class="panel-sub">Each language model\'s cache is either fixed — by context × ' +
      'sessions, or by size — or shares what is left by priority. Change one and the ' +
      'others on the same nodes move with it.</div>';
    html += this._pwNodeCards(false);
    html += '<div class="pw-conflicts" id="pw-conflicts"></div>';
    html += '<div class="pw-cards">' + this._pw.draft.models.map(function (m, i) {
      return self._pwCard(m, i);
    }).join('') + '</div>';
    return html + '<div class="pw-error" id="pw-step-error"></div>';
  },

  _pwCard(m, i) {
    var self = this;
    var where = (m.node_ids || []).map(function (id) { return self._pwNodeName(id); }).join(' + ');
    var h = '<div class="pw-card" id="pw-card-' + i + '" style="border-left-color:' +
      this._pwColor(m.id) + '">' +
      '<div class="pw-card-head"><span class="mono">' + this.esc(m.model) + '</span>' +
      ' <span class="pw-kind">' + this.esc(m.kind) + ' · ' + this.esc(where || 'not placed') +
      '</span></div>';
    if (m.kind === 'llm') {
      var mode = m.mode || 'auto';
      h += '<div class="pw-row">' +
        '<label>Cache <select class="form-input pw-small" data-pw-mode="' + i + '">' +
          [['auto', 'share what is left'], ['usage', 'context × sessions'],
           ['size', 'fixed size']].map(function (o) {
            return '<option value="' + o[0] + '"' + (mode === o[0] ? ' selected' : '') + '>' +
              o[1] + '</option>';
          }).join('') + '</select></label>' +
        '<label>Context <input class="form-input pw-small" type="number" min="4096" step="4096" ' +
          'data-pw-field="max_model_len" data-i="' + i + '" value="' + (m.max_model_len || '') +
          '" placeholder="recipe"></label>';
      if (mode === 'usage') {
        h += '<label>Sessions <input class="form-input pw-small" type="number" min="1" step="1" ' +
          'data-pw-field="sessions" data-i="' + i + '" value="' + (m.sessions || 1) + '"></label>';
      } else if (mode === 'size') {
        var planned = this._pwPlanned(m.id) || {};
        // Up to what fits: the other automatic models on its nodes shrunk to
        // their minimum — beyond that the plan would not fit at all.
        var cap = Math.max(m.cache_gb || 0, planned.cache_max_per_node_gb ||
                           planned.cache_cap_per_node_gb || 100);
        h += '<label class="pw-slider">Cache per node <input type="range" min="0" max="' +
          (Math.ceil(cap * 10) / 10) + '" step="0.1" data-pw-field="cache_gb" data-i="' + i + '" value="' +
          (m.cache_gb || 0) + '"> <span id="pw-cache-val-' + i + '">' + (m.cache_gb || 0) +
          ' GB</span></label>';
      } else {
        h += '<label>Priority <select class="form-input pw-small" data-pw-field="priority" ' +
          'data-i="' + i + '">' + [1, 2, 3, 4, 5].map(function (p) {
            return '<option value="' + p + '"' + ((m.priority || 1) === p ? ' selected' : '') +
              '>' + p + '</option>';
          }).join('') + '</select></label>';
      }
      h += '<label>KV cache <select class="form-input pw-small" data-pw-field="kv_cache_dtype" ' +
        'data-i="' + i + '">' + [['', 'default'], ['auto', 'model dtype'], ['fp8', 'fp8'],
                                 ['fp8_ds_mla', 'fp8_ds_mla']].map(function (o) {
          return '<option value="' + o[0] + '"' + ((m.kv_cache_dtype || '') === o[0] ? ' selected' : '') +
            '>' + o[1] + '</option>';
        }).join('') + '</select></label></div>';
      if ((m.extra_vllm_args || []).length) {
        h += '<div class="pw-kind">Flags: ' + this.esc(m.extra_vllm_args.join(' ')) + '</div>';
      }
    } else if (m.kind === 'image') {
      h += '<div class="pw-row"><label>Largest image <select class="form-input pw-small" ' +
        'data-pw-field="max_image_size" data-i="' + i + '">' +
        [768, 1024, 1536, 2048].map(function (s) {
          return '<option value="' + s + '"' + ((m.max_image_size || 1536) === s ? ' selected' : '') +
            '>' + s + '×' + s + '</option>';
        }).join('') + '</select></label></div>';
    }
    h += '<div class="pw-result" id="pw-res-' + i + '"></div></div>';
    return h;
  },

  _pwStepReview() {
    var self = this;
    var plan = this._pw.plan || {};
    var html = '<h3>Review</h3>' +
      '<div class="panel-sub">What will run where. Saving keeps the profile; applying it ' +
      'also converges the nodes it uses.</div>';
    html += this._pwNodeCards(false);
    html += '<table class="pw-table"><thead><tr><th>Model</th><th>Nodes</th><th>Memory</th>' +
      '<th>Context</th><th>Sessions</th><th>Cache / node</th></tr></thead><tbody>';
    (plan.models || []).forEach(function (m) {
      var where = (m.node_ids || []).map(function (id) { return self._pwNodeName(id); }).join(' + ');
      html += '<tr><td><span class="pw-dot" style="background:' + self._pwColor(m.id) +
        '"></span><span class="mono">' + self.esc(m.model) + '</span></td>' +
        '<td>' + self.esc(where) + (m.strategy && m.strategy !== 'solo' ? ' · ' + self.esc(m.strategy) : '') +
        '</td><td>' + (m.gpu_memory_utilization ? Math.round(m.gpu_memory_utilization * 100) + '%'
                       : (m.fixed_per_node_gb ? m.fixed_per_node_gb + ' GB' : '—')) + '</td>' +
        '<td>' + (m.max_model_len ? self.formatNumber(m.max_model_len) : '—') + '</td>' +
        '<td>' + (m.sessions || (m.kind === 'llm' ? 0 : '—')) + '</td>' +
        '<td>' + (m.cache_per_node_gb != null ? m.cache_per_node_gb + ' GB' : '—') + '</td></tr>';
    });
    html += '</tbody></table>';
    var stops = AINodeLib.wizardStopPreview(this._pw.running, this._pw.draft.models);
    html += '<div class="pw-group-title">Applying will stop</div>' +
      (stops.length ? '<ul class="pw-list">' + stops.map(function (r) {
        return '<li>' + self.esc(r.id) + ' on ' + self.esc(r.node_hostname || self._pwNodeName(r.node_id)) + '</li>';
      }).join('') + '</ul>' : '<div class="server-hint">Nothing — no other model runs on these nodes.</div>');
    html += '<div class="server-hint">Models that already run as planned keep running; the ' +
      'others are started one at a time per node.</div>';
    return html + '<div class="pw-conflicts" id="pw-conflicts"></div>' +
      '<div class="pw-error" id="pw-step-error"></div>';
  },

  // What changes with every plan, without redrawing the inputs.
  _pwUpdateLive() {
    var self = this;
    var pw = this._pw;
    if (!pw) return;
    var plan = pw.plan || {};
    (plan.nodes || []).forEach(function (n, i) {
      var slot = document.getElementById('pw-bar-' + i);
      if (slot) slot.innerHTML = AINodeLib.renderHouseholdBar(n, function (id) { return self._pwColor(id); });
    });
    var box = document.getElementById('pw-conflicts');
    if (box) {
      box.innerHTML = (plan.conflicts || []).map(function (c) {
        return '<div class="config-warning">⚠ ' + self.esc(c) + '</div>';
      }).join('');
    }
    pw.draft.models.forEach(function (m, i) {
      var planned = self._pwPlanned(m.id) || {};
      var slider = document.querySelector('[data-pw-field="cache_gb"][data-i="' + i + '"]');
      if (slider && planned.cache_max_per_node_gb != null && document.activeElement !== slider) {
        slider.max = String(Math.ceil(Math.max(m.cache_gb || 0, planned.cache_max_per_node_gb) * 10) / 10);
      }
      var err = document.getElementById('pw-err-' + i);
      if (err) err.textContent = (planned.errors || []).join(' ');
      var res = document.getElementById('pw-res-' + i);
      if (!res) return;
      var parts = [];
      if (planned.cache_per_node_gb != null) {
        parts.push('<strong>' + planned.cache_per_node_gb + ' GB</strong> cache per node');
        parts.push(self.formatNumber(planned.kv_tokens || 0) + ' tokens');
        parts.push((planned.sessions || 0) + ' session(s) at ' +
                   self.formatNumber(planned.max_model_len || 0));
        parts.push('memory ' + Math.round((planned.gpu_memory_utilization || 0) * 100) + '%');
      } else if (planned.fixed_per_node_gb) {
        parts.push('<strong>' + planned.fixed_per_node_gb + ' GB</strong> on its node');
      }
      if (planned.weights_source) {
        parts.push('weights ' + self.esc(planned.weights_source) +
                   (planned.kv_source ? ', cost/token ' + self.esc(planned.kv_source) : ''));
      }
      var notes = (planned.errors || []).map(function (e) {
        return '<div class="pw-error">✕ ' + self.esc(e) + '</div>';
      }).concat((planned.warnings || []).map(function (w) {
        return '<div class="pw-warn">⚠ ' + self.esc(w) + '</div>';
      })).join('');
      res.innerHTML = parts.join(' · ') + notes;
    });
    var note = document.getElementById('pw-note');
    if (note) {
      note.textContent = plan.ok ? '✓ fits' :
        ((plan.conflicts || []).length ? '⚠ does not fit' :
          (pw.draft.models.length ? '⚠ see the marked models' : ''));
    }
  },

  // -- events ------------------------------------------------------------------

  // Binds within ``root`` only (the whole panel by default): a card that is
  // redrawn is bound again on its own, so no other element gets a second
  // handler.
  _pwBind(root) {
    var self = this;
    var pw = this._pw;
    var d = pw.draft;
    var panel = root || document.getElementById('pw-panel');
    if (!panel) return;

    var on = function (selector, event, fn) {
      panel.querySelectorAll(selector).forEach(function (el) { el.addEventListener(event, fn); });
    };

    on('#pw-name', 'input', function (e) { d.name = e.target.value; });
    on('#pw-desc', 'input', function (e) { d.description = e.target.value; });
    on('#pw-default', 'change', function (e) { d.makeDefault = e.target.checked; });

    on('[data-pw-add]', 'click', function (e) {
      var model = e.currentTarget.getAttribute('data-pw-add');
      var entry = self._pw.catalog.find(function (m) { return m.model === model; }) || {};
      var item = { id: 'm' + Date.now().toString(36) + Math.floor(Math.random() * 1000),
                   model: model, kind: entry.kind || 'llm', node_ids: [] };
      if (item.kind === 'llm') { item.mode = 'auto'; item.priority = 1; }
      if (item.kind === 'image') item.max_image_size = 1536;
      d.models.push(item);
      self._pwRender();
      self._pwSchedulePlan();
    });
    on('[data-pw-remove]', 'click', function (e) {
      var id = e.currentTarget.getAttribute('data-pw-remove');
      d.models = d.models.filter(function (m) { return m.id !== id; });
      self._pwRender();
      self._pwSchedulePlan();
    });

    on('[data-pw-limit]', 'input', function (e) {
      var id = e.target.getAttribute('data-pw-limit');
      var value = parseFloat(e.target.value);
      d.limits[id] = isNaN(value) ? 0 : value;
      self._pwSchedulePlan();
    });
    on('[data-pw-place]', 'change', function (e) {
      var m = d.models[parseInt(e.target.getAttribute('data-pw-place'), 10)];
      var node = e.target.getAttribute('data-node');
      var ids = (m.node_ids || []).filter(function (x) { return x !== node; });
      if (e.target.checked) {
        // Image and embedding models run on one node: ticking another moves it.
        ids = (m.kind === 'llm') ? ids.concat([node]) : [node];
      }
      m.node_ids = ids;
      self._pwRender();
      self._pwSchedulePlan();
    });
    on('[data-pw-strategy]', 'change', function (e) {
      d.models[parseInt(e.target.getAttribute('data-pw-strategy'), 10)].strategy = e.target.value;
      self._pwSchedulePlan();
    });
    on('#pw-autoassign', 'click', function () {
      var sizes = {};
      self._pw.catalog.forEach(function (m) { sizes[m.model] = m.size_gb || 0; });
      d.models = AINodeLib.wizardAutoAssign(d.models, sizes, (pw.plan || {}).nodes || [],
                                            (pw.plan || {}).head_node_id);
      self._pwRender();
      self._pwSchedulePlan();
    });

    on('[data-pw-mode]', 'change', function (e) {
      var i = parseInt(e.target.getAttribute('data-pw-mode'), 10);
      var m = d.models[i];
      var planned = self._pwPlanned(m.id) || {};
      m.mode = e.target.value;
      // Switching mode keeps what the model has now: pinning an automatic
      // share starts from the share it had, not from zero.
      if (m.mode === 'size') m.cache_gb = planned.cache_per_node_gb || m.cache_gb || 0;
      if (m.mode === 'usage') m.sessions = Math.max(1, planned.sessions || m.sessions || 1);
      if (m.mode === 'usage' && !m.max_model_len && planned.max_model_len) {
        m.max_model_len = planned.max_model_len;
      }
      var card = document.getElementById('pw-card-' + i);
      if (card) {
        card.outerHTML = self._pwCard(m, i);
        self._pwBind(document.getElementById('pw-card-' + i));
      }
      self._pwSchedulePlan();
    });
    on('[data-pw-field]', 'input', function (e) {
      var i = parseInt(e.target.getAttribute('data-i'), 10);
      var field = e.target.getAttribute('data-pw-field');
      var m = d.models[i];
      var raw = e.target.value;
      if (field === 'kv_cache_dtype') m[field] = raw;
      else if (raw === '') delete m[field];
      else m[field] = parseFloat(raw);
      if (field === 'cache_gb') {
        var label = document.getElementById('pw-cache-val-' + i);
        if (label) label.textContent = raw + ' GB';
      }
      self._pwSchedulePlan();
    });
  },

  // -- navigation --------------------------------------------------------------

  _pwStepProblem(step) {
    var pw = this._pw;
    var d = pw.draft;
    var plan = pw.plan || {};
    if (step === 1) {
      var problem = AINodeLib.wizardNameProblem(d.name);
      if (problem) return problem;
      if (!pw.editing && pw.existing.indexOf(String(d.name).trim()) !== -1) {
        return 'A profile named "' + d.name + '" exists. Choose another name, or edit that one.';
      }
    }
    if (step === 2 && !d.models.length) return 'Choose at least one model.';
    if (step === 3) {
      var unplaced = d.models.filter(function (m) { return !(m.node_ids || []).length; });
      if (unplaced.length) return 'Place every model: ' + unplaced.map(function (m) { return m.model; }).join(', ');
      var broken = (plan.models || []).filter(function (m) {
        return (m.errors || []).some(function (e) { return e.indexOf('The cache holds') !== 0; });
      });
      if (broken.length) return 'Fix the marked models first.';
    }
    if (step === 4 && !plan.ok) {
      return (plan.conflicts || []).length ? 'The plan does not fit — see the warnings above.'
                                           : 'Fix the marked models first.';
    }
    return '';
  },

  async _pwGo(delta) {
    var pw = this._pw;
    if (!pw) return;
    if (delta > 0) {
      // The answer to the latest change, not the one before it.
      clearTimeout(pw.timer);
      pw.plan = await this._pwRequestPlan(this._pwPlanBody()) || pw.plan;
      var problem = this._pwStepProblem(pw.step);
      if (problem) {
        var slot = document.getElementById('pw-step-error');
        if (slot) slot.textContent = problem;
        this._pwUpdateLive();
        return;
      }
    }
    pw.step = Math.max(1, Math.min(5, pw.step + delta));
    if (pw.step === 5) {
      var status = await this.fetchJSON('/api/server/status').catch(function () { return null; });
      pw.running = (status && status.loaded_models) || [];
    }
    this._pwRender();
  },

  // -- saving --------------------------------------------------------------------

  async _pwSaveProfile(apply) {
    var self = this;
    var pw = this._pw;
    if (!pw || pw.busy) return;
    clearTimeout(pw.timer);
    pw.plan = await this._pwRequestPlan(this._pwPlanBody()) || pw.plan;
    var plan = pw.plan || {};
    if (!plan.ok) {
      var slot = document.getElementById('pw-step-error');
      if (slot) slot.textContent = 'The plan does not fit any more — go back and adjust it.';
      this._pwUpdateLive();
      return;
    }
    var d = pw.draft;
    var name = String(d.name || '').trim();
    pw.busy = true;
    try {
      // Node limits that changed. They apply to the node, not to the profile —
      // every launch there holds to them.
      var limitIds = Object.keys(d.limits || {});
      for (var k = 0; k < limitIds.length; k++) {
        var id = limitIds[k];
        var value = d.limits[id] || 0;
        if (Math.abs(value - (pw.origLimits[id] || 0)) < 0.05) continue;
        var lr = await fetch('/api/nodes/' + encodeURIComponent(id) + '/memory-limit', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gb: value }),
        });
        if (!lr.ok) {
          var le = await lr.json().catch(function () { return {}; });
          this.toast('Limit for ' + this._pwNodeName(id) + ' not set: ' + (le.error || lr.status), 'error');
        }
      }

      var body = {
        name: name, description: d.description || '', entries: plan.entries,
        wizard: { version: 1, models: d.models, limits: d.limits },
      };
      var method = pw.editing ? 'PUT' : 'POST';
      var url = pw.editing ? '/api/profiles/' + encodeURIComponent(name) : '/api/profiles';
      var resp = await fetch(url, {
        method: method, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (resp.status === 409 && confirm('"' + name + '" exists. Replace it?')) {
        resp = await fetch('/api/profiles/' + encodeURIComponent(name), {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
      }
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok) {
        this.toast(data.error || 'Could not save the profile', 'error');
        return;
      }
      if (d.makeDefault || pw.wasDefault) {
        await fetch('/api/profiles/' + encodeURIComponent(name) + '/default', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(d.makeDefault ? {} : { default: false }),
        });
      }
      this.toast('Saved "' + name + '"', 'success');
      this.closeProfileWizard(true);
      if (typeof this.renderProfiles === 'function') await this.renderProfiles();
      if (apply) this.applyProfile(name, null, true);
    } finally {
      if (this._pw) this._pw.busy = false;
    }
  },
});
