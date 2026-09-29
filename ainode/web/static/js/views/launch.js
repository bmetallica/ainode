/* AINode Command Center — the launch sidebar: the form, placement, image generation and the planner.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  RIGHT PANEL — LAUNCH INSTANCE
  // ========================================================================

  _launchModelsPopulated: false,

  bindLaunchForm() {
    var self = this;

    // Sharding pills
    var pillGroup = document.getElementById('sharding-pills');
    if (pillGroup) {
      pillGroup.querySelectorAll('.pill').forEach(function (pill) {
        pill.addEventListener('click', function () {
          pillGroup.querySelectorAll('.pill').forEach(function (p) { p.classList.remove('active'); });
          pill.classList.add('active');
        });
      });
    }

    // Node selector dots — populated dynamically based on cluster size
    var nodeSelector = document.getElementById('node-selector');
    var launchHint = document.getElementById('launch-hint');
    var self = this;

    // One toggle per real node. The head (this node) is pinned-selected; the
    // others default OFF (so a fresh launch is solo TP=1 — the common case).
    // Click a node to add it; TP size = number of selected nodes. The launch
    // POSTs the selected node_ids (resolved to fabric IPs server-side).
    this._renderNodeDots = function () {
      if (!nodeSelector) return;
      var headId = self.state.status && self.state.status.node_id;
      var nodes = (self.state.nodes || []).slice();
      if (!nodes.length && headId) nodes = [{ node_id: headId, node_name: 'this node' }];
      var key = nodes.map(function (n) { return n.node_id; }).join(',');
      if (nodeSelector.dataset.nodekey === key) return; // node set unchanged
      nodeSelector.dataset.nodekey = key;
      nodeSelector.innerHTML = nodes.map(function (n) {
        var isHead = headId && n.node_id === headId;
        var label = (n.node_name || n.node_id || '').replace(/-DGX|-GX10/i, '');
        return '<button class="node-dot' + (isHead ? ' active head' : '') + '"'
          + ' data-node-id="' + self.esc(n.node_id) + '"'
          + (isHead ? ' data-head="1"' : '')
          + ' title="' + self.esc(n.node_name || n.node_id) + (isHead ? ' (head)' : '') + '">'
          + self.esc(label) + (isHead ? ' ★' : '') + '</button>';
      }).join('');
      nodeSelector.querySelectorAll('.node-dot').forEach(function (dot) {
        dot.addEventListener('click', function () {
          // A single active node = solo load on THAT node (head or peer); 2+ =
          // distributed. Keep at least one selected so a load always has a target.
          if (dot.classList.contains('active') &&
              nodeSelector.querySelectorAll('.node-dot.active').length <= 1) return;
          dot.classList.toggle('active');
          // The user hand-picked nodes for this pending launch — auto-recommend
          // (fired on model-select onchange) must NOT clobber that choice.
          self._launchNodesUserPicked = true;
          updateLaunchHint();
          self.repinIfPinned();
          self.schedulePlan();
        });
      });
      updateLaunchHint();
    };
    this._renderNodeDots();

    // Pre-select head + (tp-1) peers — used when a model's proven TP is chosen.
    this._selectNodes = function (tp) {
      var sel = document.getElementById('node-selector'); if (!sel) return;
      var peersWanted = Math.max(0, tp - 1), peerN = 0;
      sel.querySelectorAll('.node-dot').forEach(function (d) {
        if (d.dataset.head) { d.classList.add('active'); return; }
        if (peerN < peersWanted) { d.classList.add('active'); peerN++; }
        else { d.classList.remove('active'); }
      });
      updateLaunchHint();
    };

    // Activate exactly this set of node ids.
    //
    // The head used to be forced on regardless, which made two things
    // inexpressible: a solo load on another node (the single-node path sends
    // the one selected dot, so head + n3 is a two-node distributed launch),
    // and a pin or a plan that deliberately leaves this node out. An empty
    // set still falls back to the head — a launch needs a target.
    this._selectNodeIds = function (ids) {
      var sel = document.getElementById('node-selector'); if (!sel) return;
      var want = {}; (ids || []).forEach(function (id) { want[id] = true; });
      var any = false;
      sel.querySelectorAll('.node-dot').forEach(function (d) {
        var on = !!want[d.dataset.nodeId];
        d.classList.toggle('active', on);
        any = any || on;
      });
      if (!any) {
        var head = sel.querySelector('.node-dot[data-head]') ||
                   sel.querySelector('.node-dot');
        if (head) head.classList.add('active');
      }
    };

    // Fit-aware hint, recomputed on every refresh + node/strategy change so it
    // survives polling. Reads the selected model's size and each node's free mem.
    function updateLaunchHint() {
      if (!launchHint) return;
      // The planner's answer wins when it describes this exact selection. It
      // read the checkpoint's own config.json and the nodes' free memory; the
      // estimate below reads a size and a percentage. The estimate stays as
      // the fallback for a model that is not on disk yet, where the planner
      // correctly refuses to guess.
      if (self.renderPlanHint()) return;
      var active = nodeSelector ? nodeSelector.querySelectorAll('.node-dot.active') : [];
      var n = active.length || 1;
      var names = Array.prototype.map.call(active, function (d) { return d.textContent.replace('★', '').trim(); }).join(' + ');
      var strategyPill = document.querySelector('#sharding-pills .pill.active');
      var strat = strategyPill ? strategyPill.dataset.value : 'tensor';
      var stratLabel = strat.charAt(0).toUpperCase() + strat.slice(1);
      var axis = { tensor: 'TP', pipeline: 'PP', data: 'DP' }[strat] || 'TP';
      var split = axis + '=' + n;
      launchHint.className = 'launch-hint';

      // An image model is not splittable at all — one process, one node.
      var msel0 = document.getElementById('launch-model');
      var opt0 = msel0 && msel0.selectedIndex >= 0
        ? msel0.options[msel0.selectedIndex] : null;
      if (opt0 && opt0.getAttribute('data-modality') === 'image') {
        launchHint.className = 'launch-hint';
        launchHint.textContent = '✓ Image model on ' + (names || 'this node') +
          ' — one process on one node. Any node can serve it; it is not split.';
        return;
      }

      // Refuse an impossible split up front rather than letting the user press
      // LAUNCH and read a 422. Same rule as the server (parallelism.py).
      if (!self.strategyAllowed(strat, n)) {
        launchHint.className = 'launch-hint warn';
        launchHint.textContent = '⚠ Tensor needs 2, 4 or 8 nodes — no model splits '
          + 'attention heads ' + n + ' ways. Use Pipeline across all ' + n
          + ' nodes, Data for ' + n + ' replicas, or select fewer nodes.';
        return;
      }

      var msel = document.getElementById('launch-model');
      var opt = msel && msel.selectedIndex >= 0 ? msel.options[msel.selectedIndex] : null;
      var size = opt ? parseFloat(opt.getAttribute('data-size-gb') || '0') : 0;
      var minMem = opt ? parseFloat(opt.getAttribute('data-min-mem') || '0') : 0;
      var req = (minMem && minMem > size) ? minMem : size * 1.2;
      if (!opt || !opt.value || !req) {  // no model picked yet → plain text
        launchHint.textContent = n <= 1 ? 'Solo — runs on this node only (TP=1).'
          : stratLabel + ' · ' + split + ' across ' + names + '.';
        return;
      }

      var byId = {};
      (self.state.nodes || []).forEach(function (nd) { byId[nd.node_id] = nd; });
      var freeOf = function (nd) { return nd ? Math.max(0, (nd.gpu_memory_gb || 0) * (1 - (nd.gpu_memory_used_pct || 0) / 100)) : 0; };
      // Tensor and pipeline both divide the weights across the nodes. Data
      // parallelism does not — every node holds a FULL replica, so the
      // per-node requirement is the whole model however many nodes are used.
      var perShard = (strat === 'data') ? req : req / n;
      var frees = Array.prototype.map.call(active, function (d) { return freeOf(byId[d.dataset.nodeId]); });
      var minFree = frees.length ? Math.min.apply(null, frees) : 0;
      if (minFree < perShard) {
        launchHint.className = 'launch-hint warn';
        launchHint.textContent = '⚠ Needs ~' + Math.round(perShard) + ' GB/node but a selected node has only ~' +
          Math.round(minFree) + ' GB free' +
          (strat === 'data' ? ' — Data keeps a full copy per node; try Pipeline.'
                            : ' — unload a model to free space.');
      } else if (n <= 1) {
        launchHint.textContent = '✓ Runs solo on ' + (names || 'this node') + ' (TP=1) — ~' + Math.round(req) + ' GB.';
      } else {
        launchHint.textContent = '✓ ' + stratLabel + ' · ' + split + ' across ' + names +
          ' (~' + Math.round(perShard) + ' GB/node).';
      }
    }
    // (dot click handlers bound inside _renderNodeDots)
    // Update hint when strategy pill changes too
    document.querySelectorAll('#sharding-pills .pill').forEach(function (pill) {
      pill.addEventListener('click', updateLaunchHint);
      // A pinned model keeps its pin current: changing the axis while the box
      // is ticked rewrites the placement rather than leaving the tick
      // describing a set of nodes that is no longer on screen.
      pill.addEventListener('click', function () { self.repinIfPinned(); });
      pill.addEventListener('click', function () { self.schedulePlan(); });
    });

    if (launchHint) {
      launchHint.addEventListener('click', function (e) {
        if (e.target && e.target.id === 'plan-apply') self.applyPlan();
      });
    }

    // The advanced fields had no listener at all, so the plan above them never
    // moved when they changed: typing a context length left the hint
    // describing a launch nobody had asked for. Each field now re-plans, and
    // records itself as the one the operator drove — see launchPlanKey, which
    // omits the OTHER constraint so the planner derives it rather than
    // repeating what is already in the box.
    [['launch-max-len', 'len'], ['launch-max-seqs', 'seqs'],
     ['launch-kv-dtype', 'kv'], ['launch-gmu', 'gmu']].forEach(
      function (pair) {
        var field = document.getElementById(pair[0]);
        if (!field) return;
        var event = field.tagName === 'SELECT' ? 'change' : 'input';
        field.addEventListener(event, function () {
          if (self._applyingPlan) return;   // our own write-back, not an edit
          self.state.launchLastEdited = pair[1];
          // For THIS model on THESE nodes. A context typed for one model is
          // no statement about the next one — see launchDrove.
          self.state.launchLastEditedFor = self.launchEditScope();
          self.schedulePlan();
        });
      });

    var pinBox = document.getElementById('launch-pin');
    if (pinBox) {
      pinBox.addEventListener('change', function () {
        self.savePlacement(pinBox.checked);
      });
    }
    this.loadPlacements();
    // And on every refresh (cluster state may change)
    this._launchHintUpdater = updateLaunchHint;
    updateLaunchHint();

    // Launch button
    var launchBtn = document.getElementById('launch-btn');
    if (launchBtn) {
      launchBtn.addEventListener('click', function () { self.launchInstance(); });
    }
  },

  // Auto-pick sharding + nodes for the selected model, aware of free memory on
  // each node (a node already serving a model has ~85% reserved by vLLM, so it
  // reads as nearly full). m: {proven_tp, size_gb, min_mem}.
  recommendLaunch(m) {
    var nodes = (this.state.nodes || []).filter(function (n) { return n.status !== 'offline'; });
    var size = m.size_gb || 0;
    if (!nodes.length || (!size && !m.min_mem)) return;
    var freeOf = function (n) { return Math.max(0, (n.gpu_memory_gb || 0) * (1 - (n.gpu_memory_used_pct || 0) / 100)); };
    var headId = this.state.status && this.state.status.node_id;
    var head = nodes.find(function (n) { return n.node_id === headId; }) || nodes[0];
    var peers = nodes.filter(function (n) { return n !== head; }).sort(function (a, b) {
      var am = a.model ? 1 : 0, bm = b.model ? 1 : 0;
      if (am !== bm) return am - bm;          // prefer idle nodes over busy ones
      return freeOf(b) - freeOf(a);           // then the most free
    });
    var N = nodes.length;
    var req = (m.min_mem && m.min_mem > size) ? m.min_mem : size * 1.2;

    // Tensor is the proven strategy on this hardware, so try it first:
    // smallest TP (>= proven_tp) where head + (tp-1) peers each hold req/tp.
    var rec = null;
    [1, 2, 4, 8].forEach(function (tp) {
      if (rec || tp > N) return;
      if (m.proven_tp && tp < m.proven_tp) return;
      var perShard = req / tp;
      if (freeOf(head) < perShard) return;
      var okPeers = peers.filter(function (n) { return freeOf(n) >= perShard; });
      if (okPeers.length >= tp - 1) rec = { tp: tp, peers: okPeers.slice(0, tp - 1) };
    });

    // No tensor split fits. Before giving up and landing the whole model on the
    // head, try pipeline across every node: it works at any node count (the
    // only option on a 3-node mesh) and divides the weights the same way.
    var strategy = 'tensor';
    if (!rec && N > 1) {
      var perNode = req / N;
      if (freeOf(head) >= perNode &&
          peers.filter(function (n) { return freeOf(n) >= perNode; }).length >= N - 1) {
        strategy = 'pipeline';
        rec = { tp: N, peers: peers.slice(0, N - 1) };
      }
    }

    var pills = document.getElementById('sharding-pills');
    if (pills) pills.querySelectorAll('.pill').forEach(function (p) {
      p.classList.toggle('active', p.dataset.value === strategy);
    });

    // Set node selection; the fit-aware updater renders the hint (and survives polling).
    // But if the user already hand-picked nodes for this launch, DON'T clobber their
    // dots — a "pick node, then model" flow was landing the load on the head. We still
    // recompute the hint text below so the recommendation is reflected.
    var ids = rec ? [head.node_id].concat(rec.peers.map(function (n) { return n.node_id; })) : [head.node_id];
    if (!this._launchNodesUserPicked && this._selectNodeIds) this._selectNodeIds(ids);
    if (this._launchHintUpdater) this._launchHintUpdater();
  },

  // ========================================================================
  //  PERSISTENT PLACEMENT — "this model runs here"
  // ========================================================================
  // The server keeps the same answer in ~/.ainode/placement.json and reads it
  // on every launch that does not name its own nodes, so a pin also holds for
  // launches this form never sees: a relaunch after a failed load, a restart,
  // a profile entry without an explicit node list.

  loadPlacements() {
    var self = this;
    return this.fetchJSON('/api/placement').then(function (data) {
      var map = {};
      ((data && data.placements) || []).forEach(function (p) {
        if (p && p.model) map[p.model] = p;
      });
      self.state.placements = map;
      self.syncPinUI();
      return map;
    });
  },

  // Fetch-once was the bug: the Models page holds two lists that were each
  // fetched a single time per page load, so a download that finished, a model
  // that was deleted or one that was launched afterwards never appeared. A
  // TTL costs one request a minute; invalidate() makes an action's own result
  // visible immediately.
  _stale(key, ttlMs) {
    this._fetchedAt = this._fetchedAt || {};
    var last = this._fetchedAt[key] || 0;
    if (Date.now() - last < (ttlMs || 60000)) return false;
    this._fetchedAt[key] = Date.now();
    return true;
  },

  invalidate() {
    this._fetchedAt = {};
    this._modelLists = null;
  },

  placementFor(model) {
    return (this.state.placements || {})[model] || null;
  },

  // What the form currently says: the dots that are on, and the active axis.
  launchSelection() {
    var sel = document.getElementById('node-selector');
    var ids = sel ? Array.prototype.map.call(
      sel.querySelectorAll('.node-dot.active'),
      function (d) { return d.dataset.nodeId; }) : [];
    var pill = document.querySelector('#sharding-pills .pill.active');
    return { node_ids: ids, strategy: pill ? pill.dataset.value : 'tensor' };
  },

  // Put the form where the placement says. Returns false when there is none,
  // so the caller can fall back to the recommendation.
  applyPlacement(model) {
    var p = this.placementFor(model);
    if (!p || !(p.node_ids || []).length) return false;
    if (this._selectNodeIds) this._selectNodeIds(p.node_ids);
    if (p.strategy) {
      var pills = document.getElementById('sharding-pills');
      if (pills) pills.querySelectorAll('.pill').forEach(function (pill) {
        pill.classList.toggle('active', pill.dataset.value === p.strategy);
      });
    }
    if (this._launchHintUpdater) this._launchHintUpdater();
    return true;
  },

  syncPinUI() {
    var box = document.getElementById('launch-pin');
    if (!box) return;
    var select = document.getElementById('launch-model');
    var model = select ? select.value : '';
    var p = model ? this.placementFor(model) : null;
    box.disabled = !model;
    box.checked = !!p;
    var note = document.getElementById('launch-pin-note');
    if (note) {
      note.textContent = p ? '\u2713 ' + (p.node_ids || []).length + ' node'
        + ((p.node_ids || []).length === 1 ? '' : 's')
        + (p.strategy ? ' \u00b7 ' + p.strategy : '') : '';
    }
  },

  repinIfPinned() {
    var box = document.getElementById('launch-pin');
    if (box && box.checked && !box.disabled) this.savePlacement(true);
  },

  async savePlacement(pinned) {
    var select = document.getElementById('launch-model');
    var model = select ? select.value : '';
    if (!model) return;
    try {
      var resp;
      if (pinned) {
        var sel = this.launchSelection();
        resp = await fetch('/api/placement', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: model, node_ids: sel.node_ids, strategy: sel.strategy,
          }),
        });
      } else {
        resp = await fetch('/api/placement/' + encodeURIComponent(model),
                           { method: 'DELETE' });
      }
      var data = await resp.json();
      if (data && data.error) this.toast(data.error, 'error');
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
    await this.loadPlacements();
  },

  // ========================================================================
  //  IMAGE GENERATION
  // ========================================================================
  // A second kind of model, served by a second engine, reached through the
  // same proxy. Everything else about it — loading, placement, the memory
  // guard, profiles — is the machinery that was already there.

  // The launch form grows three fields when the selected model is an image
  // one, and hides the KV-cache ones that mean nothing for it.
  toggleImageFields(model) {
    var select = document.getElementById('launch-model');
    var option = select && select.selectedIndex >= 0
      ? select.options[select.selectedIndex] : null;
    var isImage = !!option &&
      option.getAttribute('data-modality') === 'image';
    var block = document.getElementById('launch-image-fields');
    if (block) block.style.display = isImage ? '' : 'none';
    // One node, whichever one. The image engine runs a single process and
    // has no axis to split along, so leaving several dots lit would let
    // someone build a launch the server has to refuse.
    if (isImage && this._selectNodeIds) {
      var selector = document.getElementById('node-selector');
      var lit = selector
        ? Array.prototype.map.call(selector.querySelectorAll('.node-dot.active'),
                                   function (d) { return d.dataset.nodeId; })
        : [];
      if (lit.length > 1) this._selectNodeIds([lit[0]]);
    }
    // A diffusion run has no KV cache and no context length. Leaving those
    // fields on screen would invite someone to set them and wonder why
    // nothing changed.
    ['launch-max-seqs', 'launch-max-len', 'launch-kv-dtype'].forEach(
      function (id) {
        var field = document.getElementById(id);
        if (field && field.parentElement) {
          var label = field.previousElementSibling;
          field.style.display = isImage ? 'none' : '';
          if (label && label.classList.contains('form-label')) {
            label.style.display = isImage ? 'none' : '';
          }
        }
      });
    return isImage;
  },

  imageOverrides() {
    var out = {};
    var size = document.getElementById('launch-max-image');
    if (size && size.value) out.max_image_size = parseInt(size.value, 10);
    var steps = document.getElementById('launch-image-steps');
    if (steps && steps.value) out.image_steps = parseInt(steps.value, 10);
    var dims = document.getElementById('launch-image-size');
    if (dims && dims.value.trim()) out.image_size = dims.value.trim();
    return out;
  },

  // Which loaded models can make pictures. Read from the same instance list
  // the cards use, so the two cannot disagree about what is running.
  imageInstances() {
    var out = [];
    (this.state.nodes || []).forEach(function (node) {
      (node.instances || []).forEach(function (inst) {
        if (inst.kind === 'image' && inst.model) {
          out.push({ model: inst.model, node: node.node_name || node.node_id,
                     ready: inst.status === 'serving' });
        }
      });
    });
    return out;
  },

  async renderImages() {
    var mount = document.getElementById('images-content');
    if (!mount) return;
    var self = this;
    var models = this.imageInstances();

    if (!models.length) {
      mount.innerHTML = '<div class="config-empty">No image model is loaded. ' +
        'Load one from the launch panel — the catalog has Qwen-Image 2.1 in ' +
        'FP8 and full precision.</div>';
      return;
    }
    if (this._imagesDrawnFor === models.map(function (m) { return m.model; }).join()) {
      return;   // the 5s poll must not wipe a half-typed prompt
    }
    this._imagesDrawnFor = models.map(function (m) { return m.model; }).join();

    var html = '<div class="config-card"><div class="config-form-grid single">';
    html += '<div><label class="config-field-label">Model</label>' +
      '<select class="form-select" id="image-model">' +
      models.map(function (m) {
        return '<option value="' + self.esc(m.model) + '">' +
          self.esc(m.model) + ' · ' + self.esc(m.node) +
          (m.ready ? '' : ' (still loading)') + '</option>';
      }).join('') + '</select></div>';
    html += '<div><label class="config-field-label">Prompt</label>' +
      '<textarea class="form-input" id="image-prompt" rows="3" ' +
      'placeholder="a red cube on a white table, studio lighting"></textarea></div>';
    html += '</div><div class="config-form-grid">';
    html += '<div><label class="config-field-label">Size</label>' +
      '<input class="form-input" id="image-size" value="1024x1024"></div>';
    html += '<div><label class="config-field-label">Steps</label>' +
      '<input class="form-input" id="image-steps" type="number" min="1" ' +
      'max="500" value="20"></div>';
    html += '<div><label class="config-field-label">Seed ' +
      '<span style="color:var(--text-muted);font-weight:400">(empty = random)</span></label>' +
      '<input class="form-input" id="image-seed" type="number"></div>';
    html += '<div><label class="config-field-label">Negative prompt</label>' +
      '<input class="form-input" id="image-negative"></div>';
    html += '</div><div class="config-actions">' +
      '<button class="config-btn" id="image-go">Generate</button>' +
      '<span class="config-field-hint" id="image-status"></span>' +
      '</div></div>';
    html += '<div id="image-gallery" class="image-gallery"></div>';
    mount.innerHTML = html;

    document.getElementById('image-go').addEventListener('click', function () {
      self.generateImage();
    });
  },

  async generateImage() {
    var button = document.getElementById('image-go');
    var status = document.getElementById('image-status');
    var prompt = (document.getElementById('image-prompt') || {}).value || '';
    if (!prompt.trim()) { this.toast('A prompt, first', 'warning'); return; }

    var body = {
      model: document.getElementById('image-model').value,
      prompt: prompt.trim(),
      size: (document.getElementById('image-size') || {}).value || '1024x1024',
      steps: parseInt((document.getElementById('image-steps') || {}).value, 10) || 20,
      response_format: 'b64_json',
    };
    var seed = (document.getElementById('image-seed') || {}).value;
    if (seed) body.seed = parseInt(seed, 10);
    var negative = (document.getElementById('image-negative') || {}).value;
    if (negative && negative.trim()) body.negative_prompt = negative.trim();

    button.disabled = true;
    var started = Date.now();
    // A picture takes tens of seconds and there is no progress to report from
    // the engine, so the elapsed time is the honest thing to show.
    var ticking = setInterval(function () {
      if (status) {
        status.textContent = 'generating… ' +
          Math.round((Date.now() - started) / 1000) + 's';
      }
    }, 1000);
    try {
      var resp = await fetch('/v1/images/generations', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok || data.error) {
        var message = (data.error && (data.error.message || data.error)) ||
          ('the engine answered ' + resp.status);
        if (status) status.textContent = '';
        this.toast(String(message), 'error');
      } else {
        this.showImages(data, body, Math.round((Date.now() - started) / 1000));
        if (status) {
          status.textContent = 'done in ' +
            ((data.ainode && data.ainode.seconds) ||
             Math.round((Date.now() - started) / 1000)) + 's';
        }
      }
    } catch (err) {
      if (status) status.textContent = '';
      this.toast('Error: ' + err.message, 'error');
    }
    clearInterval(ticking);
    button.disabled = false;
  },

  showImages(data, request, seconds) {
    var gallery = document.getElementById('image-gallery');
    if (!gallery) return;
    var self = this;
    var caption = this.esc(request.prompt) + ' · ' +
      this.esc(request.size) + ' · ' + request.steps + ' steps' +
      (request.seed !== undefined ? ' · seed ' + request.seed : '') +
      ' · ' + seconds + 's';
    var cards = (data.data || []).map(function (entry) {
      // Data URI rather than a blob: the picture is already base64 in the
      // response, and a blob would need revoking to avoid leaking it.
      return '<figure class="image-card">' +
        '<img src="data:image/png;base64,' + entry.b64_json + '" alt="">' +
        '<figcaption>' + caption + '</figcaption></figure>';
    }).join('');
    gallery.insertAdjacentHTML('afterbegin', cards);
    // Newest first, and bounded: a session of forty pictures at a megabyte
    // each is a tab that runs out of memory.
    var figures = gallery.querySelectorAll('.image-card');
    for (var i = 12; i < figures.length; i++) figures[i].remove();
    void self;
  },

  // ========================================================================
  //  LAUNCH PLANNER
  // ========================================================================
  // The server reads the checkpoint's own config.json, the nodes' free memory
  // and the catalog recipe, and works out whether it fits, on how many nodes,
  // along which axis, what context that leaves and how many people can use it
  // at once. The form fills itself in from that instead of from a size
  // estimate and a percentage.

  launchPlanKey() {
    var select = document.getElementById('launch-model');
    var model = select ? select.value : '';
    var sel = this.launchSelection();
    var len = document.getElementById('launch-max-len');
    var seqs = document.getElementById('launch-max-seqs');
    // The KV dtype belongs in the key too: it halves or doubles the cost per
    // token, so a plan computed without it describes a different launch.
    var kv = document.getElementById('launch-kv-dtype');
    return {
      model: model,
      nodes: sel.node_ids,
      strategy: sel.strategy,
      max_model_len: (len && len.value) || '',
      concurrency: (seqs && seqs.value) || '',
      kv_cache_dtype: (kv && kv.value) || '',
      key: [model, sel.node_ids.join(','), sel.strategy,
            (len && len.value) || '', (seqs && seqs.value) || '',
            (kv && kv.value) || ''].join('|'),
    };
  },

  // Which field the operator drove — but only while the model and the nodes
  // are still the ones it was driven for. It used to outlive a model change,
  // so the next model was planned with one of the pair withheld although
  // nobody had touched either field for it.
  launchEditScope(want) {
    return AINodeLib.launchEditScope(want || this.launchPlanKey());
  },

  launchDrove(want) {
    if (this.state.launchLastEditedFor !== this.launchEditScope(want)) {
      this.state.launchLastEdited = null;
      this.state.launchLastEditedFor = null;
    }
    return this.state.launchLastEdited;
  },

  // Debounced: clicking three node dots in a row should ask once, not three
  // times, and the answer takes a directory walk over the weights.
  schedulePlan(delay) {
    var self = this;
    clearTimeout(this._planTimer);
    this._planTimer = setTimeout(function () { self.fetchPlan(); },
                                 delay === undefined ? 350 : delay);
  },

  // Put the derived half of the pair back in its box, so the two fields read
  // as what they are: one number the operator chose and one the cache allows.
  // Guarded by _applyingPlan, or the write would count as an edit and the two
  // fields would chase each other.
  reflectPlanIntoFields(plan) {
    // Nothing driven yet: leave be. See AINodeLib.planFieldUpdates.
    var updates = AINodeLib.planFieldUpdates(plan, this.launchDrove());
    var len = document.getElementById('launch-max-len');
    var seqs = document.getElementById('launch-max-seqs');
    this._applyingPlan = true;
    try {
      if (len && updates.max_model_len) len.value = updates.max_model_len;
      if (seqs && updates.max_num_seqs) seqs.value = updates.max_num_seqs;
      // The memory fraction is an input to both, never derived from them —
      // writing it back would fight the operator for the one field that is
      // purely theirs.
    } finally {
      this._applyingPlan = false;
    }
  },

  async fetchPlan() {
    var want = this.launchPlanKey();
    if (!want.model) { this.state.launchPlan = null; return; }
    if (this.state.launchPlan && this.state.launchPlan.key === want.key) return;
    // Context and concurrency multiply into one cache. Sending BOTH pins both
    // and the planner has nothing left to say, so the fields could never move
    // in relation to each other. Whichever the operator last touched is the
    // constraint; the other is the answer. The forecast bar is not misled by
    // the omission: the planner forecasts a derived concurrency at the value
    // it derives, which is what lands in the field and in the launch.
    var params = new URLSearchParams(
      AINodeLib.planQuery(want, this.launchDrove(want)));
    var data = await this.fetchJSON('/api/planner?' + params.toString());
    if (!data || data.error) { this.state.launchPlan = null; return; }
    data.key = want.key;
    this.state.launchPlan = data;
    this.reflectPlanIntoFields(data);
    if (this._launchHintUpdater) this._launchHintUpdater();
  },

  // How much of the node this will occupy, and how much of that it will use.
  //
  //     was mir noch fehlt ist beim modelladen über die seitenleiste, ist eine
  //     prognose wie viel vram das modell belegen wird, welche sich abhängig zu
  //     den einstellungen live aktualisiert
  //
  // Two numbers, because one would be the misleading half either way. The
  // engine TAKES gpu_memory_utilization x the node's total and fills it with
  // cache blocks whether the configured context needs them or not — that is
  // what free(1) shows and what the memory guard watches. The launch USES the
  // weights, the engine and cache for context x concurrency. On unified memory
  // the gap is real memory held and not used.
  // F3: a copied client config is a snapshot. Reloading a model with another
  // window makes it silently wrong — 608,512 in the config against 131,072 in
  // the engine, which looked like an agent that keeps stopping. The dashboard
  // remembers what it handed out (this browser only) and says when what is
  // running no longer matches it.
  rememberOpencodeConfig(data) {
    if (!data || !data.fingerprint) return;
    var models = AINodeLib.opencodeModels(data);
    try {
      localStorage.setItem('ainode.opencodeCopied', JSON.stringify(
        { fingerprint: data.fingerprint, at: Date.now(), models: models }));
    } catch (e) { /* private window: nothing to compare against later */ }
    this._opencodeDrift = { checkedAt: Date.now(), html: '' };
    var slot = document.getElementById('opencode-stale');
    if (slot) slot.innerHTML = '';
  },

  // What changed between the config that was copied and the one that would
  // be generated now — see AINodeLib.opencodeDrift.
  opencodeDrift(saved, current) {
    return AINodeLib.opencodeDrift(saved, current);
  },

  async checkOpencodeDrift() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem('ainode.opencodeCopied') || 'null'); }
    catch (e) { saved = null; }
    if (!saved || !saved.fingerprint) return;
    var state = this._opencodeDrift || {};
    // The Server view redraws every few seconds; the question needs asking
    // about once a minute, and it reads the whole cluster.
    if (state.checkedAt && Date.now() - state.checkedAt < 60000) return;
    this._opencodeDrift = { checkedAt: Date.now(), html: state.html || '' };
    var base = location.protocol + '//' + location.host;
    var current = await this.fetchJSON(
      '/api/clients/opencode?base_url=' + encodeURIComponent(base)).catch(function () { return null; });
    if (!current || !current.fingerprint) return;
    var html = '';
    if (current.fingerprint !== saved.fingerprint) {
      var changes = this.opencodeDrift(saved, current);
      html = '<div class="config-warning" style="margin-top:10px">⚠ The OpenCode ' +
        'config you copied on ' + this.esc(new Date(saved.at).toLocaleString()) +
        ' no longer matches what is running' +
        (changes.length ? ': ' + changes.map(this.esc, this).join('; ') : '') +
        '. Generate it again and replace the file, or sessions will be cut off ' +
        'or refused mid-answer.</div>';
    }
    this._opencodeDrift.html = html;
    var slot = document.getElementById('opencode-stale');
    if (slot) slot.innerHTML = html;
  },

  renderOccupancy(plan) {
    var seqs = document.getElementById('launch-max-seqs');
    return AINodeLib.renderOccupancy(plan, seqs ? seqs.value : '');
  },

  // Returns true when it has drawn the hint itself.
  renderPlanHint() {
    var hint = document.getElementById('launch-hint');
    var plan = this.state.launchPlan;
    if (!hint || !plan) return false;
    if (plan.key !== this.launchPlanKey().key) return false;

    if (!plan.fits) {
      // The USE THIS PLAN button is not drawn below, and its absence is the
      // first thing anyone notices — "ich kann nichtmehr use this plan
      // auswählen". Say why it is gone rather than leaving the reader to
      // infer it from a warning that reads as advice.
      hint.className = 'launch-hint warn';
      hint.innerHTML = '⚠ ' + this.esc(plan.blocker || 'This will not fit.') +
        (plan.warnings || []).map(function (w) {
          return '<div class="plan-warn">⚠ ' + this.esc(w) + '</div>';
        }, this).join('') +
        '<div class="plan-notes">There is no plan to apply, so the button is ' +
        'not offered. Free memory on a node, pick different nodes, or lower ' +
        'the context — the planner re-checks as you change the form. ' +
        '<strong>Launch anyway</strong> under Advanced overrides it.</div>';
      return true;
    }
    var axis = plan.strategy === 'solo' ? 'Solo'
      : (plan.strategy === 'pipeline' ? 'Pipeline · PP=' + plan.pipeline_parallel_size
                                      : 'Tensor · TP=' + plan.tensor_parallel_size);
    var names = (plan.node_ids || []).map(function (id) {
      var node = (plan.nodes || []).find(function (n) { return n.node_id === id; });
      return (node && node.name) || id;
    }).join(' + ');
    // Where the two deciding numbers came from. A plan built on the engine's
    // own figures and one built on arithmetic about a checkpoint look the
    // same otherwise, and the second has been nineteen percent out.
    var source = function (value, what) {
      var label = { measured: 'measured', calibrated: 'MoE-calibrated' }[value] ||
        'estimated';
      return ' <span class="plan-source plan-source-' + (value || 'estimated') +
        '" title="' + what + ': ' + label + '">' + label + '</span>';
    };
    var line = '✓ ' + axis + ' on ' + this.esc(names) +
      ' — ' + Math.round(plan.weights_per_node_gb) + ' GB/node' +
      source(plan.weights_source, 'weights') + ', ' +
      Math.round(plan.kv_gb) + ' GB cache' +
      source(plan.kv_source, 'cost per token');
    if (plan.max_model_len) {
      line += ' = ' + plan.kv_tokens.toLocaleString() + ' tokens, ' +
        plan.concurrent_requests + ' concurrent at ' +
        plan.max_model_len.toLocaleString();
    }
    var forecast = this.renderOccupancy(plan);
    var warn = (plan.warnings || []).map(function (w) {
      return '<div class="plan-warn">⚠ ' + this.esc(w) + '</div>';
    }, this).join('');
    // What it actually cost the last time, beside what the plan expects.
    // Not instead of: a measurement at one context length says nothing about
    // another, and the difference between the two is the interesting part.
    var m = plan.measured;
    var measured = '';
    if (m && m.memory_gb) {
      measured = '<div class="plan-measured">▣ Measured here: ' +
        m.memory_gb + ' GB' +
        (m.load_seconds ? ', ' + this.formatSeconds(m.load_seconds) + ' to load' : '') +
        (m.max_model_len ? ' at ' + m.max_model_len.toLocaleString() + ' context' : '') +
        (m.tokens_per_second ? ', ' + m.tokens_per_second + ' tok/s' : '') +
        (m.seconds_per_image ? ', ' + m.seconds_per_image + ' s/image' : '') +
        ' · ' + m.launches + ' launch' + (m.launches === 1 ? '' : 'es') +
        (m.failures ? ', ' + m.failures + ' failed' : '') +
        (m.vs_plan_gb ? ' (' + (m.vs_plan_gb > 0 ? '+' : '') + m.vs_plan_gb +
         ' GB vs the plan)' : '') +
        (Object.keys(m.memory_by_node || {}).length > 1 ? ' · per node: ' +
         Object.keys(m.memory_by_node).map(function (id) {
           var node = (plan.nodes || []).find(function (n) { return n.node_id === id; });
           return this.esc((node && node.name) || id) + ' ' + m.memory_by_node[id];
         }, this).join(', ') : '') +
        (m.overhead_gb != null ? ' · engine beyond weights and cache: ' +
         m.overhead_gb + ' GB (planned ' + plan.overhead_per_node_gb + ')' : '') +
        (m.other_build ? ' · <span class="plan-source-estimated">measured on vLLM ' +
         this.esc(m.engine_version || '?') + ', this node now runs ' +
         this.esc(m.other_build) + '</span>' : '') +
        '</div>';
    }
    hint.className = 'launch-hint' + ((plan.warnings || []).length ? ' warn' : '');
    hint.innerHTML = line + forecast + measured + warn +
      '<div class="plan-notes">' +
      (plan.notes || []).map(function (n) {
        return '<div>' + this.esc(n) + '</div>';
      }, this).join('') + '</div>' +
      '<div class="assist-row"><button class="btn-ghost server-btn-sm" ' +
      'id="plan-apply">USE THIS PLAN</button>' +
      '<span class="assist-hint">sets the nodes, the axis, the memory ' +
      'fraction and the context length</span></div>';
    return true;
  },

  applyPlan() {
    var plan = this.state.launchPlan;
    if (!plan || !plan.fits) return;
    if (this._selectNodeIds) this._selectNodeIds(plan.node_ids || []);
    this._launchNodesUserPicked = true;
    var axis = plan.strategy === 'pipeline' ? 'pipeline'
      : (plan.pipeline_parallel_size > 1 ? 'pipeline' : 'tensor');
    var pills = document.getElementById('sharding-pills');
    if (pills) pills.querySelectorAll('.pill').forEach(function (pill) {
      pill.classList.toggle('active', pill.dataset.value === axis);
    });
    var gmu = document.getElementById('launch-gmu');
    if (gmu && plan.gpu_memory_utilization) {
      gmu.value = plan.gpu_memory_utilization;
    }
    var len = document.getElementById('launch-max-len');
    if (len && plan.max_model_len) len.value = plan.max_model_len;
    // Not cosmetic: vLLM builds its CUDA graph capture list from this, and
    // capturing those graphs is paid in full at every launch. Measured here:
    // 56 sizes took 93 seconds. A cache that backs ten requests has no use
    // for fifty captured sizes.
    var seqs = document.getElementById('launch-max-seqs');
    if (seqs && plan.max_num_seqs) seqs.value = plan.max_num_seqs;
    this.repinIfPinned();
    this.toast('Plan applied — review the advanced fields before launching',
               'info');
    this.schedulePlan(0);
  },

  // The two model lists, fetched at most once a minute and shared by
  // everything that needs them. invalidate() makes a download or a delete
  // visible immediately.
  ensureModelLists() {
    var self = this;
    if (!this._stale('modelLists') && this._modelLists) {
      return Promise.resolve(this._modelLists);
    }
    return Promise.all([
      this.fetchJSON('/api/models'),
      this.fetchJSON('/api/cluster/models'),
    ]).then(function (results) {
      // Keep the last good answer: a poll that fails should not empty the
      // launch form's model list.
      self._modelLists = {
        catalog: results[0] || (self._modelLists || {}).catalog,
        disk: results[1] || (self._modelLists || {}).disk,
      };
      return self._modelLists;
    }).catch(function () {
      return self._modelLists || { catalog: null, disk: null };
    });
  },

  populateLaunchModels() {
    var select = document.getElementById('launch-model');
    if (!select) return;
    var self = this;
    var s = this.state.status;
    var loaded = (s && s.models_loaded) || [];
    var onDisk = this.state.downloadedModels || {};

    // This runs on every poll — every five seconds, for as long as a browser
    // tab is open. It used to fetch the catalog AND the downloaded list each
    // time, and both of those walk model directories on the server: two full
    // disk scans every five seconds per open tab, for a list that changes
    // when someone downloads something. Fetched on a TTL now and rendered
    // from what is already in hand the rest of the time.
    this.ensureModelLists().then(function (lists) {
      var data = lists.catalog;
      var dlData = lists.disk;
      if (!data) return;

      // Build set of repos known to be on disk
      var diskSet = Object.assign({}, onDisk);
      ((dlData && dlData.models) || []).forEach(function (m) {
        var repo = m.hf_repo || m.id;
        if (repo) diskSet[repo] = true;
      });

      var all = data.models || [];
      // Only show models that are on disk or currently loaded.
      var ready = all.filter(function (m) {
        var repo = m.hf_repo || m.id;
        if (m.downloaded || diskSet[repo]) return true;
        if (loaded.indexOf(repo) !== -1) return true;
        return false;
      });

      // Also add disk models not in catalog
      ((dlData && dlData.models) || []).forEach(function (m) {
        var repo = m.hf_repo || m.id;
        if (!ready.some(function (r) { return (r.hf_repo || r.id) === repo; })) {
          ready.push(m);
        }
      });

      var cv = select.value;
      if (ready.length === 0) {
        select.innerHTML = '<option value="">-- No models downloaded --</option>';
        return;
      }
      select.innerHTML = '<option value="">-- SELECT MODEL --</option>' +
        ready.map(function (m) {
          var repo = m.hf_repo || m.id;
          var label = m.name || repo;
          var sizeNote = m.size_gb ? ' (' + Math.round(m.size_gb) + ' GB)' : '';
          var isLoaded = ((self.state.status && self.state.status.models_loaded) || []).indexOf(repo) !== -1;
          var onDiskNow = isLoaded || !!(self.state.downloadedModels && self.state.downloadedModels[repo]) || !!diskSet[repo];
          var glyph = isLoaded ? '● ' : (onDiskNow ? '○ ' : '');
          var verifiedMark = m.verified ? ' ✓' : '';
          var pt = m.proven_tp || 0;
          return '<option value="' + self.esc(repo) + '" data-proven-tp="' + pt + '"'
            + ' data-size-gb="' + (m.size_gb || 0) + '" data-min-mem="' + (m.min_memory_gb || 0) + '"'
            + ' data-modality="' + self.esc(m.modality || 'text') + '">'
            + (m.modality === 'image' ? '\u25a3 ' : glyph)
            + self.esc(label) + sizeNote + verifiedMark + '</option>';
        }).join('');
      if (cv) select.value = cv;
      // A model can already be selected when the list is drawn — after a
      // poll, or when the page is reopened. Without this the image fields
      // stay hidden for it until someone picks it again.
      self.toggleImageFields(select.value);
      // Picking a model auto-recommends sharding + nodes from its size and the
      // free memory on each node.
      select.onchange = function () {
        var opt = select.options[select.selectedIndex];
        if (!opt || !opt.value) { self.syncPinUI(); return; }
        // A pin is a decision already made about this model. It outranks the
        // free-memory recommendation, which would otherwise move a pinned
        // model onto whichever nodes happen to be idle right now.
        self.toggleImageFields(opt.value);
        if (!self.applyPlacement(opt.value)) {
          self.recommendLaunch({
            proven_tp: parseInt(opt.getAttribute('data-proven-tp') || '0', 10),
            size_gb: parseFloat(opt.getAttribute('data-size-gb') || '0'),
            min_mem: parseFloat(opt.getAttribute('data-min-mem') || '0'),
          });
        }
        self.syncPinUI();
        self.schedulePlan();
      };
    });
  },

  // GPUs an instance occupies, across whichever axes it uses. Missing sizes
  // read as 1, so an instance advertised by an older head still measures right.
  instanceWorldSize(di) {
    if (!di) return 0;
    var tp = di.tensor_parallel_size || 1;
    var pp = di.pipeline_parallel_size || 1;
    var dp = di.data_parallel_size || 1;
    return tp * pp * dp;
  },

  // Which axes a given node count can actually use. Tensor parallelism splits
  // attention heads, and head counts are powers of two, so TP=3 has no models
  // behind it — the server refuses it; the UI should not offer it either.
  // Mirrors ainode/engine/parallelism.py.
  strategyAllowed(strategy, nodeCount) {
    return AINodeLib.strategyAllowed(strategy, nodeCount);
  },

  async launchInstance() {
    var select = document.getElementById('launch-model');
    var model = select ? select.value : '';
    if (!model) { this.toast('Select a model first', 'error'); return; }

    var pillGroup = document.getElementById('sharding-pills');
    var strategy = 'tensor';
    if (pillGroup) {
      var activePill = pillGroup.querySelector('.pill.active');
      if (activePill) strategy = activePill.dataset.value;
    }

    var nodeSelector = document.getElementById('node-selector');
    var nodeIds = [];
    if (nodeSelector) {
      nodeSelector.querySelectorAll('.node-dot.active').forEach(function (d) {
        if (d.dataset.nodeId) nodeIds.push(d.dataset.nodeId);
      });
    }

    var gmuInput = document.getElementById('launch-gmu');
    var gmu = gmuInput && gmuInput.value !== '' ? parseFloat(gmuInput.value) : null;
    if (gmu != null && isNaN(gmu)) gmu = null;

    // Advanced: concurrency and context. --max-num-seqs has no config field of
    // its own, so it rides along in extra_vllm_args with anything the operator
    // typed. Empty inputs are omitted entirely rather than sent as 0/"".
    var advanced = {};
    var numField = function (id) {
      var el = document.getElementById(id);
      if (!el || el.value === '') return null;
      var n = parseInt(el.value, 10);
      return isNaN(n) ? null : n;
    };
    var maxLen = numField('launch-max-len');
    if (maxLen != null) advanced.max_model_len = maxLen;

    // Sent as ONE command-line string, not a pre-split array. Splitting on
    // whitespace here could not express a quoted argument, so
    //     --compilation-config '{"mode":0}'
    // arrived at vLLM as the two words `--compilation-config` and
    // `'{"mode":0}'`, quotes included, and failed to parse as JSON. The server
    // shell-splits the string properly (shlex), which is what the API has
    // always accepted.
    var extraArgs = '';
    var maxSeqs = numField('launch-max-seqs');
    if (maxSeqs != null) extraArgs += '--max-num-seqs ' + maxSeqs + ' ';
    var freeForm = document.getElementById('launch-extra-args');
    if (freeForm && freeForm.value.trim()) extraArgs += freeForm.value.trim();
    if (extraArgs.trim()) advanced.extra_vllm_args = extraArgs.trim();

    // Deliberate, and never remembered: force is per launch. A checkbox that
    // stayed ticked would turn every later launch into an unchecked one.
    var forceBox = document.getElementById('launch-force');
    if (forceBox && forceBox.checked) advanced.force = true;

    // Text and select fields: an empty one is omitted so the catalog recipe's value
    // survives. Sending "" would override a proven setting with nothing.
    var textField = function (id) {
      var el = document.getElementById(id);
      return el && el.value.trim() ? el.value.trim() : null;
    };
    var kvDtype = textField('launch-kv-dtype');
    if (kvDtype) advanced.kv_cache_dtype = kvDtype;
    var quant = textField('launch-quantization');
    if (quant) advanced.quantization = quant;
    var img = textField('launch-engine-image');
    if (img) advanced.engine_image = img;
    var served = textField('launch-served-name');
    if (served) {
      advanced.served_model_name = served.split(',')
        .map(function (n) { return n.trim(); })
        .filter(function (n) { return n; });
    }
    var trc = document.getElementById('launch-trust-remote-code');
    if (trc && trc.checked) advanced.trust_remote_code = true;
    // Empty means "Automatic": the server derives the parser from the model
    // family. Sent only when the operator picked something, so the default
    // stays server-side and one place decides it.
    var toolCalling = textField('launch-tool-calling');
    if (toolCalling) advanced.tool_calling = toolCalling;
    // Engine environment: NAME=value per line. Some engine features have no
    // command-line flag at all — the experimental B12X stack is selected
    // purely by environment — so without this field those models can only be
    // launched through the API, which is not what "configure it in the UI"
    // was supposed to mean.
    var envText = document.getElementById('launch-extra-env');
    if (envText && envText.value.trim()) {
      var env = {};
      envText.value.split(/[\r\n]+/).forEach(function (line) {
        var trimmed = line.trim();
        if (!trimmed || trimmed.charAt(0) === '#') return;
        var eq = trimmed.indexOf('=');
        if (eq <= 0) return;
        env[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
      });
      if (Object.keys(env).length) advanced.extra_env = env;
    }

    // An image model takes different settings. Only for one: the fields are
    // hidden for a text model but keep whatever was last typed into them, and
    // sending max_image_size with an LLM would persist a meaningless value
    // onto its config.
    if (this.toggleImageFields && this.toggleImageFields(model)) {
      Object.assign(advanced, this.imageOverrides());
    }

    var launchBtn = document.getElementById('launch-btn');
    if (launchBtn) { launchBtn.disabled = true; launchBtn.textContent = 'LAUNCHING...'; }

    try {
      var endpoint, body;
      if (nodeIds.length > 1) {
        // Distributed launch on the chosen nodes (head = this node + the rest).
        endpoint = '/api/sharding/launch';
        body = { model: model, strategy: strategy, node_ids: nodeIds };
        if (gmu != null) body.gpu_memory_utilization = gmu;
        Object.assign(body, advanced);
      } else {
        // Single node launch — route to the CHOSEN node via the cluster load
        // route (node_id == this node dispatches locally), instead of always
        // hitting the head's local engine regardless of the picked node.
        endpoint = '/api/cluster/load';
        var target = nodeIds[0] || (this.state.status && this.state.status.node_id);
        body = { model: model, node_id: target };
        if (gmu != null) body.gpu_memory_utilization = gmu;
        Object.assign(body, advanced);
      }
      var resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json();
      if (data.error) {
        // A refusal that rests on the guard's memory of an earlier kill can
        // be dropped — what made that launch impossible is usually fixed by
        // something the record cannot see. Offer it here rather than leaving
        // the endpoint buried in a paragraph of the message.
        if (data.clearable && await this.offerToClearTheRecord(data)) {
          if (launchBtn) { launchBtn.disabled = false; launchBtn.textContent = 'LAUNCH'; }
          return this.launchModel();
        }
        // A refusal a one-key repair lifts: the checkpoint's config states
        // its quantization algorithm without naming the method vLLM selects
        // on, so the engine would load it as if it were not quantized.
        if (data.repairable && await this.offerToRepairTheConfig(data)) {
          if (launchBtn) { launchBtn.disabled = false; launchBtn.textContent = 'LAUNCH'; }
          return this.launchModel();
        }
        this.toast(data.error, 'error');
      } else {
        // A planning note means the split is not the one that was asked for.
        // It is the single most useful thing to know about the launch that is
        // now starting, so it goes in front of the operator, not in a log.
        if (data.note) this.toast(data.note, 'info');
        this.toast('Launched: ' + model, 'success');
        var usedForce = document.getElementById('launch-force');
        if (usedForce) usedForce.checked = false;
        // A launch can pull the weights in from a peer, so what is on disk
        // here may have changed too.
        this.invalidate();
        // Launch submitted — the hand-picked-nodes intent is consumed, so the next
        // model pick auto-recommends again.
        this._launchNodesUserPicked = false;
        this.refresh();
      }
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }

    if (launchBtn) { launchBtn.disabled = false; launchBtn.textContent = 'LAUNCH'; }
  },

  // True when the record was cleared and the launch is worth retrying.
  async offerToClearTheRecord(data) {
    if (!confirm(data.error + '\n\n' +
                 'Clear that record and try this launch again?\n\n' +
                 'The measurements for ' + data.clearable + ' are kept — only ' +
                 'the guard\'s memory of the kills is dropped.')) return false;
    try {
      var resp = await fetch('/api/measurements/forget-stops', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: data.clearable }),
      });
      var out = await resp.json().catch(function () { return {}; });
      if (!resp.ok || out.error) {
        this.toast(out.error || 'Could not clear the record', 'error');
        return false;
      }
      this.toast('Cleared the guard record for ' + data.clearable, 'info');
      return true;
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
      return false;
    }
  },

  // True when the config was repaired and the launch is worth retrying.
  async offerToRepairTheConfig(data) {
    if (!confirm(data.error + '\n\n' +
                 'Write that key into the checkpoint\'s config.json and try ' +
                 'again?\n\nOne key is added; the original config is kept ' +
                 'beside it as config.json.ainode-backup.')) return false;
    try {
      var resp = await fetch('/api/models/repair-quantization', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: data.repairable }),
      });
      var out = await resp.json().catch(function () { return {}; });
      if (!resp.ok || out.error) {
        this.toast(out.error || 'Could not repair the config', 'error');
        return false;
      }
      this.toast(out.changed ? ('Repaired ' + data.repairable)
                             : (out.detail || 'Nothing to repair'),
                 out.changed ? 'success' : 'info');
      return !!out.changed;
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
      return false;
    }
  },
});
