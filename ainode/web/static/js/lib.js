/*
 * AINode — the dashboard's pure functions.
 *
 * Everything here takes values and returns values: no document, no fetch, no
 * state. That is the point of the file (W1 in upgrade-fixes.md). These used
 * to live inside app.js, where the only way to test them was to read app.js as
 * text and look for a string — tests that broke on every rewording and still
 * let real bugs through. Here they run under `node --test tests/js`.
 *
 * Loaded before app.js as a classic script (window.AINodeLib), and as a
 * CommonJS module by the tests. No build step, in keeping with the rest.
 */
(function (root, factory) {
  var lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AINodeLib = lib;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // HTML-escape for text AND attribute values. The DOM version this replaces
  // (textContent → innerHTML) left quotes alone, and it was used inside
  // attributes: data-copy="<a JSON config>" ended at the config's first ".
  function esc(str) {
    return String(str == null ? '' : str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatSeconds(seconds) {
    seconds = Math.round(seconds || 0);
    if (seconds < 60) return seconds + 's';
    return Math.floor(seconds / 60) + 'm' + (seconds % 60 ? (seconds % 60) + 's' : '');
  }

  function formatBytes(bytes) {
    if (!bytes || bytes < 0) return '0 B';
    if (bytes < 1024) return bytes.toFixed(0) + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    if (bytes < 1024 * 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    return (bytes / (1024 * 1024 * 1024 * 1024)).toFixed(2) + ' TB';
  }

  // Tensor parallelism splits attention heads, which come in powers of two.
  function strategyAllowed(strategy, nodeCount) {
    if (nodeCount <= 1) return true;
    if (strategy === 'tensor') return [1, 2, 4, 8].indexOf(nodeCount) !== -1;
    return true;  // pipeline and data work at any node count
  }

  function clearedNote(out) {
    if (!out || !out.cleared_partials) return '';
    var gb = (out.reclaimed_bytes || 0) / 1e9;
    return ' (cleared ' + out.cleared_partials + ' stale staging file(s)' +
           (gb >= 0.1 ? ', ' + gb.toFixed(1) + ' GB reclaimed' : '') + ')';
  }

  function resumeCheckNote(checked) {
    if (!checked) return '';
    if (checked.note) return ' — ' + checked.note;
    var parts = [];
    if ((checked.removed || []).length) {
      parts.push('removed ' + checked.removed.length + ' unusable file(s)' +
        (checked.freed_bytes ? ', ' + formatBytes(checked.freed_bytes) : ''));
    }
    if ((checked.present || []).length) {
      parts.push(checked.present.length + ' file(s) verified');
    }
    if ((checked.will_fetch || []).length) {
      parts.push(checked.will_fetch.length + ' to fetch');
    }
    return parts.length ? ' — ' + parts.join(', ') : '';
  }

  // Which field the operator drove, scoped to the model and nodes it was
  // driven for (B6): a context typed for one model says nothing about the next.
  function launchEditScope(want) {
    want = want || {};
    return (want.model || '') + '|' + (want.nodes || []).join(',');
  }

  // The planner query for a launch form. Context and concurrency multiply
  // into one cache: whichever the operator last touched is the constraint and
  // is sent; the other is left out so the planner derives it. The KV dtype is
  // neither — it is always sent and never derived.
  function planQuery(want, drove) {
    var params = { model: want.model };
    if ((want.nodes || []).length) params.nodes = want.nodes.join(',');
    if (want.strategy) params.strategy = want.strategy;
    if (want.max_model_len && drove !== 'seqs') params.max_model_len = String(want.max_model_len);
    if (want.concurrency && drove !== 'len') params.concurrency = String(want.concurrency);
    if (want.kv_cache_dtype) params.kv_cache_dtype = want.kv_cache_dtype;
    return params;
  }

  // What a plan writes back into the launch form: the derived half of the
  // pair, never the half the operator drove, and never the memory fraction.
  function planFieldUpdates(plan, drove) {
    var out = {};
    if (!plan || plan.fits === false || !drove) return out;
    if (drove !== 'len' && plan.max_model_len) out.max_model_len = plan.max_model_len;
    if (drove !== 'seqs' && plan.max_num_seqs) out.max_num_seqs = plan.max_num_seqs;
    return out;
  }

  // The occupancy forecast (#216): what the engine takes against what the
  // launch uses, per node. ``seqs`` is the concurrency the form holds.
  function renderOccupancy(plan, seqs) {
    if (!plan || !plan.reserved_per_node_gb || !plan.node_total_gb) return '';
    var total = plan.node_total_gb;
    var reserved = plan.reserved_per_node_gb;
    var needed = plan.needed_per_node_gb;
    var share = Math.round((reserved / total) * 100);
    var idle = reserved - needed;
    var bar = function (value, cls) {
      return '<span class="occupancy-seg ' + cls + '" style="width:' +
        Math.max(0, Math.min(100, (value / total) * 100)).toFixed(1) + '%"></span>';
    };
    var html = '<div class="plan-occupancy">' +
      '<div class="occupancy-bar">' +
        bar(plan.weights_per_node_gb, 'weights') +
        bar(plan.overhead_per_node_gb, 'engine') +
        bar(plan.cache_used_per_node_gb, 'cache') +
        bar(Math.max(0, idle), 'idle') +
      '</div>' +
      '<div class="occupancy-text">▤ Occupies <strong>' +
      reserved.toFixed(1) + ' GB</strong> of ' + Math.round(total) +
      ' per node (' + share + '% — the memory fraction), of which <strong>' +
      needed.toFixed(1) + ' GB</strong> is used: ' +
      plan.weights_per_node_gb.toFixed(1) + ' weights + ' +
      plan.overhead_per_node_gb.toFixed(1) + ' engine + ' +
      plan.cache_used_per_node_gb.toFixed(1) + ' cache for ' +
      (plan.max_model_len || 0).toLocaleString() + ' x ' +
      esc(seqs == null ? '' : seqs) + ' requests.';
    if (idle > 2) {
      html += ' <span class="occupancy-idle">' + idle.toFixed(1) +
        ' GB of the pool is reserved and will not be used at this context and ' +
        'concurrency — lower the memory fraction to leave it on the node, or ' +
        'raise the concurrency to spend it.</span>';
    }
    return html + '</div></div>';
  }

  // F3: what changed between the OpenCode config that was copied and the one
  // that would be generated now.
  function opencodeModels(current) {
    var now = {};
    var list = (((current || {}).config || {}).provider || {}).vllm || {};
    Object.keys(list.models || {}).forEach(function (id) {
      var m = list.models[id];
      now[id] = { context: (m.limit || {}).context, reasoning: !!m.reasoning };
    });
    return now;
  }

  function opencodeDrift(saved, current) {
    var out = [];
    var now = opencodeModels(current);
    var before = (saved || {}).models || {};
    Object.keys(before).forEach(function (id) {
      if (!now[id]) { out.push(id + ' is no longer served'); return; }
      if (before[id].context !== now[id].context) {
        out.push(id + ': context ' + Number(before[id].context).toLocaleString('en-US') +
                 ' → ' + Number(now[id].context).toLocaleString('en-US'));
      }
      if (before[id].reasoning !== now[id].reasoning) {
        out.push(id + ': reasoning ' + before[id].reasoning + ' → ' + now[id].reasoning);
      }
    });
    Object.keys(now).forEach(function (id) {
      if (!before[id]) out.push(id + ' is new');
    });
    return out;
  }

  // ---- Profile wizard (wizzard.md) -------------------------------------

  var WIZARD_COLORS = ['#76b900', '#3fa9f5', '#f5a623', '#bd10e0', '#50e3c2',
                       '#e94e77', '#b8e986', '#9013fe', '#f8e71c', '#4a90e2'];

  function wizardColor(index) {
    return WIZARD_COLORS[Math.abs(index || 0) % WIZARD_COLORS.length];
  }

  var PROFILE_NAME = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$/;

  function wizardNameProblem(name) {
    name = String(name || '').trim();
    if (!name) return 'Give the profile a name.';
    if (!PROFILE_NAME.test(name)) {
      return 'Up to 64 letters, digits, spaces, dots, dashes or underscores, ' +
             'starting with a letter or digit.';
    }
    return '';
  }

  function _flagValue(args, name) {
    args = (args || []).map(String);
    for (var i = 0; i < args.length; i++) {
      if (args[i] === name && i + 1 < args.length) return args[i + 1];
      if (args[i].indexOf(name + '=') === 0) return args[i].slice(name.length + 1);
    }
    return null;
  }

  function _withoutFlag(args, name) {
    var out = [];
    args = (args || []).map(String);
    for (var i = 0; i < args.length; i++) {
      if (args[i] === name) {
        if (i + 1 < args.length && args[i + 1].indexOf('--') !== 0) i++;
        continue;
      }
      if (args[i].indexOf(name + '=') === 0) continue;
      out.push(args[i]);
    }
    return out;
  }

  // The wizard's draft for a saved profile. A profile the wizard made carries
  // its own draft; one that was captured is read back from its entries, each
  // cache fixed at what it was launched with — context and --max-num-seqs.
  function wizardDraftFromProfile(profile) {
    profile = profile || {};
    var saved = profile.wizard;
    if (saved && saved.models) {
      return {
        name: profile.name || '', description: profile.description || '',
        models: JSON.parse(JSON.stringify(saved.models)),
        limits: JSON.parse(JSON.stringify(saved.limits || {})),
      };
    }
    var models = (profile.entries || []).map(function (e, i) {
      var kind = e.kind || 'llm';
      var m = { id: 'm' + (i + 1), model: e.model, kind: kind,
                node_ids: (e.node_ids || []).slice() };
      if (kind === 'image') {
        m.max_image_size = e.max_image_size || 1536;
      } else if (kind === 'llm') {
        var seqs = parseInt(_flagValue(e.extra_vllm_args, '--max-num-seqs') || '0', 10);
        m.strategy = e.strategy || '';
        m.kv_cache_dtype = e.kv_cache_dtype || '';
        m.max_model_len = e.max_model_len || null;
        m.extra_vllm_args = _withoutFlag(e.extra_vllm_args, '--max-num-seqs');
        if (e.max_model_len && seqs) {
          m.mode = 'usage';
          m.sessions = seqs;
        } else {
          m.mode = 'auto';
          m.priority = 1;
        }
      }
      return m;
    });
    return { name: profile.name || '', description: profile.description || '',
             models: models, limits: {} };
  }

  // A starting placement for whatever is not placed yet: largest first, onto
  // the node with the most room left; an LLM too big for any one node is split
  // across the head and the roomiest other node (a split must include the
  // head). Sizes are on-disk GB; the planner says afterwards what really fits.
  function wizardAutoAssign(models, sizes, nodes, headId) {
    var room = {};
    (nodes || []).forEach(function (n) { room[n.node_id] = n.budget_gb || 0; });
    var out = (models || []).map(function (m) {
      return Object.assign({}, m, { node_ids: (m.node_ids || []).slice() });
    });
    out.forEach(function (m) {
      var need = (sizes[m.model] || 0) * 1.15 + 3;
      m.node_ids.forEach(function (id) { room[id] = (room[id] || 0) - need / m.node_ids.length; });
    });
    var order = out.filter(function (m) { return !m.node_ids.length; })
      .sort(function (a, b) { return (sizes[b.model] || 0) - (sizes[a.model] || 0); });
    order.forEach(function (m) {
      var need = (sizes[m.model] || 0) * 1.15 + 3;
      // A replica never goes where the same model already is.
      var taken = {};
      out.forEach(function (o) {
        if (o !== m && o.model === m.model) o.node_ids.forEach(function (id) { taken[id] = true; });
      });
      var free = Object.keys(room).filter(function (id) { return !taken[id]; })
        .sort(function (a, b) { return room[b] - room[a]; });
      if (!free.length) return;
      if (room[free[0]] >= need || m.kind !== 'llm') {
        m.node_ids = [free[0]];
        room[free[0]] -= need;
        return;
      }
      if (headId && room[headId] !== undefined && !taken[headId]) {
        var partner = free.filter(function (id) { return id !== headId; })[0];
        if (partner && room[headId] + room[partner] >= need) {
          m.node_ids = [headId, partner];
          m.strategy = m.strategy || 'tensor';
          room[headId] -= need / 2;
          room[partner] -= need / 2;
          return;
        }
      }
      m.node_ids = [free[0]];
      room[free[0]] -= need;
    });
    return out;
  }

  // One node's bar: every model a fixed part (weights, engine) and a cache
  // part, against the node's budget; the limit, if any, as a marker.
  function renderHouseholdBar(node, colorOf) {
    if (!node) return '';
    var scale = Math.max(node.budget_gb || 0, node.used_gb || 0, 1);
    var pct = function (gb) { return Math.max(0, Math.min(100, gb / scale * 100)).toFixed(2); };
    var segs = (node.segments || []).map(function (seg) {
      var color = colorOf ? colorOf(seg.id) : '#76b900';
      var title = esc(seg.model) + ': ' + seg.fixed_gb + ' GB fixed' +
        (seg.cache_gb ? ' + ' + seg.cache_gb + ' GB cache' : '');
      return '<span class="pw-seg pw-seg-fixed" title="' + title + '" style="width:' +
          pct(seg.fixed_gb) + '%;background:' + color + '"></span>' +
        (seg.cache_gb ? '<span class="pw-seg pw-seg-cache" title="' + title +
          '" style="width:' + pct(seg.cache_gb) + '%;background:' + color + '"></span>' : '');
    }).join('');
    var free = (node.free_gb || 0);
    return '<div class="pw-bar' + (node.over ? ' over' : '') + '">' + segs +
      (free > 0 ? '<span class="pw-seg pw-seg-free" style="width:' + pct(free) + '%"></span>' : '') +
      '</div>' +
      '<div class="pw-bar-text">' + (node.used_gb || 0).toFixed(1) + ' / ' +
      (node.budget_gb || 0).toFixed(1) + ' GB' +
      (node.over ? ' — <strong>' + (node.used_gb - node.budget_gb).toFixed(1) + ' GB over</strong>'
                 : (free >= 0.1 ? ' — ' + free.toFixed(1) + ' GB unplanned' : '')) +
      '</div>';
  }

  // What applying will stop: every model running on a node the profile uses
  // that the profile does not place on that node.
  function wizardStopPreview(running, models) {
    var used = {};
    var wanted = {};
    (models || []).forEach(function (m) {
      (m.node_ids || []).forEach(function (id) {
        used[id] = true;
        wanted[id + '|' + m.model] = true;
      });
    });
    return (running || []).filter(function (r) {
      return used[r.node_id] && !wanted[r.node_id + '|' + r.id];
    });
  }

  return {
    wizardColor: wizardColor,
    wizardNameProblem: wizardNameProblem,
    wizardDraftFromProfile: wizardDraftFromProfile,
    wizardAutoAssign: wizardAutoAssign,
    renderHouseholdBar: renderHouseholdBar,
    wizardStopPreview: wizardStopPreview,
    esc: esc,
    formatSeconds: formatSeconds,
    formatBytes: formatBytes,
    strategyAllowed: strategyAllowed,
    clearedNote: clearedNote,
    resumeCheckNote: resumeCheckNote,
    launchEditScope: launchEditScope,
    planQuery: planQuery,
    planFieldUpdates: planFieldUpdates,
    renderOccupancy: renderOccupancy,
    opencodeModels: opencodeModels,
    opencodeDrift: opencodeDrift,
  };
});
