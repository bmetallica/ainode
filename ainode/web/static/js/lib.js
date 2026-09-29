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

  return {
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
