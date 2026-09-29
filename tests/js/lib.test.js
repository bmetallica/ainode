// The dashboard's pure functions, tested as functions (W1 in upgrade-fixes.md).
// Run: node --test tests/js     (pytest runs it too: tests/test_js_units.py)
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../ainode/web/static/js/lib.js');

test('esc escapes quotes too — it is used inside attributes', () => {
  assert.equal(L.esc('{"a":"<b>"}'), '{&quot;a&quot;:&quot;&lt;b&gt;&quot;}');
  assert.equal(L.esc("it's & more"), 'it&#39;s &amp; more');
  assert.equal(L.esc(null), '');
  assert.equal(L.esc(42), '42');
});

test('formatSeconds and formatBytes', () => {
  assert.equal(L.formatSeconds(59), '59s');
  assert.equal(L.formatSeconds(120), '2m');
  assert.equal(L.formatSeconds(421.9), '7m2s');
  assert.equal(L.formatBytes(0), '0 B');
  assert.equal(L.formatBytes(1536), '1.5 KB');
  assert.equal(L.formatBytes(3 * 1024 ** 3), '3.00 GB');
});

test('tensor parallelism only across 1, 2, 4 or 8 nodes', () => {
  assert.equal(L.strategyAllowed('tensor', 3), false);
  assert.equal(L.strategyAllowed('tensor', 4), true);
  assert.equal(L.strategyAllowed('pipeline', 3), true);
  assert.equal(L.strategyAllowed('tensor', 1), true);
});

test('the driven field belongs to one model on one set of nodes', () => {
  const a = L.launchEditScope({ model: 'org/a', nodes: ['n1', 'n2'] });
  assert.notEqual(a, L.launchEditScope({ model: 'org/b', nodes: ['n1', 'n2'] }));
  assert.notEqual(a, L.launchEditScope({ model: 'org/a', nodes: ['n1'] }));
  assert.equal(a, L.launchEditScope({ model: 'org/a', nodes: ['n1', 'n2'] }));
});

test('planQuery leaves out the derived half, never the dtype', () => {
  const want = { model: 'm', nodes: [], strategy: '', max_model_len: '4096',
                 concurrency: '2', kv_cache_dtype: 'fp8' };
  assert.deepEqual(L.planQuery(want, 'len'),
                   { model: 'm', max_model_len: '4096', kv_cache_dtype: 'fp8' });
  assert.deepEqual(L.planQuery(want, 'seqs'),
                   { model: 'm', concurrency: '2', kv_cache_dtype: 'fp8' });
});

test('an idle pool is named, a full one is not', () => {
  const plan = { node_total_gb: 128, reserved_per_node_gb: 100,
                 needed_per_node_gb: 60, weights_per_node_gb: 50,
                 overhead_per_node_gb: 3, cache_used_per_node_gb: 7,
                 max_model_len: 65536 };
  assert.match(L.renderOccupancy(plan, '1'), /40\.0 GB of the pool is reserved/);
  assert.doesNotMatch(L.renderOccupancy({ ...plan, needed_per_node_gb: 99 }, '8'),
                      /reserved and will not be used/);
  assert.equal(L.renderOccupancy(null, '1'), '');
  // The form's value is escaped, not pasted.
  assert.match(L.renderOccupancy(plan, '<x>'), /&lt;x&gt; requests/);
});

test('a stale OpenCode config names what changed', () => {
  const saved = { models: { 'org/m': { context: 608512, reasoning: false },
                            'org/gone': { context: 4096, reasoning: false } } };
  const now = { config: { provider: { vllm: { models: {
    'org/m': { limit: { context: 131072 }, reasoning: true },
    'org/new': { limit: { context: 8192 }, reasoning: false } } } } } };
  assert.deepEqual(L.opencodeDrift(saved, now), [
    'org/m: context 608,512 → 131,072',
    'org/m: reasoning false → true',
    'org/gone is no longer served',
    'org/new is new',
  ]);
  assert.deepEqual(L.opencodeDrift(
    { models: L.opencodeModels(now) }, now), []);
});
