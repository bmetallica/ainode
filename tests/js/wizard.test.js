// The profile wizard's pure functions (wizzard.md).
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../ainode/web/static/js/lib.js');

test('a profile name follows the store\'s rule', () => {
  assert.equal(L.wizardNameProblem('Endausbau'), '');
  assert.equal(L.wizardNameProblem('Coding_RAG 2.1'), '');
  assert.match(L.wizardNameProblem(''), /name/);
  assert.match(L.wizardNameProblem('-x'), /starting/);
  assert.match(L.wizardNameProblem('a/b'), /letters/);
  assert.match(L.wizardNameProblem('a + b'), /letters/);
});

test('a wizard profile opens with its own draft', () => {
  const draft = { models: [{ id: 'm1', model: 'org/a', mode: 'size', cache_gb: 12 }],
                  limits: { s2: 100 } };
  const out = L.wizardDraftFromProfile({ name: 'P', description: 'd', wizard: draft,
                                         entries: [] });
  assert.equal(out.name, 'P');
  assert.deepEqual(out.models, draft.models);
  assert.deepEqual(out.limits, { s2: 100 });
  out.models[0].cache_gb = 99;                 // a copy, not the saved object
  assert.equal(draft.models[0].cache_gb, 12);
});

test('a captured profile is read back from its entries', () => {
  const out = L.wizardDraftFromProfile({ name: 'C', entries: [
    { model: 'org/a', node_ids: ['s1'], max_model_len: 65536,
      extra_vllm_args: ['--foo', '--max-num-seqs', '3'] },
    { model: 'org/b', node_ids: ['s1', 's2'], strategy: 'tensor' },
    { model: 'org/flux', kind: 'image', node_ids: ['s3'], max_image_size: 1024 },
    { model: 'bge', kind: 'embedding', node_ids: ['s3'] },
  ] });
  const [a, b, img, emb] = out.models;
  assert.equal(a.mode, 'usage');
  assert.equal(a.sessions, 3);
  assert.deepEqual(a.extra_vllm_args, ['--foo']);
  assert.equal(b.mode, 'auto');
  assert.equal(b.strategy, 'tensor');
  assert.equal(img.max_image_size, 1024);
  assert.equal(emb.kind, 'embedding');
});

test('auto-assign: largest first, onto the roomiest node', () => {
  const nodes = [{ node_id: 's1', budget_gb: 100 }, { node_id: 's2', budget_gb: 110 }];
  const out = L.wizardAutoAssign(
    [{ id: 'a', model: 'small', kind: 'llm', node_ids: [] },
     { id: 'b', model: 'large', kind: 'llm', node_ids: [] }],
    { small: 20, large: 60 }, nodes, 's1');
  assert.deepEqual(out.find(m => m.id === 'b').node_ids, ['s2']);
  assert.deepEqual(out.find(m => m.id === 'a').node_ids, ['s1']);
});

test('auto-assign splits a model no node holds, across the head', () => {
  const nodes = [{ node_id: 's1', budget_gb: 100 }, { node_id: 's2', budget_gb: 100 },
                 { node_id: 's3', budget_gb: 100 }];
  const [m] = L.wizardAutoAssign([{ id: 'x', model: 'huge', kind: 'llm', node_ids: [] }],
                                 { huge: 150 }, nodes, 's1');
  assert.deepEqual(m.node_ids, ['s1', 's2']);
  assert.equal(m.strategy, 'tensor');
});

test('auto-assign keeps replicas apart and placed models where they are', () => {
  const nodes = [{ node_id: 's1', budget_gb: 100 }, { node_id: 's2', budget_gb: 90 }];
  const out = L.wizardAutoAssign(
    [{ id: 'r1', model: 'coder', kind: 'llm', node_ids: ['s1'] },
     { id: 'r2', model: 'coder', kind: 'llm', node_ids: [] }],
    { coder: 40 }, nodes, 's1');
  assert.deepEqual(out[0].node_ids, ['s1']);
  assert.deepEqual(out[1].node_ids, ['s2']);
});

test('the node bar shows every model and the overrun', () => {
  const node = { budget_gb: 100, used_gb: 110, free_gb: -10, over: true, segments: [
    { id: 'a', model: 'A', fixed_gb: 60, cache_gb: 20 },
    { id: 'b', model: 'B<x>', fixed_gb: 30, cache_gb: 0 }] };
  const html = L.renderHouseholdBar(node, () => '#123456');
  assert.equal((html.match(/pw-seg-fixed/g) || []).length, 2);
  assert.equal((html.match(/pw-seg-cache/g) || []).length, 1);
  assert.match(html, /10\.0 GB over/);
  assert.match(html, /B&lt;x&gt;/);
});

test('the stop preview names what the profile does not keep', () => {
  const running = [{ node_id: 's1', id: 'keep' }, { node_id: 's1', id: 'old' },
                   { node_id: 's3', id: 'elsewhere' }];
  const models = [{ model: 'keep', node_ids: ['s1'] }, { model: 'new', node_ids: ['s2'] }];
  assert.deepEqual(L.wizardStopPreview(running, models).map(r => r.id), ['old']);
});
