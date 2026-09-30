"""Applying a profile over a model that already runs.

Reported: "unsloth/Qwen3.8-27B-NVFP4 … failed … cost 80 GB the last time it
ran here, and the roomiest node has 20 GB free" — the 27B was running (with
--max-num-seqs 13, the profile now says 11), the apply launched the new one
beside it, and the admission check counted the old one's memory.
"""

from __future__ import annotations

import asyncio

from ainode.profiles import apply
from ainode.profiles.apply import _flags_present

RUNNING = ["--max-num-seqs", "11", "--enable-prefix-caching", "--reasoning-parser", "qwen3",
           "--tool-call-parser", "qwen3_xml", "--enable-auto-tool-choice",
           "--speculative_config", '{"method":"qwen3_5_mtp","num_speculative_tokens":2}']


class TestWhatCountsAsUnchanged:
    def test_the_recipe_and_launch_flags_beside_it_do_not_matter(self):
        assert _flags_present(["--max-num-seqs", "11"], RUNNING)

    def test_a_different_value_does(self):
        assert not _flags_present(["--max-num-seqs", "13"], RUNNING)

    def test_a_missing_flag_does(self):
        assert not _flags_present(["--enforce-eager"], RUNNING)

    def test_spelling_of_the_flag_does_not(self):
        assert _flags_present(["--max_num_seqs=11"], RUNNING)

    def test_a_dropped_flag_must_be_gone(self):
        assert not _flags_present(["drop:--enable-prefix-caching"], RUNNING)
        assert _flags_present(["drop:--enforce-eager"], RUNNING)

    def test_no_flags_stated_is_always_fine(self):
        assert _flags_present([], RUNNING)


class TestARelaunchStopsTheOldOneFirst:
    def test_order(self, monkeypatch):
        from ainode.profiles.store import ProfileEntry

        calls = []

        class _Record:
            model = "m"
            api_port = 8001
            instance_id = "8001"

        class _Instance:
            record = _Record()

        class _Manager:
            def instances(self):
                return [_Instance()]

        monkeypatch.setattr(apply, "_entry_runs_here", lambda app, e: True)
        monkeypatch.setattr(apply, "_entry_matches", lambda e, i, c: False)
        monkeypatch.setattr(apply, "_stop_llm", lambda app, inst: calls.append("stop"))

        async def _start(app, entry):
            calls.append("start")
            return {"model": entry.model, "action": "failed", "ok": False}

        monkeypatch.setattr(apply, "_start_llm_entry", _start)
        app = {"config": None, "instances": _Manager()}
        asyncio.run(apply._start_wanted(app, [ProfileEntry(model="m", node_ids=["h"])],
                                        wait=False))
        assert calls == ["stop", "start"]
