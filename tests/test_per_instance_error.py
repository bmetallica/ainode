"""Each instance's failure belongs to that instance.

Two models on two different machines showed byte-identical error text on the
dashboard — same message, same process id, same second — because the card
rendered the NODE's load_error, of which there is one per node. A failure you
cannot attribute is a failure you cannot diagnose, and it sent the operator
looking for a shared cause that did not exist.
"""

from __future__ import annotations

from pathlib import Path

from ainode.api.server import _stamp_load_state
from ainode.discovery.instance import InstanceRecord

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class _Backend:
    load_error = "RuntimeError: cudaErrorIllegalInstruction"
    load_phase = "failed"
    load_detail = "capturing CUDA graphs"


class _Instance:
    def __init__(self):
        self.record = InstanceRecord(instance_id="h:m", model="a/b")
        self.backend = _Backend()


class TestTheRecordCarriesIt:
    def test_the_record_has_the_fields(self):
        record = InstanceRecord()
        assert record.load_error == "" and record.load_phase == ""

    def test_the_backend_state_is_stamped_onto_it(self):
        inst = _Instance()
        _stamp_load_state(inst)
        assert inst.record.load_error.startswith("RuntimeError")
        assert inst.record.load_phase == "failed"
        assert inst.record.load_detail == "capturing CUDA graphs"

    def test_it_crosses_the_wire(self):
        inst = _Instance()
        _stamp_load_state(inst)
        assert "load_error" in inst.record.to_dict()

    def test_a_backend_without_the_fields_is_harmless(self):
        inst = _Instance()
        inst.backend = object()
        _stamp_load_state(inst)
        assert inst.record.load_error == ""

    def test_an_older_record_still_parses(self):
        # A node on a previous build sends no such fields.
        record = InstanceRecord.from_dict({"model": "a/b", "status": "serving"})
        assert record.model == "a/b" and record.load_error == ""


class TestTheCardUsesItsOwn:
    def test_the_stacked_card_carries_the_instance_error(self):
        assert "error: inst.load_error" in APP_JS

    def test_the_renderer_prefers_the_instance_over_the_node(self):
        assert "var instError" in APP_JS
        assert "var instPhase" in APP_JS
        # The old form painted one node value on every card.
        assert "(phase === 'failed' && loadError)" not in APP_JS

    def test_the_progress_bar_follows_the_instance_too(self):
        assert "PHASE_INFO[instPhase]" in APP_JS
