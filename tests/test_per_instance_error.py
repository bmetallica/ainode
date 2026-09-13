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


class TestTheFieldsSurviveTheWire:
    """from_dict had a hardcoded field list, and it drifted.

    load_error, load_phase and load_detail were added to the record and
    serialised by to_dict — and then dropped on arrival at the head. So the
    per-instance state never crossed the wire at all, and the dashboard filled
    the blank with whatever the head's own engine was doing.
    """

    def test_a_round_trip_keeps_the_load_state(self):
        record = InstanceRecord(model="a/b", status="starting",
                                load_phase="loading_weights",
                                load_detail="reading the weights from disk",
                                load_error="boom")
        back = InstanceRecord.from_dict(record.to_dict())
        assert back.load_phase == "loading_weights"
        assert back.load_detail == "reading the weights from disk"
        assert back.load_error == "boom"

    def test_every_field_round_trips(self):
        # The list is derived from the dataclass now, so a field added later
        # cannot be forgotten here.
        record = InstanceRecord(instance_id="h:m", model="a/b",
                                head_node_id="h", peer_ips=["10.0.0.2"],
                                api_port=8001, tensor_parallel_size=2)
        back = InstanceRecord.from_dict(record.to_dict())
        assert back.to_dict() == record.to_dict()

    def test_an_unknown_key_is_still_ignored(self):
        # A node on a newer build must not break an older head.
        record = InstanceRecord.from_dict({"model": "a/b", "invented_later": 1})
        assert record.model == "a/b"


class TestTheCardDoesNotBorrowTheNodesPhase:
    """Observed: a model still loading on another machine showed READY · 100%.

    The card's own phase was empty (the wire dropped it), and the renderer fell
    back to the node's — which was ready, because the head's own engine was.
    """

    def test_a_card_with_its_own_state_never_reads_the_nodes(self):
        assert "var hasOwnState = inst.phase !== undefined" in APP_JS
        assert "PHASE_INFO[instPhase] || PHASE_INFO[phase]" not in APP_JS

    def test_an_empty_phase_falls_back_to_its_own_status(self):
        # Not to the node's. A starting instance reads STARTING even when the
        # phase detail never arrived.
        assert "inst.status === 'READY' ? 'ready' : 'starting'" in APP_JS

    def test_the_detail_line_is_per_card_too(self):
        assert "instDetail = hasOwnState" in APP_JS
