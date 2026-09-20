"""The instance card says four things; the rest is one click away.

Everything landed on that card over time, because every new piece of
information needed somewhere to go: load phases, the full error text, the
compile-cache button, the assistant and its answer, relaunch, the degraded
note, the progress bar. The panel that is looked at most often had become the
least readable one.

What is tested here is the split — what stays on the card, what moved, and
the two things that must not break in the move: the raw error is still shown
in full, and a dialog left open must keep up with a load that is still
running.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()
CSS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
       "static" / "css" / "style.css").read_text()

#: The card's own markup, from the renderer's template literal onwards.
CARD = APP_JS.split("container.innerHTML = instances.map")[1].split("}).join('')")[0]


class TestWhatStaysOnTheCard:
    def test_the_model_name(self):
        assert "instance-model" in CARD

    def test_one_status_word(self):
        assert "instance-state " in CARD
        assert "self.instanceState(inst" in CARD

    def test_details_and_unload_and_nothing_else(self):
        buttons = re.findall(r"<button[^>]*class=\"([^\"]+)\"", CARD)
        assert any("instance-delete" in b for b in buttons)
        assert "data-details=" in CARD
        # Two buttons. A third would be the start of the old card coming back.
        assert len(buttons) == 2, buttons

    def test_the_progress_bar_is_gone_from_the_card(self):
        assert "width:90px;height:5px" not in CARD

    def test_the_error_text_is_not_on_the_card(self):
        assert "instance-failed-note" not in CARD

    def test_the_cache_button_is_not_on_the_card(self):
        assert "instance-cache-clear" not in CARD


class TestTheStatusWord:
    def _state(self, name):
        block = APP_JS.split("instanceState(inst, phase) {")[1].split("\n  },")[0]
        return block

    def test_every_state_has_a_word_and_a_colour(self):
        block = self._state("")
        for label, cls in (("DEGRADED", "degraded"), ("FAILED", "failed"),
                           ("READY", "ready"), ("STOPPED ANSWERING", "stopped"),
                           ("LOADING", "loading")):
            assert label in block
            assert f"cls: '{cls}'" in block
            assert f".instance-state.{cls}" in CSS or cls in CSS

    def test_a_load_still_shows_how_far_it_got(self):
        # One word is not the same as no information: a load that is 40%
        # through says so, because the alternative reads as a hang.
        assert "'LOADING · ' + info[1] + '%'" in APP_JS

    def test_degraded_outranks_everything(self):
        # An instance that lost a rank is still "READY" by status and cannot
        # serve. Showing READY there would be the most misleading word on the
        # panel.
        block = self._state("")
        assert block.index("DEGRADED") < block.index("READY")


class TestWhatMovedIntoTheDialog:
    DIALOG = APP_JS.split("renderInstanceDetails(inst) {")[1].split(
        "\n  bindInstanceActions")[0]

    def test_the_raw_error_in_full(self):
        assert "instance-failed-note" in self.DIALOG
        assert "this.esc(error)" in self.DIALOG

    def test_the_load_timing(self):
        assert "loadTimelineBlock" in self.DIALOG

    def test_the_assistant(self):
        assert "assistBlock" in self.DIALOG

    def test_the_kernel_cache_and_the_relaunch(self):
        assert "instance-cache-clear" in self.DIALOG
        assert "instance-relaunch" in self.DIALOG

    def test_the_nodes_and_the_split(self):
        assert "'Nodes'" in self.DIALOG and "'Split'" in self.DIALOG

    def test_the_degraded_explanation(self):
        assert "instance-degraded" in self.DIALOG


class TestTheDialogKeepsUp:
    def test_it_is_re_rendered_while_it_is_open(self):
        # A load's phases move and the assistant answers late. A dialog frozen
        # at the moment it was opened would be the one place in the UI that
        # lies.
        assert "if (this._detailsFor && this._instanceCache[this._detailsFor])" \
            in APP_JS
        assert "this.renderInstanceDetails(this._instanceCache[this._detailsFor])" \
            in APP_JS

    def test_closing_it_stops_that(self):
        assert "this._detailsFor = null;" in APP_JS

    def test_unloading_from_the_dialog_closes_it(self):
        # Otherwise the dialog stays open describing an instance that no
        # longer exists, and re-renders forever from a stale cache entry.
        block = APP_JS.split("bindInstanceActions(root) {")[1]
        assert "self.closeInstanceDetails();" in block

    def test_the_actions_are_bound_where_they_are_drawn(self):
        # One binder, called with whatever root the buttons live in, so the
        # dialog and any future home for them cannot disagree.
        assert "this.bindInstanceActions(modal);" in APP_JS
