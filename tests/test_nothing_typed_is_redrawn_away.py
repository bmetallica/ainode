"""The periodic refresh does not take what someone typed or produced.

Reported from the Profiles page:

    wenn ich unter profiel die aktuelle konfiguration speichern will und oben
    einen namen dafür eingeben will wird das feld ständig geleert

and "das hatten wir jetzt schon öfter": the MQTT form once, fixed for Config
alone. Every view the 5 s refresh redraws now goes through one guard,
AINode._redraw — browser-tested against a local node (the field keeps its
value across ticks, typing across a tick works, a generated OpenCode config
stays on the Server page). These tests hold the wiring down.
"""

from __future__ import annotations

from tests.dashboard_js import parts
from pathlib import Path

JS = Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" / "js"
SOURCE = (JS / "app.js").read_text() + parts()


def test_the_refresh_redraws_through_the_guard():
    refresh = SOURCE[SOURCE.index("  async refresh() {"):]
    refresh = refresh[:refresh.index("\n  },")]
    assert "this._redraw(document.getElementById('center-stage')" in refresh
    assert "switch (this.state.currentView)" not in refresh


def test_the_guard_keeps_focus_values_and_results():
    guard = SOURCE[SOURCE.index("  _redraw(root, draw) {"):]
    guard = guard[:guard.index("\n  },")]
    assert "this._typingIn(root)" in guard
    assert "_snapshotEdits(root)" in guard and "_restoreEdits(edits)" in guard
    assert "[data-keep][id]" in guard


def test_results_are_marked_to_be_kept():
    for element in ('id="opencode-config-out" data-keep',
                    'class="image-gallery" data-keep',
                    'class="profile-report" data-keep'):
        assert element in SOURCE, element


def test_a_page_is_drawn_when_it_is_opened():
    """Not on the next tick: the click that opened it counted as 'busy', so
    Config stayed empty for up to five seconds."""
    nav = SOURCE[SOURCE.index("  navigate(view) {"):]
    nav = nav[:nav.index("\n  },")]
    assert "this._renderCurrentView(true);" in nav
    assert nav.index("_renderCurrentView(true)") < nav.index("this.refresh()")


def test_the_benchmark_panel_is_guarded_too():
    assert "self._redraw(panel, function ()" in SOURCE
