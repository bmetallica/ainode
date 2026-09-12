"""A settings form must not rebuild itself while someone is typing into it.

Observed on hardware: entering the MQTT broker, username and password was
close to impossible, because the periodic refresh re-rendered the config
section every few seconds and every keystroke since the last poll was lost.
It was never specific to that page — every form under Config had it; the MQTT
one just has the most fields to fill in.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


def _refresh_config_branch() -> str:
    """The body of refresh()'s `case 'config':`."""
    start = APP_JS.index("      case 'config':")
    return APP_JS[start:start + 900]


class TestConfigViewIsNotRebuiltByPolling:
    def test_the_poll_branch_is_guarded(self):
        branch = _refresh_config_branch()
        assert "_configViewInitialized" in branch
        # An unguarded `this.renderConfig()` on every poll is the defect.
        assert not re.search(r"case 'config':\s*\n\s*this\.renderConfig\(\)", branch)

    def test_entering_the_view_forces_a_render(self):
        # Guarding the poll must not leave the page blank when you open it.
        assert "if (view === 'config' && prevView !== 'config')" in APP_JS
        assert "_configViewInitialized = false" in APP_JS

    def test_switching_section_still_renders(self):
        nav = APP_JS[APP_JS.index("nav.querySelectorAll('.config-nav-item').forEach"):]
        click = nav[:nav.index("self.renderConfigSection();") + 30]
        assert "_configViewInitialized = true" in click

    def test_monitoring_offers_an_explicit_refresh(self):
        # The live status block was the only thing the automatic rebuild was
        # good for, so it gets a button instead.
        assert "cfg-mqtt-refresh" in APP_JS
        assert "renderConfigMonitoring();" in APP_JS
