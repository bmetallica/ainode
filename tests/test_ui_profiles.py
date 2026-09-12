"""The profile tab has to be reachable and complete without curl.

The whole point of profiles is that an operator configures a three-model
deployment once and clicks it back afterwards; a feature that only exists as
an API would not solve that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()
STYLE = (WEB / "static" / "css" / "style.css").read_text()


class TestProfilesTab:
    def test_nav_pill_and_view_exist(self):
        assert 'data-view="profiles"' in INDEX
        assert 'id="view-profiles"' in INDEX
        assert 'id="profiles-content"' in INDEX

    def test_the_view_is_rendered_when_navigated_to(self):
        # A view with no dispatch entry renders empty and looks broken.
        assert "case 'profiles':" in APP_JS
        assert "this.renderProfiles()" in APP_JS

    def test_capture_is_offered_as_the_way_in(self):
        assert 'id="profile-capture-btn"' in APP_JS
        assert "/api/profiles/capture" in APP_JS

    @pytest.mark.parametrize("call", [
        "'/api/profiles'",
        "/api/profiles/capture",
        "'/api/profiles/' + encodeURIComponent(name) + '/apply'",
        "'/api/profiles/' + encodeURIComponent(name) + '/default'",
    ])
    def test_every_action_has_a_button_behind_it(self, call):
        assert call in APP_JS

    def test_apply_warns_that_it_stops_things(self):
        # "Apply" reads like "add" to most people; it converges.
        assert "will be stopped" in APP_JS

    def test_the_default_is_explained_not_just_flagged(self):
        assert "loaded at startup" in APP_JS

    def test_per_entry_errors_are_rendered(self):
        assert "_renderApplyReport" in APP_JS
        assert "profile-report-line" in APP_JS

    def test_the_tab_is_styled(self):
        for cls in (".profile-card", ".profile-table", ".profile-report-line"):
            assert cls in STYLE
