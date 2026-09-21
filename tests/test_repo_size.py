"""A model that fits was hidden as "Too large for cluster".

AINode said 306 GB for nvidia/DeepSeek-V4-Flash-nvfp4-DSpark and hid it from
the search. The repo is 164.1 GiB — 82 GiB per node across two, comfortably
inside the 106 GiB each can address.

The size came from the Hub's safetensors dtype breakdown, which counts tensor
ELEMENTS by declared dtype. A packed 4-bit checkpoint stores many values in
one U8 or I32 element, so the arithmetic multiplies a packed count by the
container's width. Measured:

    MiniMax-M2.7 AWQ-4bit     111.6 GiB actual    912 GB estimated   8x out
    DeepSeek-V4-Flash NVFP4   164.1 GiB actual    306 GB estimated   1.9x out
    Gemma 4 26B NVFP4          17.5 GiB actual   18.8 GB estimated   close

The error is not uniform, so it cannot be corrected with a factor. The Hub
reports the real byte count as `usedStorage`, in the same response.
"""

from __future__ import annotations

from ainode.models.registry import _safetensors_size_gb, repo_size_gb


class _Safetensors:
    def __init__(self, parameters):
        self.parameters = parameters


class _Model:
    def __init__(self, used_storage=None, parameters=None):
        if used_storage is not None:
            self.used_storage = used_storage
        self.safetensors = _Safetensors(parameters) if parameters else None


#: The real breakdown of nvidia/DeepSeek-V4-Flash-nvfp4-DSpark.
PACKED = {"U8": 296_352_743_424, "F8_E4M3": 6_304_038_912,
          "BF16": 1_483_567_488, "F32": 37_741_630, "I64": 2_327_040}


class TestTheExactFigureWins:
    def test_used_storage_is_used(self):
        # 176_173_113_880 bytes = 164.1 GiB = the measured repo size.
        assert repo_size_gb(_Model(used_storage=176_173_113_880,
                                   parameters=PACKED)) == 176.2

    def test_it_beats_the_estimate_by_a_wide_margin(self):
        """The estimate is what marked this model too large for the cluster."""
        model = _Model(used_storage=176_173_113_880, parameters=PACKED)
        assert _safetensors_size_gb(model.safetensors) > 300
        assert repo_size_gb(model) < 200

    def test_the_camelcase_name_is_accepted_too(self):
        """huggingface_hub exposes used_storage; a raw API dict says
        usedStorage. Both reach this."""
        model = _Model(parameters=PACKED)
        model.usedStorage = 176_173_113_880
        assert repo_size_gb(model) == 176.2


class TestTheFallback:
    def test_no_storage_figure_falls_back_to_the_estimate(self):
        model = _Model(parameters={"BF16": 1_000_000_000})
        assert repo_size_gb(model) == 2.0

    def test_zero_is_not_a_size(self):
        """A repo that reports 0 has not been measured, and 0 GB would read as
        "fits anywhere"."""
        model = _Model(used_storage=0, parameters={"BF16": 1_000_000_000})
        assert repo_size_gb(model) == 2.0

    def test_nonsense_does_not_raise(self):
        model = _Model(used_storage="später", parameters={"BF16": 500_000_000})
        assert repo_size_gb(model) == 1.0

    def test_nothing_at_all_is_zero_not_an_error(self):
        assert repo_size_gb(_Model()) == 0.0


class TestTheListEndpointCannotBeAskedForIt:
    """The first fix broke the search outright.

        Bad request:
        * Invalid option: expected one of "author"|…|"safetensors"|… at expand[1]

    usedStorage is valid on the single-model endpoint — which is where it was
    verified — and rejected on the list endpoint. list_models then raised, the
    search returned nothing, and "downloading from HF stopped working".
    """

    def test_the_list_call_does_not_ask_for_it(self):
        import inspect

        from ainode.models.registry import ModelManager

        source = inspect.getsource(ModelManager.search_huggingface)
        # The bug this guards: usedStorage is valid on the single-model
        # endpoint and a 400 on the list one, and asking for it there turned
        # every search into no results at all. What else is expanded may grow
        # — tags and the library name now decide whether a repo is loadable
        # here — but this must never be among it.
        expanded = source.split("expand=")[1].split("]")[0]
        assert '"usedStorage"' not in expanded
        assert '"safetensors"' in expanded

    def test_the_reason_is_recorded_where_someone_would_re_add_it(self):
        import inspect

        from ainode.models.registry import ModelManager

        source = inspect.getsource(ModelManager.search_huggingface)
        assert "NOT usedStorage" in source

    def test_the_exact_lookup_uses_the_single_model_endpoint(self):
        import inspect

        from ainode.models.registry import exact_repo_size_gb

        source = inspect.getsource(exact_repo_size_gb)
        assert 'model_info(repo_id, expand=["usedStorage"])' in source

    def test_an_unreachable_hub_yields_no_size_rather_than_raising(self):
        from unittest import mock

        from ainode.models import registry

        with mock.patch("huggingface_hub.HfApi.model_info",
                        side_effect=OSError("no route to host")):
            assert registry.exact_repo_size_gb("org/model") == 0.0


class TestSharpeningIsBounded:
    """A search box that takes half a minute is a search box nobody uses."""

    def _rows(self, n, gb):
        return [{"hf_repo": "org/m%d" % i, "size_gb": gb} for i in range(n)]

    def test_small_models_are_not_looked_up(self):
        from unittest import mock

        from ainode.models import registry

        rows = self._rows(10, 8.0)
        with mock.patch.object(registry, "exact_repo_size_gb") as lookup:
            registry._sharpen_sizes(rows)
        lookup.assert_not_called()

    def test_large_ones_are(self):
        from unittest import mock

        from ainode.models import registry

        rows = self._rows(3, 200.0)
        with mock.patch.object(registry, "exact_repo_size_gb", return_value=176.2):
            registry._sharpen_sizes(rows)
        assert all(r["size_gb"] == 176.2 for r in rows)

    def test_the_number_of_lookups_is_capped(self):
        from unittest import mock

        from ainode.models import registry

        rows = self._rows(50, 200.0)
        with mock.patch.object(registry, "exact_repo_size_gb",
                               return_value=176.2) as lookup:
            registry._sharpen_sizes(rows)
        assert lookup.call_count == registry._EXACT_SIZE_LOOKUPS

    def test_a_repo_that_reports_nothing_keeps_its_estimate(self):
        from unittest import mock

        from ainode.models import registry

        rows = self._rows(2, 200.0)
        with mock.patch.object(registry, "exact_repo_size_gb", return_value=0.0):
            registry._sharpen_sizes(rows)
        assert all(r["size_gb"] == 200.0 for r in rows)
