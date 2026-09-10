"""Tests for ainode.core.config."""

import json
from ainode.core.config import NodeConfig


def test_config_defaults():
    """Default config has expected values."""
    config = NodeConfig()
    assert config.api_port == 8000
    assert config.web_port == 3000
    assert config.host == "0.0.0.0"
    assert config.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert config.gpu_memory_utilization == 0.5  # lowered for unified-memory (was 0.9)
    assert config.onboarded is False
    assert config.node_id is None


def test_config_save_load(tmp_path, monkeypatch):
    """Config round-trips through save/load."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")

    config = NodeConfig(node_id="abc123", model="test-model", api_port=9000)
    config.save()

    loaded = NodeConfig.load()
    assert loaded.node_id == "abc123"
    assert loaded.model == "test-model"
    assert loaded.api_port == 9000


def test_config_load_missing(tmp_path, monkeypatch):
    """Loading from nonexistent file returns defaults."""
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "nope.json")
    config = NodeConfig.load()
    assert config.api_port == 8000
    assert config.node_id is None


def test_config_load_ignores_unknown_fields(tmp_path, monkeypatch):
    """Unknown fields in config file are silently ignored."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"node_id": "x", "unknown_field": True}))
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)

    config = NodeConfig.load()
    assert config.node_id == "x"
    assert not hasattr(config, "unknown_field")


def test_fabric_fields_default_to_detection():
    """coord_interface / rdma_hcas empty means "detect", which for a
    direct-attach cluster resolves back to cluster_interface — so an existing
    2-/4-node TP install sees no change."""
    config = NodeConfig()
    assert config.coord_interface == ""
    assert config.rdma_hcas == []
    assert config.cluster_interface == "eno1"


def test_discovery_port_matches_installer():
    """The code default used to be 5678 while scripts/install.sh wrote 5679,
    so a hand-installed node could not see an installer-provisioned one."""
    assert NodeConfig().discovery_port == 5679


def test_fabric_fields_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")

    NodeConfig(
        cluster_interface="enP2p1s0f1np1",
        coord_interface="enP7s7",
        rdma_hcas=["rocep1s0f0", "roceP2p1s0f1"],
    ).save()

    loaded = NodeConfig.load()
    assert loaded.cluster_interface == "enP2p1s0f1np1"
    assert loaded.coord_interface == "enP7s7"
    assert loaded.rdma_hcas == ["rocep1s0f0", "roceP2p1s0f1"]


def test_fabric_fields_are_patchable():
    """A mesh node has to be able to set these without editing config.json
    by hand, so PATCH /api/config must expose them."""
    from ainode.api.server import PATCHABLE_CONFIG_FIELDS

    assert {"cluster_interface", "coord_interface", "rdma_hcas"} <= PATCHABLE_CONFIG_FIELDS
    # Guard the "keep this list tight" comment above it.
    assert "cluster_secret" not in PATCHABLE_CONFIG_FIELDS
    assert "hf_token" not in PATCHABLE_CONFIG_FIELDS
