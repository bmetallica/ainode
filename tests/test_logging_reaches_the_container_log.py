"""AINode's own INFO lines reach stderr — the container log.

Reported: `docker logs ainode` on a node that was being asked to launch a
distributed model showed the banner and nothing else. Nothing configured
logging, so Python printed warnings only.
"""

from __future__ import annotations

import logging


def test_info_lines_are_printed(capsys, monkeypatch):
    from ainode.cli.main import setup_logging

    logger = logging.getLogger("ainode")
    saved = (list(logger.handlers), logger.level)
    try:
        logger.handlers[:] = []
        logger.setLevel(logging.NOTSET)
        monkeypatch.delenv("AINODE_LOG_LEVEL", raising=False)
        setup_logging()
        setup_logging()          # twice: one handler, not two
        logging.getLogger("ainode.engine.backends.eugr").info("Starting distributed vLLM")
        err = capsys.readouterr().err
        assert err.count("Starting distributed vLLM") == 1
        assert "INFO" in err and "ainode.engine.backends.eugr" in err
    finally:
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])


def test_the_level_can_be_set(capsys, monkeypatch):
    from ainode.cli.main import setup_logging

    logger = logging.getLogger("ainode")
    saved = (list(logger.handlers), logger.level)
    try:
        logger.handlers[:] = []
        logger.setLevel(logging.NOTSET)
        monkeypatch.setenv("AINODE_LOG_LEVEL", "WARNING")
        setup_logging()
        logging.getLogger("ainode.x").info("quiet")
        assert "quiet" not in capsys.readouterr().err
    finally:
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])
