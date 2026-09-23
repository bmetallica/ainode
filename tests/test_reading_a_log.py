"""An engine log is not a text file in the strict sense.

From the operator's side:

    $ grep -n "recipe environment" ~/.ainode/logs/vllm.log
    grep: /home/admin/.ainode/logs/vllm.log: binary file matches

grep says it in its own way and answers with nothing. Python said it by
raising UnicodeDecodeError — a ValueError, not an OSError — from
Path.read_text(), which the backends guarded against with `except OSError`.
So the log vanished into an empty string, and the UI showed an empty box
exactly when there was something to read.
"""

from __future__ import annotations

import inspect

from ainode.engine.logs import read_log_tail


class TestUndecodableBytes:
    def test_a_truncated_character_does_not_lose_the_log(self, tmp_path):
        log = tmp_path / "vllm.log"
        # A multi-byte character cut in half, as a killed container leaves it.
        log.write_bytes("loading weights\n".encode() + b"\xe2\x9c" + b"\ndone\n")
        text = read_log_tail(log)
        assert "loading weights" in text
        assert "done" in text

    def test_a_nul_byte_does_not_either(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_bytes(b"before\n\x00\x00\nafter\n")
        assert "before" in read_log_tail(log)
        assert "after" in read_log_tail(log)

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert read_log_tail(tmp_path / "nope.log") == ""


class TestTheTail:
    def test_it_returns_the_last_lines(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_text("\n".join(f"line {i}" for i in range(500)) + "\n")
        text = read_log_tail(log, 10)
        assert text.splitlines()[0] == "line 490"
        assert len(text.splitlines()) == 10

    def test_a_progress_bar_is_split_into_its_frames(self, tmp_path):
        # One 4000-character line of carriage returns is not "one line" in
        # any sense a tail of N lines can use.
        log = tmp_path / "vllm.log"
        log.write_bytes(b"start\nLoading 10%\rLoading 50%\rLoading 100%\nend\n")
        lines = read_log_tail(log, 2).splitlines()
        assert lines == ["Loading 100%", "end"]

    def test_a_huge_log_is_not_read_whole(self, tmp_path):
        # It grows across every launch a node has ever done, and the UI polls.
        log = tmp_path / "vllm.log"
        log.write_bytes(b"x" * (3 * 1024 * 1024) + b"\ntail line\n")
        assert read_log_tail(log, 1).strip() == "tail line"


class TestEveryBackendUsesIt:
    def test_the_eugr_backend_does(self):
        from ainode.engine.backends.eugr import EugrBackend

        assert "read_log_tail" in inspect.getsource(EugrBackend.logs)

    def test_the_diffusers_backend_does(self):
        from ainode.engine.backends.diffusers import DiffusersBackend

        assert "read_log_tail" in inspect.getsource(DiffusersBackend.logs)

    def test_neither_guards_only_against_oserror_any_more(self):
        from ainode.engine.backends.diffusers import DiffusersBackend
        from ainode.engine.backends.eugr import EugrBackend

        for backend in (EugrBackend, DiffusersBackend):
            source = inspect.getsource(backend.logs)
            assert "read_text()" not in source
