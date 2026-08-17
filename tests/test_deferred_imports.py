#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import builtins
import json
import math
import os
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fresh_import_modules(module: str) -> dict[str, bool]:
    script = (
        "import importlib,json,sys; "
        f"importlib.import_module({module!r}); "
        "print(json.dumps({name: name in sys.modules for name in "
        "['nltk', 'pyloudnorm', 'scipy', 'scipy.signal']}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_string_import_defers_nltk_and_scipy():
    modules = _fresh_import_modules("pipecat.utils.string")

    assert modules["nltk"] is False
    assert modules["scipy"] is False


def test_frames_import_defers_pyloudnorm_and_scipy():
    modules = _fresh_import_modules("pipecat.frames.frames")

    assert modules["pyloudnorm"] is False
    assert modules["scipy"] is False


def test_calculate_audio_volume_preserves_loudness_result():
    from pipecat.audio.utils import calculate_audio_volume

    sample_rate = 16_000
    samples = [
        int(1200 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(sample_rate // 2)
    ]
    audio = struct.pack(f"<{len(samples)}h", *samples)

    assert calculate_audio_volume(audio, sample_rate) == pytest.approx(0.7783933811238548)


def test_deferred_loaders_coalesce_concurrent_calls(monkeypatch):
    from pipecat.audio import utils as audio_utils
    from pipecat.utils.string import _sent_tokenizer

    original_import = builtins.__import__
    import_started = threading.Event()
    release_import = threading.Event()
    pyloudnorm_imports = 0

    def controlled_import(name, *args, **kwargs):
        nonlocal pyloudnorm_imports
        if name == "pyloudnorm":
            pyloudnorm_imports += 1
            import_started.set()
            assert release_import.wait(timeout=2)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(audio_utils, "_PYLOUDNORM_MODULE", None)
    monkeypatch.setattr(builtins, "__import__", controlled_import)
    with ThreadPoolExecutor(max_workers=8) as executor:
        tokenizers = tuple(executor.map(lambda _: _sent_tokenizer(), range(8)))
        loudness_futures = [executor.submit(audio_utils._pyloudnorm) for _ in range(8)]
        assert import_started.wait(timeout=1)
        release_import.set()
        loudness_modules = tuple(future.result(timeout=1) for future in loudness_futures)

    assert all(tokenizer is tokenizers[0] for tokenizer in tokenizers)
    assert all(module is loudness_modules[0] for module in loudness_modules)
    assert pyloudnorm_imports == 1
