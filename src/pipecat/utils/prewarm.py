#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Background loading of Pipecat's deferred third-party imports."""

from loguru import logger


def warm_deferred_imports() -> None:
    """Load deferred imports while network services are connecting.

    This function is blocking and CPU-bound, so event-loop callers should run
    it in a thread. Repeat calls are cheap. A dependency that fails to load is
    left for its point of use to report.
    """
    try:
        from pipecat.utils.string import _sent_tokenizer

        _sent_tokenizer()
    except Exception as e:
        logger.trace(f"Could not warm the NLTK sentence tokenizer: {e}")

    try:
        from pipecat.audio.utils import _pyloudnorm

        _pyloudnorm()
    except Exception as e:
        logger.trace(f"Could not warm pyloudnorm: {e}")
