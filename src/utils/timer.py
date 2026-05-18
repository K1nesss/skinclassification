from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[callable]:
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed

