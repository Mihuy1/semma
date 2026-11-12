from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs the real model on a GPU; run with GEMMA_GPU_TESTS=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("GEMMA_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set GEMMA_GPU_TESTS=1 to run model tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
