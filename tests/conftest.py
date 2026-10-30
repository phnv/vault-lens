import os
import urllib.request

import pytest


def es_url() -> str:
    return os.environ.get("ES_URL", "http://localhost:9200")


def es_up() -> bool:
    try:
        urllib.request.urlopen(es_url(), timeout=2)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if es_up():
        return
    skip = pytest.mark.skip(reason=f"Elasticsearch not reachable at {es_url()}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
