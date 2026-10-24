import pytest

from llmbench.metrics import percentile, summarize


def test_percentile_matches_known_values():
    data = [1, 2, 3, 4, 5]
    assert percentile(data, 50) == 3
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 5
    assert percentile(data, 25) == 2
    assert percentile([10, 20], 50) == 15


def test_percentile_single_and_empty():
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize():
    s = summarize([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["median"] == 3 and s["min"] == 1 and s["max"] == 5
    assert s["p95"] == pytest.approx(4.8)
    assert summarize([]) is None
    assert summarize([2.0])["stdev"] == 0.0
