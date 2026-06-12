"""Tests for the API-free parts of experiment.py.

Run with: uv run --with pytest pytest -q
"""

import random
import string
from pathlib import Path

import pytest

import experiment


def _csv(tmp_path: Path, n: int) -> Path:
    p = tmp_path / "urls.csv"
    rows = ["url,type"] + [f"http://example.com/path/{i},benign" for i in range(n)]
    p.write_text("\n".join(rows) + "\n")
    return p


def test_url_dataset_disjoint_and_deterministic(tmp_path: Path) -> None:
    csv = _csv(tmp_path, 2000)
    members, non_members = experiment.url_dataset(1000, 500, 0, csv)
    assert len(members) == 1000
    assert len(non_members) == 500
    assert not set(members) & set(non_members)
    assert (members, non_members) == experiment.url_dataset(1000, 500, 0, csv)


def test_queries_split_and_membership(tmp_path: Path) -> None:
    members, non_members = experiment.url_dataset(1000, 500, 0, _csv(tmp_path, 2000))
    member_qs, non_member_qs = experiment.make_queries(
        members, non_members, 200, seed=1
    )
    assert len(member_qs) == 100
    assert len(non_member_qs) == 100
    assert set(member_qs) <= set(members)
    assert set(non_member_qs) <= set(non_members)


def test_score() -> None:
    s = experiment.score(["YES"] * 9 + [None], ["NO"] * 98 + ["YES", None])
    assert s["false_negative_rate"] == 0.1
    assert abs(s["false_positive_rate"] - 0.01) < 1e-9
    assert s["num_failed_queries"] == 2


def test_run_queries_aborts_on_systematic_failure() -> None:
    calls = []

    def failing_ask(item):
        calls.append(item)
        return None

    with pytest.raises(SystemExit):
        experiment.run_queries(failing_ask, ["a"] * 10, ["b"] * 10)
    assert len(calls) == experiment.FAIL_FAST_QUERIES


def test_run_queries_splits_answers() -> None:
    member_qs, non_member_qs = ["a", "b", "c"], ["x", "y"]
    answers = {"a": "YES", "b": "YES", "c": None, "x": "NO", "y": "YES"}
    m, nm = experiment.run_queries(answers.get, member_qs, non_member_qs)
    assert m == ["YES", "YES", None]
    assert nm == ["NO", "YES"]


def test_gzip_bits_random_vs_repetitive() -> None:
    rng = random.Random(0)
    rand_text = "".join(rng.choice(string.ascii_letters) for _ in range(10000))
    assert experiment.gzip_bits(rand_text) > 8 * 7000
    assert experiment.gzip_bits("a" * 10000) < 8 * 100


def test_analytic_curves() -> None:
    assert abs(experiment.bloom_fpr(10) - 0.6185**10) / 0.6185**10 < 0.01
    assert experiment.lower_bound_fpr(10) == 2**-10
    for b in (2, 4, 8, 16, 32):
        # The Bloom frontier lies above the information-theoretic bound.
        assert experiment.lower_bound_fpr(b) < experiment.bloom_fpr(b)
