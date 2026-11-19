import pytest

pytest.importorskip("seqeval")

from srl.metrics import ignore_predicate, srl_metrics


def test_ignore_predicate():
    assert ignore_predicate(["B-A0", "B-V", "B-A1"]) == ["B-A0", "O", "B-A1"]


def test_predicate_is_not_scored_as_argument():
    gold = [["B-A0", "I-A0", "B-V", "B-A1", "O"]]
    # the predicate is wrong, but it is not an argument: all argument spans are right
    pred = [["B-A0", "I-A0", "O", "B-A1", "O"]]
    metrics = srl_metrics(gold, pred)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_argument_spans_must_match_exactly():
    gold = [["B-A0", "I-A0", "B-V", "B-A1", "O"]]
    pred = [["B-A0", "O", "B-V", "B-A1", "O"]]  # A0 span too short: wrong; A1 right
    metrics = srl_metrics(gold, pred)
    assert metrics["precision"] == metrics["recall"] == 0.5
    assert metrics["accuracy"] == pytest.approx(4 / 5)


def test_word_level_counts_each_argument_once():
    gold = [["B-A0", "I-A0", "B-V", "B-AM-DIS", "B-A1", "O"]]
    metrics = srl_metrics(gold, gold, per_role=True)
    assert metrics["f1"] == 1.0
    assert {role: scores["support"] for role, scores in metrics["roles"].items()} == {"A0": 1, "AM-DIS": 1, "A1": 1}


def test_empty_predictions_score_zero():
    gold = [["B-A0", "B-V", "B-A1"]]
    pred = [["O", "B-V", "O"]]
    metrics = srl_metrics(gold, pred)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0
