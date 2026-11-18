import pytest

from srl import combine_postproc_srls, postproc_srl, remove_bio_prefix
from tests import GOLD_EXAMPLES, GoldExample

HUND_JAGT_KATZE_EXAMPLE = GOLD_EXAMPLES["hund_jagt_katze"]


class StubModel:
    """Returns fixed BIO labels per sentence (keyed by its token texts) for each requested predicate."""
    def __init__(self, labels_by_sentence: dict[tuple[str, ...], list[str]]):
        self.labels_by_sentence = labels_by_sentence
        self.calls = []

    @classmethod
    def from_examples(cls, *examples: GoldExample) -> "StubModel":
        return cls({tuple(ex.token_texts): ex.labels(ex.pred_idx) for ex in examples})

    def annotate_srl(self, token_texts, pred_indices, pretty_print=False):
        self.calls.append((token_texts, pred_indices))
        labels = self.labels_by_sentence[tuple(token_texts)]
        return {idx: list(zip(token_texts, labels)) for idx in pred_indices}


def test_remove_bio_prefix():
    assert remove_bio_prefix("B-A0") == "A0"
    assert remove_bio_prefix("I-AM-LOC") == "AM-LOC"
    assert remove_bio_prefix("B-R-A1") == "R-A1"
    assert remove_bio_prefix("O") == "O"


@pytest.mark.parametrize("example", GOLD_EXAMPLES.values(), ids=GOLD_EXAMPLES.keys())
def test_postproc_srl_builds_argument_spans(example):
    srls = postproc_srl(example.tokens, example.single_sentence(), StubModel.from_examples(example))
    assert srls == {
        pred_idx: {
            "text": example.token_texts[pred_idx],
            "args": example.args[pred_idx],
        }
        for pred_idx in example.pred_indices
    }


def test_postproc_srl_uses_document_wide_indices_for_later_sentences():
    second_sentence = GOLD_EXAMPLES["er_schlaeft"]
    offset = len(HUND_JAGT_KATZE_EXAMPLE.tokens)
    # head indices are document-wide
    second_tokens = [
        {**t, "dep": {**t["dep"], "head": t["dep"]["head"] + offset}} if "head" in t["dep"] else t
        for t in second_sentence.tokens
    ]
    sentences = {"0": [0, offset], "1": [offset, offset + len(second_sentence.tokens)]}
    srls = postproc_srl(HUND_JAGT_KATZE_EXAMPLE.tokens + second_tokens, sentences, StubModel.from_examples(HUND_JAGT_KATZE_EXAMPLE, second_sentence))
    expected_args = [{**arg, "span": [i + offset for i in arg["span"]]} for arg in second_sentence.args[second_sentence.pred_idx]]
    assert srls[second_sentence.pred_idx + offset]["args"] == expected_args


def test_postproc_srl_sentence_initial_inside_tag_starts_span():
    example = HUND_JAGT_KATZE_EXAMPLE
    labels = list(example.labels(example.pred_idx))
    assert labels[0] == "B-A0"
    labels[0] = "I-A0"  # "Der": I- tag without a preceding B- tag
    model = StubModel({tuple(example.token_texts): labels})
    srls = postproc_srl(
        tokens=example.tokens,
        sentences=example.single_sentence(),
        srl_model=model,
    )
    # treated like the gold B- tag: same spans as the gold annotation
    assert srls[example.pred_idx]["args"] == example.args[example.pred_idx]


def test_postproc_srl_inside_tag_of_different_role_starts_new_span():
    example = HUND_JAGT_KATZE_EXAMPLE
    labels = list(example.labels(example.pred_idx))
    assert labels[:2] == ["B-A0", "I-A0"]
    labels[1] = "I-A1"  # "Hund": I- tag of a different role than the open A0 span
    model = StubModel({tuple(example.token_texts): labels})
    srls = postproc_srl(example.tokens, example.single_sentence(), model)
    gold_a0, gold_a1 = example.args[example.pred_idx]
    # the gold A0 span "Der Hund" is split into A0 "Der" and a new A1 span "Hund"; the gold A1 span is unaffected.
    # These two spans are this malformed input's expected output, which is not part of the gold annotation.
    assert srls[example.pred_idx]["args"] == [
        {"span": [0, 0], "text": "Der", "label": "A0"},
        {"span": [1, 1], "text": "Hund", "label": "A1"},
        gold_a1,
    ]


@pytest.mark.parametrize("example", GOLD_EXAMPLES.values(), ids=GOLD_EXAMPLES.keys())
def test_combine_postproc_srls(example):
    srls = postproc_srl(
        tokens=example.tokens,
        sentences=example.single_sentence(),
        srl_model=StubModel.from_examples(example)
    )
    combined = combine_postproc_srls(srls)
    expected = {}
    for arg in example.args[example.pred_idx]:
        start, end = arg["span"]
        for tkn_idx in range(start, end + 1):
            expected[tkn_idx] = [{
                "pred_idx": example.pred_idx,
                "predicate": example.predicate,
                "srl": arg["label"]
            }]
    # tokens outside all arguments, including the predicate itself, are left out
    assert combined == expected
