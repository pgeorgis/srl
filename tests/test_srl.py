import os

import pytest

from srl import (BEGINNING_TAG, END_PREDICATE_TOKEN, NULL_SRL_TAG,
                 START_PREDICATE_TOKEN, SRLModel, get_predicate_indices,
                 mark_predicate, remove_bio_prefix)
from srl.constants import MODEL_DIR_ENV_VAR
from tests import GOLD_EXAMPLES, serialize_token


def test_mark_predicate():
    marked = mark_predicate(tokens=["Er", "schläft", "."], pred_index=1)
    assert marked == ["Er", START_PREDICATE_TOKEN, "schläft", END_PREDICATE_TOKEN, "."]


@pytest.mark.parametrize("example", GOLD_EXAMPLES.values(), ids=GOLD_EXAMPLES.keys())
def test_predicate_indices_match_gold(example):
    # e.g. "Er hat das Buch gelesen": only the participle is a predicate, not its auxiliary head <hat>
    assert get_predicate_indices(example.tokens) == example.pred_indices


def test_predicate_indices_respects_document_offset():
    # Head indices are document-wide; the sentence starts at token 10
    tokens = [
        serialize_token(text="hat", pos="AUX", tag="VAFIN", dep="ROOT"),
        serialize_token(text="gelesen", pos="VERB", tag="VVPP", head=10),
    ]
    assert get_predicate_indices(tokens, doc_offset=10) == [1]


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get(MODEL_DIR_ENV_VAR), reason=f"{MODEL_DIR_ENV_VAR} not set")
def test_annotate_srl_with_trained_model():
    example = GOLD_EXAMPLES["hund_jagt_katze"]
    pred_idx = example.pred_idx
    model = SRLModel.from_pretrained()
    out = model.annotate_srl(token_texts=example.token_texts, pred_indices=[pred_idx])[pred_idx]
    assert [tk for tk, _ in out] == example.token_texts
    labels = [label for _, label in out]
    assert labels[pred_idx] == f"{BEGINNING_TAG}-V"
    # roles of the argument heads, regardless of B/I prefix
    for position, token in enumerate(example.tokens):
        if token["dep"].get("head") == pred_idx and example.labels(pred_idx)[position] != NULL_SRL_TAG:
            assert remove_bio_prefix(labels[position]) == remove_bio_prefix(example.labels(pred_idx)[position])


def test_training_output_dir_from_config_and_cli(tmp_path, monkeypatch):
    for module in ("datasets", "seqeval", "matplotlib", "yaml"):
        pytest.importorskip(module)
    from srl import train

    calls = []

    class FakeLabeler:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def train(self, **kwargs):
            calls[-1]["train_kwargs"] = kwargs

    monkeypatch.setattr(train, "SemanticRoleLabeler", FakeLabeler)
    config = tmp_path / "config.yml"
    config.write_text("model_outdir: ./out/from_config\ntrain:\n  num_train_epochs: 1\n")

    train.main(str(config))
    assert calls[-1]["model_outdir"] == "./out/from_config"
    assert calls[-1]["train_kwargs"]["outdir"] == "./out/from_config"

    train.main(str(config), output_dir="cli/override")  # command line overrides config
    assert calls[-1]["model_outdir"] == "cli/override"

    train.main()  # no config: defaults
    assert calls[-1]["model_outdir"] == train.DEFAULT_SRL_MODEL_DIR
