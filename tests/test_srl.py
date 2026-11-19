import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from srl import (BEGINNING_TAG, END_PREDICATE_TOKEN, NULL_SRL_TAG,
                 START_PREDICATE_TOKEN, SRLModel, get_predicate_indices,
                 mark_predicate, remove_bio_prefix)
from srl.constants import (DEFAULT_BASE_MODEL, DEFAULT_OUTDIR_ROOT,
                           DEFAULT_SPACY_MODEL, MODEL_DIR_ENV_VAR)
from srl.predict import load_spacy_model
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

    train.main()  # no config: output directory follows the default base model
    assert calls[-1]["model_outdir"] == str(Path(DEFAULT_OUTDIR_ROOT) / DEFAULT_BASE_MODEL)

    config.write_text("base_model: xlm-roberta-base\n")  # config without output directory
    train.main(str(config))
    assert calls[-1]["base_model"] == "xlm-roberta-base"
    assert calls[-1]["model_outdir"] == str(Path(DEFAULT_OUTDIR_ROOT) / "xlm-roberta-base")
    assert calls[-1]["train_kwargs"]["outdir"] == calls[-1]["model_outdir"]


def _import_train():
    for module in ("datasets", "seqeval", "matplotlib", "yaml"):
        pytest.importorskip(module)
    from srl import train
    return train


def test_failed_base_model_load_leaves_no_output_dir(tmp_path, monkeypatch):
    train = _import_train()

    def fail_to_load(name, *args, **kwargs):
        raise OSError(f"{name} is not a local folder and is not a valid model identifier")

    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", fail_to_load)
    outdir = tmp_path / "out" / "misspelled-model"
    with pytest.raises(OSError):
        train.SemanticRoleLabeler(
            train_conllup="unused",
            validation_conllup="unused",
            test_conllup="unused",
            model_outdir=str(outdir),
            base_model="misspelled-model",
        )
    assert not outdir.exists()


def test_existing_output_dir_fails_before_loading_base_model(tmp_path, monkeypatch):
    train = _import_train()

    def must_not_load(self):
        raise AssertionError("base model tokenizer loaded although the output directory exists")

    monkeypatch.setattr(train.SemanticRoleLabeler, "init_tokenizer", must_not_load)
    with pytest.raises(FileExistsError):
        train.SemanticRoleLabeler(
            train_conllup="unused",
            validation_conllup="unused",
            test_conllup="unused",
            model_outdir=str(tmp_path),
        )


def test_load_spacy_model_without_spacy_exits_with_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)  # makes `import spacy` raise ImportError
    with pytest.raises(SystemExit, match="spaCy is not installed.*uv sync"):
        load_spacy_model()


@pytest.mark.parametrize("name, message", [
    pytest.param(DEFAULT_SPACY_MODEL, f"model {DEFAULT_SPACY_MODEL} is not installed.*uv sync", id="default_model"),
    pytest.param("my_model", "model my_model could not be loaded: .*E050", id="other_model"),
])
def test_load_spacy_model_missing_model_exits_with_instructions(monkeypatch, name, message):
    def load(model_name):
        raise OSError(f"[E050] Can't find model '{model_name}'.")

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=load))
    with pytest.raises(SystemExit, match=message):
        load_spacy_model(name)
