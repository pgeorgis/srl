import os

import pytest

from srl import SRLModel, remove_bio_prefix
from srl.constants import BEGINNING_TAG, MODEL_DIR_ENV_VAR, NULL_SRL_TAG
from tests import (GOLD_EXAMPLES, assert_output_labels_match_gold,
                   assert_output_tokens_match_input, init_fake_srl_model,
                   with_blank_tokens)

HUND_JAGT_KATZE_EXAMPLE = GOLD_EXAMPLES["hund_jagt_katze"]


@pytest.mark.parametrize("example", GOLD_EXAMPLES.values(), ids=GOLD_EXAMPLES.keys())
def test_one_label_per_token_without_blank_tokens(example):
    for pred_idx in example.pred_indices:
        labels = example.labels(pred_idx)
        out = init_fake_srl_model(labels, pred_idx).predict_srl(example.token_texts, pred_idx)
        assert_output_tokens_match_input(example.token_texts, out)
        assert_output_labels_match_gold(labels, out)


@pytest.mark.parametrize("blanks", [
    pytest.param({2: "\n"}, id="blank_before_predicate"),
    pytest.param({0: "\n", 3: " ", 4: "\n"}, id="several_blanks_incl_sentence_start"),
    pytest.param({4: "\n"}, id="blank_after_predicate"),
    pytest.param({6: "\n"}, id="blank_at_sentence_end"),
])
def test_blank_tokens_get_outside_label_and_keep_later_labels_aligned(blanks):
    tokens, pred_idx, labels = with_blank_tokens(HUND_JAGT_KATZE_EXAMPLE, HUND_JAGT_KATZE_EXAMPLE.pred_idx, blanks)
    out = init_fake_srl_model(labels, pred_idx).predict_srl(tokens, pred_index=pred_idx)
    assert_output_tokens_match_input(tokens, out)
    assert_output_labels_match_gold(labels, out)
    assert all(out[position][1] == NULL_SRL_TAG for position in blanks)


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get(MODEL_DIR_ENV_VAR), reason=f"{MODEL_DIR_ENV_VAR} not set")
def test_real_model_with_blank_token():
    example = GOLD_EXAMPLES["anna_sieht_maria"]
    tokens, pred_idx, gold = with_blank_tokens(example, example.pred_idx, {1: "\n"})
    model = SRLModel.from_pretrained()
    out = model.annotate_srl(token_texts=tokens, pred_indices=[pred_idx])[pred_idx]
    assert_output_tokens_match_input(tokens, out)
    labels = [label for _, label in out]
    assert labels[1] == NULL_SRL_TAG  # newline token
    assert labels[pred_idx] == f"{BEGINNING_TAG}-V"
    # roles of the arguments (agent, patient), regardless of B/I prefix
    for position, gold_label in enumerate(gold):
        if gold_label != NULL_SRL_TAG:
            assert remove_bio_prefix(labels[position]) == remove_bio_prefix(gold_label)
