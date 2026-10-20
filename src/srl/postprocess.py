"""Post-processing of SRL model output: BIO labels per token -> argument spans per predicate."""
import re
from collections import defaultdict
from functools import lru_cache

from .constants import BEGINNING_TAG, INSIDE_TAG, NULL_SRL_TAG
from .predict import SRLModel, get_predicate_indices


@lru_cache
def remove_bio_prefix(label: str) -> str:
    """
    Remove BIO prefix from semantic label indicating beginning vs. inside/intermediate token,
    e.g. I-A0 -> A0, B-AM-LOC -> AM-LOC
    """
    return re.sub(rf"(^[{BEGINNING_TAG}{INSIDE_TAG}]-)", "", label)


def postproc_srl(tokens: list[dict],
                 sentences: dict,
                 srl_model: SRLModel,
                 ) -> dict[int, list[dict]]:
    """Returns a nested dictionary of token indices with their predicted semantic role labels per predicate.

    tokens: serialized token dicts for the whole document (keys "text", "pos", "tag", and "dep", see
        `get_predicate_indices`, e.g. from `token_dicts_from_spacy`), indexed document-wide.
    sentences: dict mapping sentence index to [start_token, end_token] absolute indices.
    """
    srls = {}
    for sent_i in sorted(sentences.keys(), key=int):
        sent_start, sent_end = sentences[sent_i]
        sentence_tokens = tokens[sent_start:sent_end]
        sentence_token_texts = [t["text"] for t in sentence_tokens]
        pred_indices = get_predicate_indices(sentence_tokens, doc_offset=sent_start)
        srl_labels = srl_model.annotate_srl(
            token_texts=sentence_token_texts,
            pred_indices=pred_indices,
        )
        for pred_idx, labels in srl_labels.items():
            pred_abs_idx = pred_idx + sent_start
            srls[pred_abs_idx] = {
                "text": sentence_token_texts[pred_idx],
                "args": []
            }
            current_span = None  # holds (label, start_idx, tokens)
            for tkn_idx, (tkn_text, label) in enumerate(labels):
                abs_idx = tkn_idx + sent_start

                if label == NULL_SRL_TAG or abs_idx == pred_abs_idx:
                    if current_span:
                        # close span
                        srls[pred_abs_idx]["args"].append(current_span)
                        current_span = None
                    continue

                prefix, role = label.split("-", 1)

                if prefix == BEGINNING_TAG:
                    # close previous span
                    if current_span:
                        srls[pred_abs_idx]["args"].append(current_span)

                    current_span = {
                        "span": [abs_idx, abs_idx],
                        "text": tkn_text,
                        "label": role
                    }

                elif prefix == INSIDE_TAG and current_span and current_span["label"] == role:
                    # extend span
                    current_span["span"][1] = abs_idx
                    current_span["text"] += f" {tkn_text}"

                else:
                    # edge case: I without matching B → treat as B
                    if current_span:
                        srls[pred_abs_idx]["args"].append(current_span)

                    current_span = {
                        "span": [abs_idx, abs_idx],
                        "text": tkn_text,
                        "label": role
                    }

            # flush last span
            if current_span:
                srls[pred_abs_idx]["args"].append(current_span)

    return srls


def combine_postproc_srls(srls: dict, remove_label_prefix: bool = True) -> dict:
    """Returns a dictionary of token indices with their predicted semantic role labels combined over all relevant predicates in a text."""
    srls_combined = defaultdict(lambda: [])
    for pred_idx, pred_data in srls.items():
        pred_text = pred_data["text"]
        for arg in pred_data["args"]:
            start, end = arg["span"]
            label = arg["label"]
            if remove_label_prefix:
                label = remove_bio_prefix(label)
            for tkn_idx in range(start, end + 1):
                srls_combined[tkn_idx].append({
                    "pred_idx": pred_idx,
                    "predicate": pred_text,
                    "srl": label,
                })

    return srls_combined
