import argparse

import spacy
import torch
from spacy.tokens import Doc
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)

from constants import (DEFAULT_SPACY_MODEL, DEFAULT_SRL_MODEL_DIR,
                       END_PREDICATE_TOKEN, START_PREDICATE_TOKEN, VERB_TAGS)


def load_pretrained_srl_model(model_dir: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def get_predicate_indices(doc):
    """Return indices of verbs/auxiliaries"""
    verbs_and_aux = [idx for idx, tok in enumerate(doc) if tok.pos_ in VERB_TAGS]
    predicate_idx = []
    heads_to_skip = set()
    for idx in verbs_and_aux:
        token = doc[idx]
        if token.tag_ in {
            'VVPP',     # past participle, e.g. <hat ... gesehen>
            'VAPP',     # auxiliary past participle, e.g. <ist ... gewesen>
            'VMPP',     # modal past participle, e.g. <hat ... gekonnt>
            'VVINF',    # full infinitive, e.g. <würde ... sehen> 
            'VMINF',    # modal infinitive, e.g. <hätte ... sehen können>
        } and token.has_head():
            heads_to_skip.add(token.head.i)
        predicate_idx.append(idx)
    predicate_idx = [idx for idx in predicate_idx if idx not in heads_to_skip]
    return predicate_idx


def mark_predicate(tokens, pred_index):
    """Insert [PRED] and [/PRED] around predicate"""
    marked = []
    for i, tok in enumerate(tokens):
        if i == pred_index:
            marked.extend([START_PREDICATE_TOKEN, tok, END_PREDICATE_TOKEN])
        else:
            marked.append(tok)
    return marked


def predict_srl(tokens: list,
                pred_index: int,
                model,
                tokenizer,
                ):
    """Run SRL for one predicate"""
    model.eval()  # switch model to evaluation mode
    marked_tokens = mark_predicate(tokens, pred_index)
    encoding = tokenizer(
        marked_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**encoding)

    preds = torch.argmax(outputs.logits, dim=-1)[0]
    word_ids = encoding.word_ids()

    results = []
    seen_words = set()

    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue

        seen_words.add(word_id)

        token = marked_tokens[word_id]
        label = model.config.id2label[preds[i].item()]

        # skip special tokens
        if token in [START_PREDICATE_TOKEN, END_PREDICATE_TOKEN]:
            continue

        results.append((token, label))

    return results


def pretty_print_srl(predicate, srl_prediction):
    """Nicely print SRL output"""
    print(f"\n=== Predicate: {predicate} ===")
    for tok, label in srl_prediction:
        print(f"{tok:20} {label}")


def annotate_srl(
        preproc_doc: Doc,
        srl_model: PreTrainedModel,
        srl_tokenizer: PreTrainedTokenizerBase,
        pretty_print: bool = False,
        ):
    tokens = [tk.text for tk in preproc_doc]
    # Get indices of predicates in text
    pred_indices = get_predicate_indices(preproc_doc)
    # Perform SRL prediction for each predicate
    srl_predictions = {}
    for pred_idx in pred_indices:
        predicate = tokens[pred_idx]
        prediction = predict_srl(tokens, pred_idx, srl_model, srl_tokenizer)
        srl_predictions[pred_idx] = prediction
        # Optionally pretty-print results
        if pretty_print:
            pretty_print_srl(predicate, prediction)
    return srl_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loads a pretrained semantic role labeling model and performs prediction on a given text.")
    parser.add_argument('--text', help='Input text to label with semantic roles')
    parser.add_argument('--srl_model', default=DEFAULT_SRL_MODEL_DIR, help='Path to pretrained semantic role labeling model')
    parser.add_argument('--spacy_model', default=DEFAULT_SPACY_MODEL, help='Name of spaCy model for preprocessing text (NB: must have been pre-downloaded)')
    args = parser.parse_args()

    # Load models
    nlp_preproc = spacy.load(args.spacy_model)
    model, tokenizer = load_pretrained_srl_model(args.srl_model)

    # Annotate and pretty-print semantic role label predictions
    print(f"Text: {args.text}")
    annotate_srl(
        preproc_doc=nlp_preproc(args.text),
        srl_model=model,
        srl_tokenizer=tokenizer,
        pretty_print=True,
    )
