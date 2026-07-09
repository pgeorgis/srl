import argparse
import logging
from dataclasses import dataclass

import spacy
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)

from constants.models import DEFAULT_SPACY_MODEL, DEFAULT_SRL_MODEL_DIR
from constants.srl import END_PREDICATE_TOKEN, START_PREDICATE_TOKEN
from constants.tags import (PAST_PARTICIPLE_AUX_TAG, PAST_PARTICIPLE_TAG,
                            VERB_TAGS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SRLModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def predict_srl(self, tokens: list, pred_index: int) -> list:
        """Run SRL for one predicate"""
        model = self.model
        tokenizer = self.tokenizer
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

    def annotate_srl(self,
                     token_texts: list,
                     pred_indices: list,
                     pretty_print: bool = False,
                     ) -> dict:
        srl_predictions = {}
        for pred_idx in pred_indices:
            predicate = token_texts[pred_idx]
            prediction = self.predict_srl(token_texts, pred_idx)
            srl_predictions[pred_idx] = prediction
            # Optionally pretty-print results
            if pretty_print:
                pretty_print_srl(predicate, prediction)
        return srl_predictions


def load_pretrained_srl_model(model_dir: str) -> SRLModel:
    logger.info(f"Loading SRL model from {model_dir} ...")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return SRLModel(
        model=model,
        tokenizer=tokenizer,
    )


def get_predicate_indices(tokens: list, doc_offset: int = 0) -> list:
    """Return sentence-relative predicate indices from a sentence-level list of serialized token dicts.

    doc_offset: absolute token index of the first token in this sentence within the document,
    used to correctly map document-level head indices back to sentence-relative indices.
    """
    verbs_and_aux = [idx for idx, tok in enumerate(tokens) if tok["pos"] in VERB_TAGS]
    predicate_idx = []
    heads_to_skip = set()
    for idx in verbs_and_aux:
        token = tokens[idx]
        if token.get("tag") in {
            PAST_PARTICIPLE_TAG,     # (VVPP) past participle, e.g. <hat ... gesehen>
            PAST_PARTICIPLE_AUX_TAG, # (VAPP) auxiliary past participle, e.g. <ist ... gewesen>
            'VMPP',                  # modal past participle, e.g. <hat ... gekonnt>
            'VVINF',                 # full infinitive, e.g. <würde ... sehen> 
            'VMINF',                 # modal infinitive, e.g. <hätte ... sehen können>
        } and "head" in token.get("dep", {}):
            head_sent_idx = token["dep"]["head"] - doc_offset
            if 0 <= head_sent_idx < len(tokens):
                heads_to_skip.add(head_sent_idx)
        predicate_idx.append(idx)
    return [idx for idx in predicate_idx if idx not in heads_to_skip]


def mark_predicate(tokens, pred_index):
    """Insert [PRED] and [/PRED] around predicate"""
    marked = []
    for i, tok in enumerate(tokens):
        if i == pred_index:
            marked.extend([START_PREDICATE_TOKEN, tok, END_PREDICATE_TOKEN])
        else:
            marked.append(tok)
    return marked


def pretty_print_srl(predicate, srl_prediction):
    """Nicely print SRL output"""
    print(f"\n=== Predicate: {predicate} ===")
    for tok, label in srl_prediction:
        print(f"{tok:20} {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loads a pretrained semantic role labeling model and performs prediction on a given text.")
    parser.add_argument('--text', help='Input text to label with semantic roles')
    parser.add_argument('--srl_model', default=DEFAULT_SRL_MODEL_DIR, help='Path to pretrained semantic role labeling model')
    parser.add_argument('--spacy_model', default=DEFAULT_SPACY_MODEL, help='Name of spaCy model for preprocessing text (NB: must have been pre-downloaded)')
    args = parser.parse_args()

    # Load models
    nlp_preproc = spacy.load(args.spacy_model)
    srl_model = load_pretrained_srl_model(args.srl_model)

    # Annotate and pretty-print semantic role label predictions
    print(f"Text: {args.text}")
    doc = nlp_preproc(args.text)
    token_dicts = [
        {
            "text": t.text,
            "pos": t.pos_,
            "tag": t.tag_,
            "dep": {
                "tag": t.dep_,
                "head": t.head.i
            } if t.has_head() else {"tag": t.dep_}
        }
        for t in doc
    ]
    token_texts = [t["text"] for t in token_dicts]
    pred_indices = get_predicate_indices(token_dicts, doc_offset=0)
    srl_model.annotate_srl(token_texts=token_texts, pred_indices=pred_indices, pretty_print=True)
