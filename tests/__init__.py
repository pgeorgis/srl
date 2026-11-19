from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from srl import SRLModel, mark_predicate
from srl.constants import (BEGINNING_TAG, CORE_SR_LABELS, END_PREDICATE_TOKEN,
                           NULL_SRL_TAG, START_PREDICATE_TOKEN)


def serialize_token(text: str, pos: str, tag: str, head=None, dep="X") -> dict:
    return {
        "text": text,
        "pos": pos,
        "tag": tag,
        "dep": {"tag": dep, "head": head} if head is not None else {"tag": dep},
    }


@dataclass(frozen=True)
class GoldExample:
    """A gold-annotated sentence.

    tokens: serialized token dicts (output format of `serialize_token`); heads are sentence-internal indices.
    srl: predicate index -> one BIO label per token.
    args: predicate index -> expected argument spans as returned by `postproc_srl` (sentence starting at 0).
    """
    text: str
    tokens: list[dict]
    srl: dict[int, list[str]]
    args: dict[int, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        pred_tag = f"{BEGINNING_TAG}-V"
        for pred_idx, labels in self.srl.items():
            assert len(labels) == len(self.tokens), f"{self.text!r}: wrong number of labels for predicate {pred_idx}"
            assert labels[pred_idx] == pred_tag, f"{self.text!r}: predicate {pred_idx} is not labeled {pred_tag}"

    @property
    def token_texts(self) -> list[str]:
        return [t["text"] for t in self.tokens]

    @property
    def pred_indices(self) -> list[int]:
        return sorted(self.srl)

    @property
    def pred_idx(self) -> int:
        """Index of the predicate of a single-predicate example."""
        if len(self.srl) != 1:
            raise ValueError(f"{self.text!r} has {len(self.srl)} predicates {self.pred_indices}; use `pred_indices`")
        (pred_idx,) = self.srl
        return pred_idx

    @property
    def predicate(self) -> str:
        """Text of the predicate of a single-predicate example."""
        return self.token_texts[self.pred_idx]

    def labels(self, pred_idx: int) -> list[str]:
        return self.srl[pred_idx]

    def single_sentence(self) -> dict[str, list[int]]:
        """The example as the only sentence of a document, in the `sentences` format of `postproc_srl`."""
        return {"0": [0, len(self.tokens)]}


GOLD_EXAMPLES = {
    "hund_jagt_katze": GoldExample(
        text="Der Hund jagt die Katze.",
        tokens=[
            serialize_token("Der", "DET", "ART", 1),
            serialize_token("Hund", "NOUN", "NN", 2),
            serialize_token("jagt", "VERB", "VVFIN", dep="ROOT"),
            serialize_token("die", "DET", "ART", 4),
            serialize_token("Katze", "NOUN", "NN", 2),
            serialize_token(".", "PUNCT", "$.", 2),
        ],
        srl={2: ["B-A0", "I-A0", "B-V", "B-A1", "I-A1", "O"]},
        args={2: [
            {"span": [0, 1], "text": "Der Hund", "label": "A0"},
            {"span": [3, 4], "text": "die Katze", "label": "A1"},
        ]},
    ),
    "er_hat_buch_gelesen": GoldExample(
        # only the participle is a predicate, not its auxiliary head <hat>
        text="Er hat das Buch gelesen",
        tokens=[
            serialize_token("Er", "PRON", "PPER", 1),
            serialize_token("hat", "AUX", "VAFIN", dep="ROOT"),
            serialize_token("das", "DET", "ART", 3),
            serialize_token("Buch", "NOUN", "NN", 4),
            serialize_token("gelesen", "VERB", "VVPP", 1),
        ],
        srl={4: ["B-A0", "O", "B-A1", "I-A1", "B-V"]},
        args={4: [
            {"span": [0, 0], "text": "Er", "label": "A0"},
            {"span": [2, 3], "text": "das Buch", "label": "A1"},
        ]},
    ),
    "er_schlaeft": GoldExample(
        text="Er schläft",
        tokens=[
            serialize_token("Er", "PRON", "PPER", 1),
            serialize_token("schläft", "VERB", "VVFIN", dep="ROOT"),
        ],
        srl={1: ["B-A0", "B-V"]},
        args={1: [{"span": [0, 0], "text": "Er", "label": "A0"}]},
    ),
    "anna_sieht_maria": GoldExample(
        text="Anna sieht Maria.",
        tokens=[
            serialize_token("Anna", "PROPN", "NE", 1),
            serialize_token("sieht", "VERB", "VVFIN", dep="ROOT"),
            serialize_token("Maria", "PROPN", "NE", 1),
            serialize_token(".", "PUNCT", "$.", 1),
        ],
        srl={1: ["B-A0", "B-V", "B-A1", "O"]},
        args={1: [
            {"span": [0, 0], "text": "Anna", "label": "A0"},
            {"span": [2, 2], "text": "Maria", "label": "A1"},
        ]},
    ),
    "er_muss_buch_haben": GoldExample(
        # modal with an auxiliary infinitive (VAINF): only the infinitive <haben> is a predicate, not <muss>
        text="Er muss das Buch haben",
        tokens=[
            serialize_token("Er", "PRON", "PPER", 1),
            serialize_token("muss", "AUX", "VMFIN", dep="ROOT"),
            serialize_token("das", "DET", "ART", 3),
            serialize_token("Buch", "NOUN", "NN", 4),
            serialize_token("haben", "AUX", "VAINF", 1),
        ],
        srl={4: ["B-A0", "B-AM-MOD", "B-A1", "I-A1", "B-V"]},
        args={4: [
            {"span": [0, 0], "text": "Er", "label": "A0"},
            {"span": [1, 1], "text": "muss", "label": "AM-MOD"},
            {"span": [2, 3], "text": "das Buch", "label": "A1"},
        ]},
    ),
}


def with_blank_tokens(example: GoldExample, pred_idx: int, blanks: dict[int, str]) -> tuple[list[str], int, list[str]]:
    """Inserts blank tokens into an example's token texts.

    blanks: position in the resulting token list -> blank token text (e.g. "\\n").
    Returns the token texts, the shifted predicate index, and the gold labels (outside label for blanks).
    """
    tokens, labels = list(example.token_texts), list(example.labels(pred_idx))
    for position in sorted(blanks):
        tokens.insert(position, blanks[position])
        labels.insert(position, NULL_SRL_TAG)
        if position <= pred_idx:
            pred_idx += 1
    return tokens, pred_idx, labels


class FakeEncoding(dict):
    def __init__(self, word_ids):
        super().__init__(input_ids=torch.zeros(1, len(word_ids), dtype=torch.long))
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __init__(self, max_words=None):
        self.max_words = max_words
        self.word_ids = None

    def __call__(self, words, **kwargs):
        word_ids = [None]
        for i, word in enumerate(words):
            if self.max_words is not None and i >= self.max_words:
                break
            # every non-empty/whitespace word is split into two sub-word tokens
            if word.strip():
                word_ids += [i, i]
        word_ids.append(None)
        self.word_ids = word_ids
        return FakeEncoding(word_ids)


class FakeModel:
    """Predicts the given label for each word of the predicate-marked input (the outside label for the markers)."""
    def __init__(self, marked_labels, tokenizer):
        self.marked_labels = marked_labels
        self.tokenizer = tokenizer
        # core labels, plus any other labels to predict (e.g. B-AM-MOD), but not the predicate markers
        markers = {START_PREDICATE_TOKEN, END_PREDICATE_TOKEN}
        self.label_set = list(dict.fromkeys(CORE_SR_LABELS + [l for l in marked_labels if l not in markers]))
        self.config = SimpleNamespace(id2label=dict(enumerate(self.label_set)))

    def eval(self):
        pass

    def __call__(self, input_ids, **kwargs):
        logits = torch.zeros(1, input_ids.shape[1], len(self.label_set))
        for position, word_id in enumerate(self.tokenizer.word_ids):
            label = NULL_SRL_TAG if word_id is None else self.marked_labels[word_id]
            if label not in self.label_set:  # predicate markers
                label = NULL_SRL_TAG
            logits[0, position, self.label_set.index(label)] = 1.0
        return SimpleNamespace(logits=logits)


def init_fake_srl_model(labels: list[str], pred_idx: int, max_words=None) -> SRLModel:
    """A model that predicts `labels` (one per input token) for the predicate at `pred_idx`."""
    tokenizer = FakeTokenizer(max_words)
    model = FakeModel(mark_predicate(labels, pred_idx), tokenizer)
    return SRLModel(model=model, tokenizer=tokenizer)


def assert_output_tokens_match_input(input_tokens: list[str], out: list[tuple[str, str]]):
    assert [token for token, _ in out] == input_tokens


def assert_output_labels_match_gold(gold_labels: list[str], out: list[tuple[str, str]]):
    assert [label for _, label in out] == gold_labels
