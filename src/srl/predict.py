import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from constants import DEFAULT_SPACY_MODEL

########################################
# CONFIG
########################################
MODEL_PATH = "./models/srl"

########################################
# LOAD MODELS
########################################

nlp = spacy.load(DEFAULT_SPACY_MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

model.eval()

########################################
# HELPER FUNCTIONS
########################################

def get_predicate_indices(doc):
    """Return indices of verbs/auxiliaries"""
    return [i for i, tok in enumerate(doc) if tok.pos_ in {"VERB", "AUX"}]


def mark_predicate(tokens, pred_index):
    """Insert [PRED] and [/PRED] around predicate"""
    marked = []
    for i, tok in enumerate(tokens):
        if i == pred_index:
            marked.extend(["[PRED]", tok, "[/PRED]"])
        else:
            marked.append(tok)
    return marked


def predict(tokens, pred_index):
    """Run SRL for one predicate"""
    
    marked_tokens = mark_predicate(tokens, pred_index)

    # DEBUG (optional)
    # print("INPUT:", " ".join(marked_tokens))

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
        if token in ["[PRED]", "[/PRED]"]:
            continue

        results.append((token, label))

    return results


def pretty_print(predicate, prediction):
    """Nicely print SRL output"""
    print(f"\n=== Predicate: {predicate} ===")
    for tok, label in prediction:
        print(f"{tok:20} {label}")


########################################
# MAIN TEST
########################################

test_sentence = """Ich denke, dass Sophie das Buch schon gelesen hat, das von ihrem Bruder veröffentlicht wurde."""

doc = nlp(test_sentence)
tokens = [tk.text for tk in doc]

pred_indices = get_predicate_indices(doc)

print("Sentence:")
print(test_sentence)

for pred_idx in pred_indices:
    predicate = tokens[pred_idx]
    prediction = predict(tokens, pred_idx)
    pretty_print(predicate, prediction)
