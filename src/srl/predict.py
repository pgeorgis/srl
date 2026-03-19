import spacy
from constants import DEFAULT_SPACY_MODEL
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

nlp = spacy.load(DEFAULT_SPACY_MODEL)
model_path = "./models/srl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

def predict(sentence_tokens):
    encoding = tokenizer(
        sentence_tokens,
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

        token = sentence_tokens[word_id]
        label = model.config.id2label[preds[i].item()]

        results.append((token, label))

    return results

test_sentence = """Ich glaube, dass er das Buch gelesen hat."""
doc = nlp(test_sentence)
tokens = [tk.text for tk in doc]

prediction = predict(tokens)

print('done')