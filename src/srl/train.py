from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from constants import END_PREDICATE_TOKEN, START_PREDICATE_TOKEN

train_datafile = '/Users/pgeorgis/Documents/tools/Universal Dependencies 2.9/ud_up_test/de_gsd-ud-up-train.conllup'
dev_datafile = '/Users/pgeorgis/Documents/tools/Universal Dependencies 2.9/ud_up_test/de_gsd-ud-up-dev.conllup'
test_datafile = '/Users/pgeorgis/Documents/tools/Universal Dependencies 2.9/ud_up_test/de_gsd-ud-up-test.conllup'
model_outdir = '/Users/pgeorgis/Documents/projects/c07-dpron-korpus/models/srl'

########################################
# 1. Parse CONLLUP
########################################

def read_conllup(path):
    sentences = []
    current = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")

            # skip multiword tokens like "6-7 im"
            if "-" in cols[0]:
                continue

            token = cols[1]
            pred = cols[8]
            argspans = cols[10]

            current.append({
                "token": token,
                "pred": pred,
                "argspans": argspans
            })

    return sentences


########################################
# 2. Convert to SRL examples
########################################

def extract_examples(sentences):
    examples = []

    for sent in sentences:
        tokens = [w["token"] for w in sent]

        # find predicates
        for i, w in enumerate(sent):
            if w["pred"] != "_":
                predicate_index = i
                argspans = w["argspans"]

                labels = ["O"] * len(tokens)

                # mark predicate
                labels[predicate_index] = "B-V"

                if argspans != "_":
                    spans = argspans.split("|")

                    for span in spans:
                        role, positions = span.split(":")
                        start, end = map(int, positions.split("-"))

                        # convert 1-based → 0-based
                        start -= 1
                        end -= 1

                        labels[start] = f"B-{role}"
                        for j in range(start + 1, end + 1):
                            labels[j] = f"I-{role}"

                # insert predicate markers
                marked_tokens = []
                marked_labels = []

                for idx, token in enumerate(tokens):
                    if idx == predicate_index:
                        marked_tokens.append(START_PREDICATE_TOKEN)
                        marked_labels.append("O")

                        marked_tokens.append(token)
                        marked_labels.append(labels[idx])

                        marked_tokens.append(END_PREDICATE_TOKEN)
                        marked_labels.append("O")
                    else:
                        marked_tokens.append(token)
                        marked_labels.append(labels[idx])

                examples.append({
                    "tokens": marked_tokens,
                    "labels": marked_labels
                })

    return examples


########################################
# 3. Build label set
########################################

def build_label_list(examples):
    label_set = set()
    for ex in examples:
        label_set.update(ex["labels"])

    label_list = sorted(label_set)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    return label_list, label2id, id2label


########################################
# 4. Tokenization + alignment
########################################

def tokenize_and_align(example, tokenizer, label2id):
    encoding = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    word_ids = encoding.word_ids()

    labels = []
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        else:
            labels.append(label2id[example["labels"][word_id]])

    encoding["labels"] = labels
    return encoding


########################################
# 5. Load data
########################################

train_sentences = read_conllup(train_datafile)
dev_sentences = read_conllup(dev_datafile)
test_sentences = read_conllup(test_datafile)

train_examples = extract_examples(train_sentences)
dev_examples = extract_examples(dev_sentences)
test_examples = extract_examples(test_sentences)

label_list, label2id, id2label = build_label_list(train_examples)

print("Labels:", label_list)

########################################
# 6. Tokenizer
########################################

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
special_tokens = {"additional_special_tokens": [START_PREDICATE_TOKEN, END_PREDICATE_TOKEN]}
tokenizer.add_special_tokens(special_tokens)

train_dataset = Dataset.from_list(train_examples)
dev_dataset = Dataset.from_list(dev_examples)
test_dataset = Dataset.from_list(test_examples)

train_dataset = train_dataset.map(
    lambda x: tokenize_and_align(x, tokenizer, label2id),
    batched=False
)

dev_dataset = dev_dataset.map(
    lambda x: tokenize_and_align(x, tokenizer, label2id),
    batched=False
)

test_dataset = test_dataset.map(
    lambda x: tokenize_and_align(x, tokenizer, label2id),
    batched=False
)

########################################
# 7. Model
########################################

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

model.resize_token_embeddings(len(tokenizer))

########################################
# 8. Training
########################################

training_args = TrainingArguments(
    output_dir=model_outdir,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    #tokenizer=tokenizer
)

trainer.train()

########################################
# 9. Save
########################################

trainer.save_model(model_outdir)
tokenizer.save_pretrained(model_outdir)

print("Training complete!")
