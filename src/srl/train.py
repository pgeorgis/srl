import argparse
import contextlib
import json
import logging
import os
import warnings

import yaml
from datasets import Dataset
from seqeval.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.utils import logging as hf_logging

from constants import (DEFAULT_SRL_MODEL_DIR, END_PREDICATE_TOKEN,
                       SRL_BASE_MODEL, SRL_MODEL_DATA_DIR,
                       START_PREDICATE_TOKEN)
from utils import _LineBufferedWriter

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DATA = os.path.join(SRL_MODEL_DATA_DIR, "de_gsd-ud-up-train.conllup")
DEFAULT_VALIDATION_DATA = os.path.join(SRL_MODEL_DATA_DIR, "de_gsd-ud-up-dev.conllup")
DEFAULT_TEST_DATA = os.path.join(SRL_MODEL_DATA_DIR, "de_gsd-ud-up-test.conllup")


class SemanticRoleDataset:
    def __init__(self, data_filepath: str):
        self.filepath = os.path.abspath(data_filepath)
        self.examples = self.load_data()
        self.labels = self.get_labels()

    def load_data(self) -> list:
        logger.info(f"Loading data from {self.filepath}")
        sentences = self.read_conllup(self.filepath)
        examples = self.extract_predicate_examples(sentences)
        return examples

    def get_labels(self) -> set:
        label_set = set()
        for ex in self.examples:
            label_set.update(ex["labels"])
        return label_set

    @staticmethod
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

    @staticmethod
    def extract_predicate_examples(sentences):
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


class SemanticRoleLabeler:
    def __init__(self,
                 train_conllup: str,
                 validation_conllup: str,
                 test_conllup: str,
                 model_outdir: str = DEFAULT_SRL_MODEL_DIR,
                 base_model: str = SRL_BASE_MODEL,
                 ):
        # Set model output directory
        self.outdir = self.init_outdir(model_outdir)

        # Initialize base model and tokenizer
        self.base_model_name = base_model
        logger.info("Initializing tokenizer...")
        self.tokenizer = self.init_tokenizer()
        self.labels = set()

        # Initialize semantic role datasets from input .conllup files
        # NB: Label set is built during this step, thus should not tokenize until after
        logger.info("Initializing input data...")
        train_set  = self.init_input_data(train_conllup)
        validation_set = self.init_input_data(validation_conllup)
        test_set = self.init_input_data(test_conllup)

        # Extract semantic role label <-> ID mappings
        logger.info(f"Semantic label set: {sorted(self.labels)}")
        self.label2id, self.id2label = self.get_label_maps()

        # Preprocess SemanticRoleDatasets: tokenization and alignment
        logger.info("Tokenizing data...")
        self.train_set = self.preprocess_dataset(train_set)
        self.validation_set = self.preprocess_dataset(validation_set)
        self.test_set = self.preprocess_dataset(test_set)

        # Initialize model
        self.model = self.init_base_model()

    @staticmethod
    def init_outdir(outdir: str):
        outdir = os.path.abspath(outdir)
        try:
            os.makedirs(outdir)
        except FileExistsError as exc:
            raise FileExistsError(f"Output directory {outdir} already exists; model would be overwritten.") from exc
        return outdir

    @ staticmethod
    def configure_logging(outdir):
        training_log = os.path.join(os.path.abspath(outdir), "train.log")
        logger.info(f"Training log: {training_log}")
        root_logger = logging.getLogger()

        # Avoid adding duplicate file handlers if train() is called multiple times
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == training_log:
                return

        # Remove console handlers so output is only written to the log file
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(training_log, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

        # Route transformers logs through the root logger (and disable its default console handler)
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()

        # Capture warnings (e.g., torch warnings) into the log file
        logging.captureWarnings(True)
        warnings.simplefilter("default")
        return training_log

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        special_tokens = {"additional_special_tokens": [START_PREDICATE_TOKEN, END_PREDICATE_TOKEN]}
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer
    
    def init_input_data(self, data_filepath: str) -> SemanticRoleDataset:
        dataset = SemanticRoleDataset(data_filepath)
        self.labels.update(dataset.labels)
        return dataset

    def preprocess_dataset(self, dataset: SemanticRoleDataset) -> Dataset:
        dataset = Dataset.from_list(dataset.examples)
        dataset = dataset.map(
            lambda x: self.tokenize_and_align(x),
            batched=False
        )
        return dataset
    
    def tokenize_and_align(self, example):
        encoding = self.tokenizer(
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
                labels.append(self.label2id[example["labels"][word_id]])

        encoding["labels"] = labels
        return encoding

    def get_label_maps(self) -> tuple[dict, dict]:
        label_list = sorted(self.labels)
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for l, i in label2id.items()}
        return label2id, id2label
    
    def init_base_model(self):
        logger.info(f"Loading base model <{self.base_model_name}> ...")
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def train(self, outdir: str = None, **kwargs):
        if outdir is None:
            outdir = self.outdir
        if "disable_tqdm" not in kwargs:
            kwargs["disable_tqdm"] = True
        training_log = self.configure_logging(outdir)
        training_args = TrainingArguments(
            output_dir=outdir,
            **kwargs
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.validation_set,
            compute_metrics=self.compute_metrics,
        )
        with open(training_log, "a", encoding="utf-8", buffering=1) as logf, \
                contextlib.redirect_stdout(_LineBufferedWriter(logf)), \
                contextlib.redirect_stderr(_LineBufferedWriter(logf)):
            trainer.train()
            logger.info("Model training complete.")

            # Final evaluation on test set
            test_metrics = trainer.evaluate(self.test_set, metric_key_prefix="test")
            self.log_metrics(test_metrics, outdir=outdir)
            logger.info(f"Test metrics: {test_metrics}")

        # Save trained model
        trainer.save_model(outdir)
        self.tokenizer.save_pretrained(outdir)
        logger.info(f"Model saved to {outdir}")

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)

        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(preds, labels):
            seq_labels = []
            seq_preds = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                seq_labels.append(self.id2label[label_id])
                seq_preds.append(self.id2label[pred_id])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    def log_metrics(self, metrics: dict, outdir: str):
        logfile = os.path.join(outdir, "test_metrics.json")
        with open(logfile, "w") as f:
            json.dump(metrics, f, indent=4)


def load_config(config_path: str) -> dict:
    loaded_config = {}
    with open(config_path, 'r') as f:
        loaded_config: dict = yaml.safe_load(f)
    return loaded_config


def main(config_path: str):
    # Load run config
    config = load_config(config_path=config_path)
    logger.info(f'Loaded model config from {config_path}')

    # Extract required arguments from run config or use defaults
    model_outdir = config.get("model_outdir", DEFAULT_SRL_MODEL_DIR)
    base_model_name = config.get("base_model", SRL_BASE_MODEL)
    logger.info(f"Base model: {base_model_name}")
    train_datafile = config.get("data", {}).get("train", DEFAULT_TRAIN_DATA)
    validation_datafile = config.get("data", {}).get("validation", DEFAULT_VALIDATION_DATA)
    test_datafile = config.get("data", {}).get("test", DEFAULT_TEST_DATA)
    logger.info(f"Training data: {train_datafile}")
    logger.info(f"Validation data: {validation_datafile}")
    logger.info(f"Test data: {test_datafile}")
    training_parameters = config.get("train", {})
    if training_parameters:
        logger.info(f"Loaded training parameters: {json.dumps(training_parameters, indent=4)}")

    # Initialize model with train/validation/test data
    srl_model = SemanticRoleLabeler(
        train_conllup=train_datafile,
        validation_conllup=validation_datafile,
        test_conllup=test_datafile,
        base_model=base_model_name,
        model_outdir=model_outdir,
    )

    # Train model
    srl_model.train(outdir=model_outdir, **training_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tunes a semantic role labeling model on CONLLU-formatted data with Universal Dependencies + Universal Propositions annotations.")
    parser.add_argument('--config', default=None, help='Path to config.yml file')
    args = parser.parse_args()
    main(args.config)
