"""Constants and defaults for predicate-conditioned semantic role labeling (SRL)."""

# Special tokens inserted immediately before/after the predicate the model labels
START_PREDICATE_TOKEN = "[PRED]"
END_PREDICATE_TOKEN = "[/PRED]"

# BIO tagging scheme for argument spans, e.g. B-A0, I-A0, B-AM-TMP
BEGINNING_TAG = "B"
INSIDE_TAG = "I"
NULL_SRL_TAG = "O"  # O = outside any argument

# Universal POS tags of verbs, and STTS tags of participles, used to identify predicates
VERB_TAGS = {
    "VERB",
    "AUX",
}
PAST_PARTICIPLE_TAG = "VVPP"      # past participle, e.g. <hat ... gesehen>
PAST_PARTICIPLE_AUX_TAG = "VAPP"  # auxiliary past participle, e.g. <ist ... gewesen>

# Base model (HuggingFace Hub ID) fine-tuned for German SRL
DEFAULT_BASE_MODEL = "liaad/srl-en_xlmr-base"

# spaCy German model used by the command-line interface to preprocess text
DEFAULT_SPACY_MODEL = "de_core_news_lg"

# Training defaults (relative to the working directory the training script is run from)
DEFAULT_TRAIN_DATA = "data/de_gsd-ud-up-train.conllup"
DEFAULT_VALIDATION_DATA = "data/de_gsd-ud-up-dev.conllup"
DEFAULT_TEST_DATA = "data/de_gsd-ud-up-test.conllup"
DEFAULT_MODEL_OUTDIR = "pretrained/liaad/srl-en_xlmr-base"

# Environment variable that may point to the directory of a trained SRL model
MODEL_DIR_ENV_VAR = "SRL_MODEL_DIR"
