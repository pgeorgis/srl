"""Constants and defaults for predicate-conditioned semantic role labeling (SRL)."""

# Special tokens inserted immediately before/after the predicate the model labels
START_PREDICATE_TOKEN = "[PRED]"
END_PREDICATE_TOKEN = "[/PRED]"

# BIO tagging scheme for argument spans, e.g. B-A0, I-A0, B-AM-TMP
BEGINNING_TAG = "B"
INSIDE_TAG = "I"
NULL_SRL_TAG = "O"  # O = outside any argument

# Core semantic arguments and labels
CORE_ARGS = ["A0", "A1", "A2"]
CORE_SR_LABELS = [
    NULL_SRL_TAG,
    f"{BEGINNING_TAG}-V",
] + [
    f"{BEGINNING_TAG}-{core_srl}" for core_srl in CORE_ARGS
] + [
    f"{INSIDE_TAG}-{core_srl}" for core_srl in CORE_ARGS
]

# Universal POS tags of verbs, and STTS tags of participles, used to identify predicates
VERB_TAGS = {
    "VERB",
    "AUX",
}
# STTS tags of non-finite verbs whose (auxiliary or modal) head is not a predicate of its own
NONFINITE_VERB_TAGS = {
    "VVPP",             # past participle, e.g. <hat ... gesehen>
    "VAPP",             # auxiliary past participle, e.g. <ist ... gewesen>
    "VMPP",             # modal past participle, e.g. <hat ... gekonnt>
    "VVINF",            # full infinitive, e.g. <würde ... sehen>
    "VAINF",            # auxiliary infinitive, e.g. <muss ... haben>, <kann ... sein>, <wird ... werden>
    "VMINF",            # modal infinitive, e.g. <hätte ... sehen können>
}

# Base model (HuggingFace Hub ID) fine-tuned for German SRL
DEFAULT_BASE_MODEL = "liaad/srl-en_xlmr-base"

# spaCy German model used by the command-line interface to preprocess text
DEFAULT_SPACY_MODEL = "de_core_news_lg"

# Training defaults (relative to the working directory the training script is run from)
DEFAULT_TRAIN_DATA = "data/de_gsd-ud-up-train.conllup"
DEFAULT_VALIDATION_DATA = "data/de_gsd-ud-up-dev.conllup"
DEFAULT_TEST_DATA = "data/de_gsd-ud-up-test.conllup"
# Parent directory of trained models if no output directory is given: <DEFAULT_OUTDIR_ROOT>/<base model>
DEFAULT_OUTDIR_ROOT = "out"

# Environment variable that may point to the directory of a trained SRL model
MODEL_DIR_ENV_VAR = "SRL_MODEL_DIR"
