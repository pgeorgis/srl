"""Predicate-conditioned semantic role labeling (SRL) of German text."""
from .constants import (BEGINNING_TAG, END_PREDICATE_TOKEN, INSIDE_TAG,
                        NULL_SRL_TAG, START_PREDICATE_TOKEN)
from .paths import find_model
from .predict import (SRLModel, get_predicate_indices, mark_predicate,
                      token_dicts_from_spacy)

__all__ = [
    "SRLModel",
    "get_predicate_indices",
    "mark_predicate",
    "token_dicts_from_spacy",
    "find_model",
    "BEGINNING_TAG",
    "INSIDE_TAG",
    "NULL_SRL_TAG",
    "START_PREDICATE_TOKEN",
    "END_PREDICATE_TOKEN",
]
