"""Word-level evaluation metrics of predicted argument spans."""
from seqeval.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

from .constants import BEGINNING_TAG, NULL_SRL_TAG

PREDICATE_TAG = f"{BEGINNING_TAG}-V"


def ignore_predicate(labels: list[str]) -> list[str]:
    """Replace the predicate's label by the outside label.

    Reasoning: The predicate is marked in the model's input,
    so scoring it as an argument would inflate the metrics.
    """
    return [NULL_SRL_TAG if label == PREDICATE_TAG else label for label in labels]


def srl_metrics(true_labels: list[list[str]],
                pred_labels: list[list[str]],
                per_role: bool = False,
                ) -> dict:
    """Word-level metrics of argument spans, from one BIO label per word for each predicate example.

    accuracy: share of words with the correct label (the predicate counts as "O").
    precision, recall, f1: micro-averaged over argument spans (seqeval, exact span and role match);
        the predicate itself (B-V) is not evaluated as an argument.
    per_role: also return precision, recall, f1, and support per role under "roles".
    """
    true_labels = [ignore_predicate(seq) for seq in true_labels]
    pred_labels = [ignore_predicate(seq) for seq in pred_labels]
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "f1": f1_score(true_labels, pred_labels, zero_division=0),
    }
    if per_role:
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        metrics["roles"] = {
            role: {key: float(value) for key, value in scores.items()}
            for role, scores in report.items() if not role.endswith(" avg")
        }
    return metrics
