"""Evaluate a trained SRL model on gold-annotated .conllup data.

Example usage:
```
python -m srl.evaluate --model /path/to/model --data data/de_gsd-ud-up-test.conllup
```
"""
import argparse
import json
import logging

from .constants import (DEFAULT_TEST_DATA, END_PREDICATE_TOKEN,
                        START_PREDICATE_TOKEN)
from .metrics import srl_metrics
from .predict import SRLModel
from .train import SemanticRoleDataset

logger = logging.getLogger(__name__)


def unmark_example(example: dict) -> tuple[list[str], int, list[str]]:
    """Remove the predicate markers from a training example.

    Returns the tokens, the index of the predicate, and one gold label per token.
    """
    markers = {START_PREDICATE_TOKEN, END_PREDICATE_TOKEN}
    pred_idx = example["tokens"].index(START_PREDICATE_TOKEN)
    tokens, labels = zip(*[
        (token, label) for token, label in zip(example["tokens"], example["labels"]) if token not in markers
    ])
    return list(tokens), pred_idx, list(labels)


def evaluate(srl_model: SRLModel, examples: list[dict], per_role: bool = True) -> dict:
    """Predict each (predicate-marked) example and compute word-level metrics."""
    true_labels, pred_labels = [], []
    for example in examples:
        tokens, pred_idx, gold = unmark_example(example)
        prediction = srl_model.predict_srl(tokens, pred_idx)
        true_labels.append(gold)
        pred_labels.append([label for _, label in prediction])
    metrics = srl_metrics(true_labels, pred_labels, per_role=per_role)
    metrics["examples"] = len(examples)
    return metrics


def format_metrics(metrics: dict) -> str:
    lines = [
        f"{metrics['examples']} predicate examples",
        f"accuracy {metrics['accuracy']:.4f}  precision {metrics['precision']:.4f}  "
        f"recall {metrics['recall']:.4f}  f1 {metrics['f1']:.4f}",
    ]
    if "roles" in metrics:
        lines.append(f"{'role':12} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>8}")
        for role, scores in sorted(metrics["roles"].items(), key=lambda item: -item[1]["support"]):
            lines.append(f"{role:12} {scores['precision']:9.4f} {scores['recall']:9.4f} "
                         f"{scores['f1-score']:9.4f} {int(scores['support']):8d}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluates a trained semantic role labeling model on .conllup data.")
    parser.add_argument('--model', default=None, help='Trained model directory or HuggingFace Hub ID (default: $SRL_MODEL_DIR)')
    parser.add_argument('--data', default=DEFAULT_TEST_DATA, help=f'.conllup file to evaluate on (default: {DEFAULT_TEST_DATA})')
    parser.add_argument('--output', default=None, help='Optional .json file to save the metrics to')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    srl_model = SRLModel.from_pretrained(args.model)
    examples = SemanticRoleDataset(args.data).examples
    metrics = evaluate(srl_model, examples)
    print(format_metrics(metrics))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
