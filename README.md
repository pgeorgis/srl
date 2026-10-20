# Semantic Role Labeling (SRL)

Predicate-conditioned semantic role labeling (SRL) of German text. Originally developed for a [German corpus study of pronouns](https://github.com/SFB1252/c07-dpron-korpus) within the [CRC 1252](https://sfb1252.uni-koeln.de/en/) at the University of Cologne (Universität zu Köln), Germany.

## Overview
This project fine-tunes a pretrained transformer for predicate-conditioned SRL on German data. Each predicate in a sentence becomes its own training example. The model predicts BIO semantic role labels (`A0`, `A1`, etc.) for arguments of the marked predicate.


## Installation
This library requires Python 3.11 or later and is best managed with [uv](https://docs.astral.sh/uv/).

As a development environment:
```bash
uv sync                                  # inference only
uv sync --extra cli                      # also spaCy (with numpy<2), for the command-line interface
uv sync --extra train                    # also the dependencies for fine-tuning new models
```

The command-line interface additionally needs the spaCy German model, which is not on PyPI:
```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.5.0/de_core_news_lg-3.5.0.tar.gz
```
(`uv sync` removes packages that are not in the lock file, so re-run this after syncing, or sync with `uv sync --inexact`.)

### Trained model
**No trained model is included in this repository** (the weights are too large). Obtain a trained model directory (containing `config.json`, `model.safetensors`, `tokenizer.json`, ...) and either pass its path (or a Hugging Face Hub model ID) to `SRLModel.from_pretrained(model_path)`, or set the environment variable `SRL_MODEL_DIR` to its location, e.g.:
```bash
export SRL_MODEL_DIR=/path/to/your/pretrained/model
```
To reuse one copy of the model across several projects, keep it in a single central location, e.g. in the git-ignored `pretrained/` folder of one clone of this repository (or train it, see [Training](#training)), and point `SRL_MODEL_DIR` at it in each project.

The base model `liaad/srl-en_xlmr-base` is downloaded from the Hugging Face Hub on first use in training.

## Data
Input data consist of Universal Dependences (UD) German datasets augmented with Universal Propositions (UP) annotations in `.conllup` format. These files include sentences annotated with predicates and argument spans for each predicate.
- `PRED` column indicates whether a token is a predicate
- `ARGHEADS` column marks the head token of each argument of the predicate
- `ARGSPANS` column encodes argument spans for that predicate


For example:
```
# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL UP:PRED UP:ARGHEADS UP:ARGSPANS
# sent_id = train-s617
# text = Auch am nächsten Tag konnte ein anderer Mitarbeiter unsere Fragen nicht beantworten.
1	Auch	auch	ADV	ADV	_	13	advmod	_	_	_
2-3	am	_	_	_	_	_	_	_	_	_
2	an	an	ADP	APPR	_	5	case	_	_	_
3	dem	der	DET	ART	Case=Dat|Definite=Def|Gender=Masc|Number=Sing|PronType=Art	5	det	_	_	_
4	nächsten	nah	ADJ	ADJA	Case=Dat|Gender=Masc|Number=Sing	5	amod	_	_	_
5	Tag	Tag	NOUN	NN	Case=Dat|Gender=Masc|Number=Sing	13	obl	_	_	_
6	konnte	können	AUX	VMFIN	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	13	aux	_	_	_
7	ein	ein	DET	ART	Case=Nom|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art	9	det	_	_	_
8	anderer	ander	ADJ	ADJA	Case=Nom|Gender=Masc|Number=Sing	9	amod	_	_	_
9	Mitarbeiter	Mitarbeiter	NOUN	NN	Case=Nom|Gender=Masc|Number=Sing	13	nsubj	_	_	_
10	unsere	unser	DET	PPOSAT	Case=Acc|Gender=Fem|Number=Plur|Number[psor]=Plur|Person=1|Poss=Yes|PronType=Prs	11	det:poss	_	_	_
11	Fragen	Frage	NOUN	NN	Case=Acc|Gender=Fem|Number=Plur	13	obj	_	_	_
12	nicht	nicht	PART	PTKNEG	Polarity=Neg	13	advmod	_	_	_
13	beantworten	beantworten	VERB	VVINF	VerbForm=Inf	0	root	answer.01	AM-TMP:5|AM-MOD:6|A0:9|A1:11|AM-NEG:12	AM-TMP:5-5|AM-MOD:6-6|A0:9-9|A1:10-11|AM-NEG:12-12
14	.	.	PUNCT	$.	_	13	punct	_	_	_
```

### Data Preparation
The prepared train, validation, and test datasets are saved in this repository under [`data/`](./data/).

Source: https://github.com/UniversalPropositions/UP_German-GSD

Following instructions at the source repository, UP annotations are merged with UD datasets using:
```
python3 up2/merge_ud_up.py --input_ud=<ud_path_to_UD_German-GSD> --input_up=<up_path_to_UP_German-GSD> --output=<path_to_UD+UP_German-GSD>
```

The above command is to be run from within the cloned source repository after also cloning its dependencies and setting up the `up2` environment. See instructions in the source repository [`README`](https://github.com/UniversalPropositions/UP_German-GSD/blob/main/README.md) for details.

## Model Inputs
Inputs are purely text tokens plus two special predicate markers:
- `[PRED]` immediately before the predicate
- `[/PRED]` immediately after the predicate

No explicit UD parse features (head/deprel/pos/feats) are used as model inputs. Any syntactic information is learned implicitly by the base model from context.

### Training Examples
For each predicate in a sentence:
- Mark the predicate with special tokens.
- Build BIO labels from ARGSPANS (e.g., `A0`, `A1`, `AM-TMP`, etc.).
- Label the predicate token as `B-V`.

This yields one training instance per predicate with a token sequence and a BIO label sequence of the same length.

## Model
- Default base model: [`liaad/srl-en_xlmr-base`](https://huggingface.co/liaad/srl-en_xlmr-base)
- Task head: token classification with one label per token
- Special tokens are added to the tokenizer and embedding matrix is resized

### Training
- Training is performed with the [`train.py`](./src/srl/train.py) module together with an input config .yml file (e.g. [`srl_config.yml`](./configs/srl_config.yml)). Install the training dependencies (`uv sync --extra train`) and run from the repository root:
  ```bash
  uv run python -m srl.train --config configs/srl_config.yml
  ```
  The output directory is set by `model_outdir` in the config file, or on the command line with `--output-dir`. Paths are relative to the working directory.
- Tokenization uses `is_split_into_words=True` and aligns word-level BIO labels to subword tokens.
- Loss ignores special tokens and padding (label `-100`).
- Validation is performed during training with `seqeval` metrics.

### Evaluation
- Metrics: accuracy, precision, recall, F1 (`seqeval`)
- Test metrics are saved to `test_metrics.json` within the model output directory

Three base models were fine-tuned and tested on the German SRL datasets (test set; the metrics and configs of each are saved under [`results/`](./results/)):
- [`xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base): multilingual model (100 languages) pre-trained on 2.5TB of filtered CommonCrawl data.
- [`dbmdz/bert-base-german-cased`](https://huggingface.co/dbmdz/bert-base-german-cased): monolingual German model pre-trained on 16GB of German text from web crawl.
- [`liaad/srl-en_xlmr-base`](https://huggingface.co/liaad/srl-en_xlmr-base): `xlm-roberta-base` model fine-tuned on the English CoNLL formatted OntoNotes v5.0 semantic role labeling data.

| Model                     | Accuracy | Precision | Recall  | F1 Score |
|--------------------------|----------|-----------|---------|----------|
| xlm-roberta-base         | 0.8674   | 0.7172    | 0.7653  | 0.7405   |
| dbmdz/bert-base-german-cased   | 0.8730   | 0.7078    | 0.7644  | 0.7350   |
| liaad/srl-en_xlmr-base   | **0.8749** | **0.7345** | **0.7724** | **0.7530** |

`liaad/srl-en_xlmr-base` performed the best in all test metrics and is the default model.

### Outputs
- `train.log`: full training and evaluation logs
- `test_metrics.json`: final test evaluation metrics
- `training_curves.png`: plot of loss and accuracy curves during training
- model and tokenizer artifacts

## Inference
Trained models can annotate the semantic roles of arguments in novel input sentences.

Example usage:
```python
from srl import SRLModel, get_predicate_indices

model = SRLModel.from_pretrained("/path/to/model")  # or set $SRL_MODEL_DIR
tokens = ["Der", "Hund", "jagt", "die", "Katze", "."]
model.annotate_srl(token_texts=tokens, pred_indices=[2], pretty_print=True)
```

Example output:
```
=== Predicate: jagt ===
Der                  B-A0
Hund                 I-A0
jagt                 B-V
die                  B-A1
Katze                I-A1
.                    O
```


Example usage of the command-line interface (requires `uv sync --extra cli`, the spaCy model `de_core_news_lg`, and a trained model, see [above](#trained-model)):

```
python -m srl --text "Die Kirche, die die Dorfbewohner während des Krieges als Zufluchtsort benutzten, wurde von Mönchen im 14. Jahrhundert erbaut."

Text: Die Kirche, die die Dorfbewohner während des Krieges als Zufluchtsort benutzten, wurde von Mönchen im 14. Jahrhundert erbaut.

=== Predicate: benutzten ===
Die                  B-A1
Kirche               I-A1
,                    O
die                  B-R-A1
die                  B-A0
Dorfbewohner         I-A0
während              B-AM-TMP
des                  I-AM-TMP
Krieges              I-AM-TMP
als                  B-A2
Zufluchtsort         I-A2
benutzten            B-V
,                    O
wurde                O
von                  O
Mönchen              O
im                   O
14.                  O
Jahrhundert          O
erbaut               O
.                    O

=== Predicate: erbaut ===
Die                  B-A1
Kirche               I-A1
,                    I-A1
die                  I-A1
die                  I-A1
Dorfbewohner         O
während              O
des                  O
Krieges              O
als                  O
Zufluchtsort         O
benutzten            O
,                    O
wurde                O
von                  B-A0
Mönchen              I-A0
im                   B-AM-TMP
14.                  I-AM-TMP
Jahrhundert          I-AM-TMP
erbaut               B-V
.                    O
```

### Post-processing
The model outputs a BIO label per token for each predicate. [`postprocess.py`](./src/srl/postprocess.py) converts these into argument spans for a whole parsed document:
```python
from srl import SRLModel, postproc_srl, combine_postproc_srls

# tokens: token dicts for the whole document (keys "text", "pos", "tag", "dep"), e.g. from token_dicts_from_spacy(doc)
# sentences: {sentence index: [first token index, end token index]}
srls = postproc_srl(tokens, sentences, SRLModel.from_pretrained())
# {predicate token index: {"text": "jagt", "args": [{"span": [0, 1], "text": "Der Hund", "label": "A0"}, ...]}}

combined = combine_postproc_srls(srls)  # {token index: [{"pred_idx", "predicate", "srl"}, ...]} over all predicates
```

## Repository structure
* [`src/srl/`](./src/srl/)
    * [`predict.py`](./src/srl/predict.py): `SRLModel` (prediction), `get_predicate_indices`, command-line interface.
    * [`postprocess.py`](./src/srl/postprocess.py): `postproc_srl` and `combine_postproc_srls` (BIO labels to argument spans).
    * [`train.py`](./src/srl/train.py): fine-tuning on `.conllup` data.
    * [`paths.py`](./src/srl/paths.py), [`constants.py`](./src/srl/constants.py): locating the trained model, special tokens, BIO tags, and defaults.
* [`configs/srl_config.yml`](./configs/srl_config.yml): example training configuration.
* [`data/`](./data/): combined UD+UP training, validation, and test data.
* [`results/`](./results/): configs and evaluation metrics of the trained models (without the weights).

