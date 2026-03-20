# Semantic Role Labeling (SRL)

## Overview
This project fine-tunes a pretrained transformer for predicate-conditioned SRL on German data. Each predicate in a sentence becomes its own training example. The model predicts BIO semantic role labels (`A0`, `A1`, etc.) for arguments of the marked predicate.

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
The prepared train, validation, and test datasets are saved in this repository under `./models/srl/data/`.

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
- Default base model: `xlm-roberta-base`
- Task head: token classification with one label per token
- Special tokens are added to the tokenizer and embedding matrix is resized

### Training
- Tokenization uses `is_split_into_words=True` and aligns word-level BIO labels to subword tokens.
- Loss ignores special tokens and padding (label `-100`).
- Validation is performed during training with `seqeval` metrics.

### Evaluation
- Metrics: accuracy, precision, recall, F1 (`seqeval`)
- Test metrics are saved to `test_metrics.json` within the model output directory

### Outputs
- `train.log`: full training and evaluation logs
- `test_metrics.json`: final test evaluation metrics
- `training_curves.png`: plot of loss and accuracy curves during training
- model and tokenizer artifacts
