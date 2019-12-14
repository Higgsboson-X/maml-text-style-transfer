# MAML-Text-Style-Transfer
MAML framework applied to text style transfer

## Model Overview
This repository contains two text style transfer base models, *CrossAlign* and *VAE*, as well as their meta-learning versions. This project is to extend the usual concepts of style to general writing styles. Because of the small data size of such styles, we propose the meta-learning scheme to enable models to utilize information from other domains (style pairs) and quickly adapt to a specific domain using fine-tuning.

### CrossAlign for Style Transfer

*CrossAlign* model for text style transfer is proposed by [Shen et al.](https://github.com/shentianxiao/language-style-transfer) in 2017. The model structure is shown in the following figure, where *s*'s are the style labels, *z*'s are the encoded sentence vectors. *D* denotes adversarial discriminators. The encoder and decoder are a seq2seq model with single GRU units. The adversaries and seq2seq model are trained sequentially in each step of updates.

[cross align](https://github.com//Higgsboson-X/maml-text-style-transfer/blob/master/images/crossalign.png "Cross Align")

### VAE for Style Transfer

Variational autoencoder for style transfer is proposed by [John et al.](https://github.com/vineetjohn/linguistic-style-transfer) in 2018. It uses a set of style- and content-oriented losses to disentangle style and content embedding in latent space. The model diagram is shown below.

[vae](https://github.com//Higgsboson-X/maml-text-style-transfer/blob/master/images/vae.png "VAE")

### Model Agnostic Meta-learning

Compared with other model-based meta-learning methods, model agnostic meta-learning only uses the gradient information, and thus is well suited for algorithms that optimize thier objectives using gradient descent. The model architecture is shown below.

[maml](https://github.com//Higgsboson-X/maml-text-style-transfer/blob/master/images/maml.png "MAML")

Specifically, we adapt this model architecture to our two base models above for our model, called Small-data Text Style Transfer (ST2).

[st2](https://github.com//Higgsboson-X/maml-text-style-transfer/blob/master/images/st2.png "ST2")

## Dependencies

- python 3.x
- nltk 3.4.5
- kenlm 0.0.0 (for evaluation)
- sklearn (for evaluation)
- gensim 3.8.1
- numpy 1.17.4
- torch 1.3.1

## Folder Structure

```
maml-cross-align/maml-vae
  |--- code
  |--- config
  |--- data
  |     |--- s1
  |          |--- processed (processed vocab, seqs, lengths, etc.)
  |                   |--- pretrain
  |                   |--- {n_tasks}t
  |          |--- train
  |          |--- val
  |          |--- (infer)
  |     |--- s2
  |          |--- processed (processed vocab, seqs, lengths, etc.)
  |                   |--- pretrain
  |                   |--- {n_tasks}t
  |          |--- train
  |          |--- val
  |          |--- (infer)
  |--- ckpt
  |     |--- s1
  |     |--- s2
  |--- emb
  |     |--- s1
  |     |--- s2
  |--- output
  |     |--- s1
  |     |--- s2
```

## Prepare Dataset

The project is organized as a multi-task problem. Two datasets are experimented in the paper, the *Literature Translations* dataset and the *Grouped Standard Dataset*, denoted as *s1* and *s2* in the folder. All the files are named as `t{task_id}`, where `task_id` are from `1, 2, 3, ..., n_tasks`. If the number of task data sets provided is larger than the number of tasks specified when running the program, e.g., in the `config` file, than the first `n_tasks` will be used for running. The two datasets used in the paper are described below. Only the common-source works used for testing are listed in the LT set, and the training data is collected from other works by the same writer in the 2nd and 3rd columns.

### Literature Translations (LT)

|Common Source | Writer A | Writer B|
|:------------:|:--------:|:-------:|
|Notre-Dame de Paris|Alban Kraisheimer|Isabel F. Hapgood|
|The Brothers Karamazov|Andrew R. MacAndrew|Richard Pevear|
|The Story of Stone|David Hawkes|Yang Xianyi|
|The Magic Mountain|John E. Woods|H. T. Lowe-Porter|
|The Illiad|Ian C. Johnston|Robert Fagles|
|Les Miserables|Isabel F. Hapgood|Julie Rose|
|Crime and Punishment|Michael R. Katz|Richard Pevear|

### Grouped Standard Datasets (GSD)

|Task ID|Dataset|Style 0/1|
|:-----:|:-----:|:---:|
|1|Yelp| (health) negative/positive|
|2|Amazon| (musical instrument) negative/positive|
|3|GYAFC| (relations) informal/formal|
|4|Wikipedia| simple/standard|
|5|Bible|easy/standard|
|6|Britannica|simple/standard|
|7|Shakespeare|modern/original|

## Run

### MAML-CrossAlign

Suppose the test set is `${s}` and the sample task id is `${t}`, with 7 tasks.
#### Change directory to `code`.
```
$ cd maml-cross-align/code
```
#### Prepare merged data for pretrain
```
$ bash scripts/get_pretrain_text.sh s${s}
```
#### Make directories for output
```
bash scripts/make_dirs.sh s${s} 7
```
#### Run original *CrossAlign* model with pretrain
```
python3 original.py --config-path ../config/s${s}.json --corpus s${s} --task-id ${t} --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 10
```
Add `--load-model` to load the last ckpt specified by the `config` file, and `--load-data` if the data are processed and saved in the `processed` directory.
#### Run MAML-CrossAlign
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s}$ --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --maml-batch-size 32 --sub-batch-size 64 --train-batch-size 64
```
Add `--load-model` to load the last ckpt specified by the `config` file, and `--load-data` if the data are processed and saved in the `processed` directory.
#### Extract embeddings
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s} --extract-embeddings --task-id ${t} --ckpt epoch-1.t${t}
```
Add `--from-pretrain` if loading vocab from `pretrained` directory.
#### Online inference
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s} --extract-embeddings --task-id ${t} --ckpt epoch-1.t${t}
```
Add `--from-pretrain` if loading vocab from `pretrained` directory.

### MAML-VAE

Suppose the test set is `${s}` and the sample task id is `${t}`, with 7 tasks.
#### Change directory to `code`.
```
$ cd maml-vae/code
```
#### Prepare merged data for pretrain
```
$ bash scripts/get_pretrain_text.sh s${s}
```
#### Make directories for output
```
bash scripts/make_dirs.sh s${s} 7
```
#### Train Word2Vec model
```
python3 wdv.py --config-path ../config/s${s}.json --corpus s${s} --task-id ${t} --model vae
```
Change `vae` to `maml` and use `--task-id 0` if using the MAML model rather than original VAE.
#### Run original *VAE* model with pretrain
```
python3 original.py --config-path ../config/s${s}.json --corpus s${s} --task-id ${t} --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 5
```
Add `--load-model` to load the last ckpt specified by the `config` file, and `--load-data` if the data are processed and saved in the `processed` directory.
#### Run inference with original model (and dump embeddings)
```
python3 original.py --config-path ../output/s${s}/t${t}.json --corpus s${s} --task-id ${t} --inference --dump-embeddings
```
Add `--from-pretrain` if pretrain phase exists.
#### Run MAML-VAE
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s}$ --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 --maml-batch-size 16 --sub-batch-size 64 --train-batch-size 64
```
Add `--load-model` to load the last ckpt specified by the `config` file, and `--load-data` if the data are processed and saved in the `processed` directory.
#### Run inference (and dump embeddings)
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s} --task-id ${t} --dump-embeddings
```
The checkpoint is reloaded as specified in the `config` file.
#### Extract embeddings
```
python3 main.py --config-path ../config/s${s}.json --corpus s${s} --extract-embeddings --task-id t${t} --ckpt epoch-1.t${t}$
```
Add `--from-pretrain` if loading vocab from `pretrained` directory.
#### Online inference
```
python3 main.py --online-inference --config-path ../config/s${s}.json --corpus s${s} --ckpt epoch-1.t${t} --tgt-file ../data/s${s}/val/t${t}.0
```
The `--tgt-file` argument specifies the document from which to extract conditioning style embeddings.

## Evaluation
Check `evaluation` folder for more details.
* TextCNN classifier
* KenLM bigram language model for perplexity
* t-SNE plots for style and content embedding
