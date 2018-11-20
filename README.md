# OpenAI-GPT

Language model pretraining for NLU with [openai_gpt](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). 

## 1. Environment

```
tensorflow-gpu==1.9.0
cuda==9.0
```

## 2. Data

/dccstor/jinfeng_nlp/chinese_data/original_data

```
Wikipedia
Sohu news
AIchallenger
DuReader
```

## 3. Quick Start

### Cleaning your data with your own desire

tokenizer, stop words, bpe, etc.

### Training data to TFrecord files

`python train.py --preprocess=True --char_word=char --checkpoint_name=pretrain_char --data_dir=data --raw_data_name=corpus.char.all`

The params related to preprocess are as follow:

```python
--preprocess, bool, 'preprocess for tfrecord files'
--checkpoint_name, string, 'model name'
--data_dir, string, 'data folder where contrain the raw_data and the generated tfrecord file will save in data_dir/checkpoint_name/'
--raw_data_name, string, 'raw data filename'
--char_word, string, 'char level or word level'
```

### Config the model params

### Train your model with distributed TF

For example, training with 32 nodes and each node with 1 gpu core.

`python RunDistributedJob.py -N=32 -G=1 train.py`

The params related to RunDistributedJob are as follow:

```python
-N int 'number of calculation nodes'
-G int 'number of GPUs required for each node'
```

**Note**: the num of ps(parameter sever) is N/2 by default, you can config it by edit related codes in RunDistributedJob.py.

## 4. Tips for CCC Platform

submit a job: 

`jbsub -queue x86_7d -cores 4 -mem 64g python train.py`

kill jobs: 

`jbadmin -kill -proj LM_pretrain all`

moitor jobs: 

`watch -n 5 jbinfo -long`

