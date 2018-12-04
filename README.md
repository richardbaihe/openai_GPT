# OpenAI-GPT

Language model pretraining for NLU with [openai_gpt](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). 
Note: coded of pretraining are published in this repository, while the pre-trained models are private and cannot be published on my own desire. You can train your own model with your GPU Cluster and monolingual data.

## 1. Environment

```
python 3.6
cuda 9.0
tensorflow-gpu 1.9.0
```

## 2. Data


```
Wikipedia
Sohu news
AIchallenger
DuReader
```

## 3. Quick Start

### 3.1 Cleaning your data with your own desire

tokenizer, stop words, bpe, etc.

### 3.2 Training data to TFrecord files

`python train.py --preprocess=True --char_word=char --checkpoint_name=pretrain_char --data_dir=data --raw_data_name=corpus.char.all`

The params related to preprocess are as follow:

```python
--preprocess, bool, 'preprocess for tfrecord files'
--checkpoint_name, str, 'model name'
--data_dir, str, 'data folder where contrain the raw_data and the generated tfrecord file will save in data_dir/checkpoint_name/'
--raw_data_name, str, 'raw data filename'
--char_word, str, 'char level or word level'
```

### 3.3 Config the model params and training config

training config

```python
--bootstrap_host, str,'The hostname or IP of the bootstrapping server'
--bootstrap_port, int, 'The port of the bootstrapping server'
--n_gpu, int, 'nums of gpu used'
--num_ps, int, 'number of parameter server'
--save_dir, str, 'folder name for model to be saved'
--steps_to_validate, int, 'validating every n steps'
--n_iter, int, 'total epochs'
--n_step, int, 'total steps'
--n_batch, int, 'batch size'
--seed, int, 'random seed'
--max_grad_norm, int, 'max grad norm'
--opt, str, 'gradient updating method, like adam'
--b1, float,'adam'
--b2, float,'adam'
--e, float, 'adam'
--lr, float, 'learning rate'
--lr_schedule, str, 'warm up schedule'
--lr_warmup, float, 'warm up percent'
--pre_load, bool, 'pre load previous model or not'
--n_transfer, int, 'nums of layers to be load'
--lm_coef, float, 'language model weight in multi-task training'
```

model params

```python
--n_embd, int, 'embedding size'
--n_head, int, 'nums of multi-head'
--n_layer, int, 'nums of layers'
--embd_pdrop, float, 'dropout prob of embedding'
--attn_pdrop, float, 'dropout prob of attention'
--resid_pdrop, float, 'dropout prob of residual'
--clf_pdrop, float, 'dropout prob of classify'
--l2, float, 'l2 regularization'
--vector_l2, bool, 'whether vector L2'
--afn, str, 'activation function'
```

### 3.4 Train your model with distributed TF

For example, training with 32 nodes and each node with 1 gpu core.

`python RunDistributedJob.py -N=32 -G=1 train.py`

The params related to RunDistributedJob are as follow:

```python
-N int 'number of calculation nodes'
-G int 'number of GPUs required for each node'
```

**Note**: the num of ps(parameter sever) is N/2 by default, you can config it by editing related codes in RunDistributedJob.py.

## 4. Tips for CCC Platform

submit a job: 

`jbsub -queue x86_7d -cores 4 -mem 64g python train.py`

kill jobs: 

`jbadmin -kill -proj LM_pretrain all`

moitor jobs: 

`watch -n 5 jbinfo -long`

