# Language model pretraining for NLU
Training with multi-gpu on multi-machines with **CCC** platform.

Examples: 

## 1. Training

`python RunDistributedJob.py -N=32 -G=1 train.py`

## 2. Tips for CCC Platform

submit a job: 

`jbsub -queue x86_7d -cores 4 -mem 64g python train.py`
kill jobs: 

`jbadmin -kill -proj LM_pretrain all`
moitor jobs: 

`watch -n 5 jbinfo -long`

