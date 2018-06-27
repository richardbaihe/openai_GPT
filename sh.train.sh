#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u train.py --n_gpu=1 --n_ctx=73 --n_batch=128 --n_iter=10 --lr=1e-4 --n_head=8 --n_layer=6 --lm_coef=0.2 --desc=nctx73_lm
