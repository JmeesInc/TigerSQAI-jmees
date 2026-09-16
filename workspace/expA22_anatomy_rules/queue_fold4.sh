#!/usr/bin/env bash
# fold1 (GPU0) の完走を待って fold4 を同じ GPU で流す
cd "$(dirname "$0")"
while pgrep -f "train.py --fold 1$" > /dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=0 ./run.sh 4 > train_fold4.log 2>&1
