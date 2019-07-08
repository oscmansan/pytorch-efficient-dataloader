#!/usr/bin/env bash
module load cuda/10.0 cudnn/7.4
source venv/bin/activate
export PYTHONUNBUFFERED=1
srun -p gpi.compute -t 1-00 -c 8 --mem 8GB --gres gpu:turing:1 --pty python src/train.py