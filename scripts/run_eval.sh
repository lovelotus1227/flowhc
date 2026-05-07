#!/bin/bash
# make sure to set cfg.load_from

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5 accelerate launch main.py --config "config/sth_com.yaml" --mode "eval"