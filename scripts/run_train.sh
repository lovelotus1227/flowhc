#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch main.py --config "config/base.yaml" --mode "train"