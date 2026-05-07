#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5,7 accelerate launch main.py --config "config/sth_com.yaml" --mode "train"