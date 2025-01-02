#!/bin/bash

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'path/to/vx-xxx/checkpoint-xxx' \
    --merge_lora true \