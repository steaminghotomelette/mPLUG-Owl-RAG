#!/bin/bash

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type mplug_owl3_241101 \
    --model "/home/ec2-user/Repos/mPLUG-Owl-RAG/refactor/training/iic/mPLUG-Owl3-7B-241101" \
    --train_type lora \
    --attn_impl flash_attn \
    --dataset "/home/ec2-user/Repos/mPLUG-Owl-RAG/output.json" \
    --deepspeed zero2 \
    --output_dir output \
    --num_train_epochs 3 \
    --freeze_aligner False \
    --strict False \
    --split_dataset_ratio 0.001 \