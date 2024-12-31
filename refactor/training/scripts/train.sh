#!/bin/bash

CUDA_VISIBLE_DEVICES=0 swift sft --model_type mplug_owl3 --model "/home/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/iic/mPLUG-Owl3-1B-241014" --train_type lora --attn_impl flash_attn --dataset "/home/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/output.json" --deepspeed default-zero2 --output_dir output --num_train_epochs 5 --freeze_aligner False

swift infer \
    --adapters "C:\Users\ILLEGEAR\personal-projects\mPLUG-Owl-RAG\refactor\training\output\v3-20241229-030738\checkpoint-6000" \
    --merge_lora true \