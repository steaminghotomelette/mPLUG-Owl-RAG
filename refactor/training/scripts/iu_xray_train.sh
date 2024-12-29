#!/bin/bash

CUDA_VISIBLE_DEVICES=0
swift sft \
  --model_type mplug_owl3 \
  --model "C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/services/chat/iic/mPLUG-Owl3-1B-241014" \
  --train_type lora \
  --dataset "C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/iu_xray/iu_xray_train_data.json" \
  --val_dataset "C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/iu_xray/iu_xray_valid_data.json" \
  --output_dir output \
  --num_train_epochs 5 \
  --freeze_aligner False

swift infer \
    --adapters "C:\Users\ILLEGEAR\personal-projects\mPLUG-Owl-RAG\refactor\training\output\v3-20241229-030738\checkpoint-6000" \
    --merge_lora true \