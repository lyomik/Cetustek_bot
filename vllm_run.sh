#!/bin/bash
vllm serve /data/models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-lora \
  --lora-modules cetustek-bot=./finetune_model/cetustek_qwen3_14b_lora_v4 \
  --max-lora-rank 16 \
  --enforce-eager                       # 關掉 CUDA graph，省 0.78 GiBank 32 \
  
#紀錄：QLora epoch:5 | Lora epoch:8


#vllm 0.20.1
# torch==2.11.0
# torchaudio==2.11.0
# torchvision==0.26.0
# nccl -> (2, 28, 9)