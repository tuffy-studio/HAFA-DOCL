#!/bin/bash

# ==================== Paths ====================

pretrain_path=""
tr_data=""
te_data=""
save_dir=""
save_model=True

mkdir -p $save_dir
mkdir -p ${save_dir}/models

# ==================== Training Params ====================

lr=1e-5
head_lr_ratio=10
n_epochs=50
batch_size=512
num_workers=32

use_amp=True
verbose=True
use_hierarchical=True
moe=False
encoder_embed_dim=768
restart=False

# ==================== Run ====================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_CACHE_DISABLE=1
ulimit -n 65535

torchrun --nproc_per_node=4  ../src/run_finetune_ddp.py \
    --data_train ${tr_data} \
    --data_val ${te_data} \
    --save_dir ${save_dir} \
    --lr ${lr} \
    --head_lr_ratio ${head_lr_ratio} \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    $( [ "$use_amp" = "True" ] && echo "--use_amp" ) \
    $( [ "$verbose" = "True" ] && echo "--verbose" ) \
    $( [ "$restart" = "True" ] && echo "--restart" ) \
    $( [ "$use_hierarchical" = "True" ] && echo "--use_hierarchical" ) \
    $( [ "$moe" = "True" ] && echo "--moe" ) \
    $( [ "$save_model" = "True" ] && echo "--save_model" ) \
    --pretrain_path ${pretrain_path} \
    --encoder_embed_dim ${encoder_embed_dim}
