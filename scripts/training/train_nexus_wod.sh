#!/usr/bin/env bash

# ========== Logging & Experiment Settings ==========
SAVE_DIR="YOUR_SAVE_DIR"                    # Where logs and checkpoints will be saved
EXPERIMENT="YOUR_EXPERIMENT_NAME"           # Experiment name
JOB_NAME="YOUR_JOB_NAME"                    # Job name used for logging/checkpoints

# ========== Cache Settings ==========
CACHE_DIR="YOUR_CACHE_DIR"                  # Path to cache directory
CACHE_META_PATH="YOUR_CACHE_META_PATH"      # Path to cache metadata CSV

# ========== Waymo Dataset Paths ==========
TRAINING_TOKEN_LIST_PATH="YOUR_TRAINING_TOKEN_LIST_PATH"  # Path to training token list
VALIDATION_TOKEN_LIST_PATH="YOUR_VALIDATION_TOKEN_LIST_PATH"  # Path to validation token list

# ========== Training Configuration ==========
NUM_GPUS=YOUR_NUM_GPUS                     # Number of GPUs
BATCH_SIZE_PER_GPU=YOUR_BATCH_SIZE         # Batch size per GPU
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))    # Adjust workers based on hardware
NUM_WORKERS=YOUR_NUM_WORKERS              # Manually adjust if needed

# ========== NuPlan Paths ==========
export NUPLAN_DEVKIT_PATH="YOUR_NUPLAN_DEVKIT_PATH"      # Path to nuplan-devkit repo
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

# ========== OpenBLAS and OpenMP Settings ==========
export OPENBLAS_NUM_THREADS=1    # To avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1         # Control the number of threads per process for OpenMP

# ========== Weights & Biases ==========
export WANDB_PROJECT="YOUR_WANDB_PROJECT"
export WANDB_EXP_NAME="YOUR_WANDB_EXP_NAME"
export WANDB_ENTITY="YOUR_WANDB_ENTITY"

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.cache_metadata_path=$CACHE_META_PATH \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    experiment_name=$EXPERIMENT \
    job_name=$JOB_NAME \
    py_func=train \
    seed=0 \
    +training=training_wod_nexus \
    scenario_builder=wod_v1_1 \
    scenario_builder.training_token_list_path=$TRAINING_TOKEN_LIST_PATH \
    scenario_builder.validation_token_list_path=$VALIDATION_TOKEN_LIST_PATH \
    scenario_builder.subsample_ratio=1 \
    scenario_builder.start_index=0 \
    lightning.trainer.params.fast_dev_run=true\
    lightning.trainer.params.num_sanity_val_steps=0 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    lightning.trainer.params.max_epochs=350 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.gradient_clip_val=1.0 \
    lightning.trainer.params.strategy=ddp \
    lightning.trainer.params.detect_anomaly=false \
    lightning.trainer.params.log_every_n_steps=10\
    lightning.trainer.checkpoint.monitor=loss/train_loss \
    +lightning.trainer.overfitting.enable=false \
    +lightning.trainer.overfitting.params.overfit_batches=0 \
    +lightning.trainer.params.val_check_interval=1.0 \
    lightning.trainer.params.accumulate_grad_batches=$NUM_ACCUM_BATCHES\
    data_loader.params.batch_size=$BATCH_SIZE_PER_GPU \
    data_loader.params.num_workers=$NUM_WORKERS \
    data_loader.params.pin_memory=true \
    worker=single_machine_thread_pool \
    model=nexus_wod \
    model.downstream_task=null \
    optimizer=adamw \
    optimizer.lr=1e-3 \
    optimizer.weight_decay=0.01 \
    scenario_filter=all_scenarios \
    lr_scheduler=warmup_cos_lr \
    lightning.trainer.checkpoint.resume_training=true \
    scenario_filter=training_scenarios\
    +checkpoint.resume=true \
    +checkpoint.strict=true \
    +checkpoint.ckpt_path=null

