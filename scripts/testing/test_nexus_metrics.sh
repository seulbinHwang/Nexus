#!/usr/bin/env bash

# ========== Logging & Experiment Settings ==========
SAVE_DIR="YOUR_SAVE_DIR"              # Where logs and checkpoints will be saved
EXPERIMENT="YOUR_EXPERIMENT_NAME"     # Experiment name
JOB_NAME="YOUR_JOB_NAME"              # Job name used for logging/checkpoints

# ========== Cache Settings ==========
CACHE_DIR="YOUR_CACHE_DIR"            # Path to cache directory
CACHE_META_PATH="YOUR_CACHE_META_PATH"  # Path to cache metadata CSV

# ========== NuPlan Dataset Paths ==========
export NUPLAN_DEVKIT_PATH="YOUR_NUPLAN_DEVKIT_PATH"       # Path to nuplan-devkit repo
export NUPLAN_SENSOR_ROOT="YOUR_SENSOR_BLOBS_PATH"        # Path to sensor blobs
export NUPLAN_DATA_ROOT="YOUR_DATA_ROOT_PATH"             # Path to train/val split data
export NUPLAN_MAPS_ROOT="YOUR_MAPS_ROOT_PATH"             # Path to map files

# ========== Training Configuration ==========
NUM_GPUS=YOUR_NUM_GPUS
BATCH_SIZE_PER_GPU=YOUR_BATCH_SIZE
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=YOUR_NUM_WORKERS

# ========== Python Environment ==========
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

# ========== Weights & Biases ==========
export WANDB_PROJECT="YOUR_WANDB_PROJECT"        # Set your W&B project name here
export WANDB_EXP_NAME="YOUR_WANDB_EXP_NAME"      # Name of the experiment
export WANDB_ENTITY="YOUR_WANDB_ENTITY"          # Your W&B entity or username

# ========== Model Checkpoint ==========
CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH"           # Replace with actual checkpoint path

# ========== Constrain Mode Settings ==========
# Choose different constrain modes from: keep, clip, velocity, sma, collision, map (please refer to the nexus paper for more details)
# You can also set the force for each constrain mode in the range of [0,1]
# Example usage: +model.diffuser.constrain_mode=[keep,clip], +model.diffuser.constrain_gamma=[0.5,0.5] (or model.diffuser.constrain_gamma=0.5 for short)
# The model will execute all constraints in the order of the constrain_mode list with the corresponding force 

# ========== Scheduling Matrix ==========
# Choose different scheduling_matrix from: pyramid, full_sequence, half_half_sequence, autoregressive, trapezoid (please refer to the nexus paper for more details)
# Full_sequence will be used as default if you do not specify the scheduling_matrix
# Example usage: model.diffuser.scheduling_matrix=pyramid

# ========== Waymo Motion Dataset Split ==========
# Choose which split of Waymo Motion dataset to evaluate (val or test)
# Example usage: model.downstream_task=sim_agent_val for validation split, model.downstream_task=sim_agent_test for test split

# the script will evaluate the model on full validation set after one-sample training
python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.cache_metadata_path=$CACHE_META_PATH \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    cache.versatile_caching=false \
    experiment_name=$EXPERIMENT \
    job_name=$JOB_NAME \
    py_func=train \
    seed=0 \
    +training=testing_nuplan_nexus \
    scenario_builder=nuplan \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    lightning.trainer.params.num_sanity_val_steps=0 \
    lightning.trainer.params.fast_dev_run=false\
    lightning.trainer.params.max_epochs=350 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.gradient_clip_val=1.0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    lightning.trainer.params.detect_anomaly=false \
    lightning.trainer.params.log_every_n_steps=10\
    lightning.trainer.checkpoint.monitor=loss/train_loss \
    +lightning.trainer.overfitting.enable=false \
    +lightning.trainer.overfitting.params.overfit_batches=0 \
    +lightning.trainer.params.val_check_interval=1 \
    lightning.trainer.params.accumulate_grad_batches=$NUM_ACCUM_BATCHES\
    data_loader.params.batch_size=$BATCH_SIZE_PER_GPU \
    data_loader.params.num_workers=$NUM_WORKERS \
    data_loader.params.pin_memory=true \
    worker=single_machine_thread_pool \
    model=nexus \
    +model.diffuser.constrain_mode=['map','sma','clip'] \
    +model.diffuser.constrain_gamma=[1.0,1.0,1.0] \
    +model.diffuser.scheduling_matrix=full_sequence \
    optimizer=adamw \
    optimizer.lr=1e-3 \
    optimizer.weight_decay=0.01 \
    scenario_filter=all_scenarios \
    lr_scheduler=warmup_cos_lr \
    +lightning.trainer.resume_training=true \
    +checkpoint.ckpt_path=$CHECKPOINT_PATH \
    +checkpoint.resume=true \
    +checkpoint.strict=true
    # +checkpoint.ckpt_path=null