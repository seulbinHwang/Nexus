#!/usr/bin/env bash

# ========== Logging & Experiment Settings ==========
SAVE_DIR="YOUR_SAVE_DIR"                      # Where logs and checkpoints will be saved
EXPERIMENT="YOUR_EXPERIMENT_NAME"             # Experiment name
JOB_NAME="YOUR_JOB_NAME"                      # Job name used for logging/checkpoints

# ========== Cache Settings ==========
CACHE_DIR="YOUR_CACHE_DIR"                    # Path to cache directory

# ========== Waymo Dataset Paths ==========
WOD_PATH="YOUR_WOD_PATH"                      # Path to Waymo dataset
TRAINING_TOKEN_LIST_PATH="YOUR_TRAINING_TOKEN_LIST_PATH"  # Path to training token list
VALIDATION_TOKEN_LIST_PATH="YOUR_VALIDATION_TOKEN_LIST_PATH"  # Path to validation token list

# ========== NuPlan Devkit Path ==========
export NUPLAN_DEVKIT_PATH="YOUR_NUPLAN_DEVKIT_PATH"  # Path to nuplan-devkit repo

# ========== Python Environment ==========
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

# ========== OpenBLAS and OpenMP Settings ==========
export OPENBLAS_NUM_THREADS=1    # To avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1         # Control the number of threads per process for OpenMP

# ========== Data Split (Optional) ==========
# Uncomment the following lines if you need to use data splits
# SPLIT="YOUR_SPLIT"  # e.g., "1/4", "2/4", etc.
# CACHE_DIR="$CACHE_DIR/cache_$(echo $SPLIT | sed 's/\//_/g')"
# echo "CURRENT SPLIT: $SPLIT"
# echo "CACHE_DIR: $CACHE_DIR"

# ========== Worker Configuration ==========
NUM_WORKERS="YOUR_NUM_WORKERS"  # Number of workers for caching

python nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    experiment_name=$EXPERIMENT \
    job_name=$JOB_NAME \
    py_func=cache \
    +training=training_wod_nexus \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=$NUM_WORKERS \
    scenario_builder=wod_v1_1 \
    scenario_builder.data_root=$WOD_PATH \
    scenario_builder.training_token_list_path=$TRAINING_TOKEN_LIST_PATH \
    scenario_builder.validation_token_list_path=$VALIDATION_TOKEN_LIST_PATH \
    scenario_builder.subsample_ratio=1 \
    scenario_builder.start_index=0 \
    cache.force_feature_computation=true \
    cache.versatile_caching=false 
    # +split=$SPLIT
