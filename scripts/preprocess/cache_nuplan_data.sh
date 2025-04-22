#!/usr/bin/env bash

# ========== Logging & Experiment Settings ==========
SAVE_DIR="YOUR_SAVE_DIR"                    # Where logs and checkpoints will be saved
EXPERIMENT="YOUR_EXPERIMENT_NAME"           # Experiment name
JOB_NAME="YOUR_JOB_NAME"                    # Job name used for logging/checkpoints

# ========== Cache Settings ==========
CACHE_DIR="YOUR_CACHE_DIR"                  # Path to cache directory

# ========== NuPlan Dataset Paths ==========
NUPLAN_SENSOR_ROOT="YOUR_NUPLAN_SENSOR_ROOT"  # Path to sensor blobs
NUPLAN_DATA_ROOT="YOUR_NUPLAN_DATA_ROOT"     # Path to train/val split data
NUPLAN_MAPS_ROOT="YOUR_NUPLAN_MAPS_ROOT"     # Path to map files

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
    cache.force_feature_computation=true \
    cache.versatile_caching=false \
    py_func=cache \
    +caching=cache_nuplan_nexus \
    scenario_builder=nuplan \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    scenario_builder.map_root=$NUPLAN_MAPS_ROOT \
    scenario_builder.sensor_root=$NUPLAN_SENSOR_ROOT \
    scenario_builder.scenario_mapping.subsample_ratio_override=0.5 \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=$NUM_WORKERS \
    model=nexus \
    scenario_filter.timestamp_threshold_s=15 \
    scenario_filter.expand_scenarios=false \
    scenario_filter.remove_invalid_goals=false 
    # +split=$SPLIT
