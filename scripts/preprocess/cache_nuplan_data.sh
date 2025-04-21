#! /usr/bin/env bash

# logs saved under SAVE_DIR/EXPERIMENT/JOB_NAME
SAVE_DIR=/cpfs01/shared/opendrivelab/opendrivelab_hdd/zhouyunsong/nuplan/trainval/cache_nuPlan
EXPERIMENT=caching
JOB_NAME=nuplan

# data will be cached under CACHE_DIR
CACHE_DIR=/cpfs01/shared/opendrivelab/yenaisheng/nuplan_test

# specify nuplan dataset paths here
NUPLAN_SENSOR_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/nuplan-v1.1/sensor_blobs
NUPLAN_DATA_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/nuplan-v1.1/splits/trainval
NUPLAN_MAPS_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/maps

export NUPLAN_DEVKIT_PATH=/cpfs01/user/yenaisheng/Nexus/third_party/nuplan-devkit
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

# To cache only a shard of the data, set SPLIT (e.g., 1/4, 2/4,..., 4/4) and uncomment the lines below, including the '+split=$SPLIT' argument
# SPLIT=1/4
# CACHE_DIR="$CACHE_DIR/cache_$(echo $SPLIT | sed 's/\//_/g')"
# echo "CURRENT SPLIT: $SPLIT"
# echo "CACHE_DIR: $CACHE_DIR"

# number of workers for caching
NUM_WORKERS=1

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
