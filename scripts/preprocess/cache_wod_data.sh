#! /usr/bin/env bash

# logs saved under SAVE_DIR/EXPERIMENT/JOB_NAME
SAVE_DIR=/cpfs01/shared/opendrivelab/opendrivelab_hdd/zhouyunsong/nuplan/trainval/cache_nuPlan
EXPERIMENT=caching
JOB_NAME=wod

# data cached under CACHE_DIR
CACHE_DIR=/cpfs01/shared/opendrivelab/yenaisheng/wod_test

# specify wod path here
WOD_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl
TRAINING_TOKEN_LIST_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl/training_token_list.txt
VALIDATION_TOKEN_LIST_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl/validation_token_list.txt

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

export DEBUG=1

# To cache only a shard of the data, set SPLIT (e.g., 1/4, 2/4,..., 4/4) and uncomment the lines below, including the '+split=$SPLIT' argument
# SPLIT=1/4
# CACHE_DIR="$CACHE_DIR/cache_$(echo $SPLIT | sed 's/\//_/g')"
# echo "CURRENT SPLIT: $SPLIT"
# echo "CACHE_DIR: $CACHE_DIR"

python nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    experiment_name=$EXPERIMENT \
    job_name=$JOB_NAME \
    py_func=cache \
    +training=training_wod_nexus \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=36 \
    scenario_builder=wod_v1_1 \
    scenario_builder.data_root=$WOD_PATH \
    scenario_builder.training_token_list_path=$TRAINING_TOKEN_LIST_PATH \
    scenario_builder.validation_token_list_path=$VALIDATION_TOKEN_LIST_PATH \
    scenario_builder.subsample_ratio=1 \
    scenario_builder.start_index=0 \
    cache.force_feature_computation=true \
    cache.versatile_caching=false 
    # +split=$SPLIT
