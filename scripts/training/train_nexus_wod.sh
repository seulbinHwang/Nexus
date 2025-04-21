#! /usr/bin/env bash
SAVE_DIR="/cpfs01/user/yenaisheng/SceneGen/logs"
EXPERIMENT="train"
JOB_NAME="wod"

CACHE_DIR=/cpfs01/shared/opendrivelab/yenaisheng/cache_waymo
CACHE_META_PATH=/cpfs01/shared/opendrivelab/yenaisheng/cache_waymo/trainval/metadata/trainval_metadata_node_0.csv

# specify wod path here
TRAINING_TOKEN_LIST_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl/training_token_list.txt
VALIDATION_TOKEN_LIST_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl/validation_token_list.txt

NUM_GPUS=8
BATCH_SIZE_PER_GPU=32
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))
NUM_WORKERS=12

export NUPLAN_DEVKIT_PATH=/cpfs01/user/yenaisheng/Nexus/third_party/nuplan-devkit
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

export WANDB_PROJECT="scenegen_wod"
export WANDB_EXP_NAME="waymo_3loss"
export WANDB_ENTITY="opendrivelab"

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

