#! /usr/bin/env bash
SAVE_DIR="/cpfs01/user/yenaisheng/SceneGen/logs"
EXPERIMENT="train"
JOB_NAME="nuplan"

CACHE_DIR=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_tokenized_log
CACHE_META_PATH=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_tokenized_log/metadata/cache_tokenized_log_metadata_node_0.csv

export NUPLAN_SENSOR_ROOT=/cpfs01/shared/opendrivelab/datasets/nuplan/dataset/nuplan-v1.1/sensor_blobs
export NUPLAN_DATA_ROOT=/cpfs01/shared/opendrivelab/datasets/nuplan/dataset/nuplan-v1.1/splits/trainval
export NUPLAN_MAPS_ROOT=/cpfs01/shared/opendrivelab/datasets/nuplan/dataset/maps

NUM_GPUS=8
BATCH_SIZE_PER_GPU=64
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))
NUM_WORKERS=12

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

export USE_WANDB=False # True False
export WANDB_PROJECT="scenegen_wod"
export WANDB_EXP_NAME="nuplan_wophytime"
export WANDB_ENTITY="opendrivelab"

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
    +training=training_nuplan_nexus \
    scenario_builder=nuplan \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    lightning.trainer.params.max_epochs=350 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.gradient_clip_val=1.0 \
    lightning.trainer.params.num_sanity_val_steps=0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    lightning.trainer.params.fast_dev_run=true\
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
    model=nexus \
    optimizer=adamw \
    optimizer.lr=1e-3 \
    optimizer.weight_decay=0.01 \
    scenario_filter=all_scenarios \
    lr_scheduler=warmup_cos_lr \
    +lightning.trainer.resume_training=true \
    +checkpoint.resume=true \
    +checkpoint.strict=true \
    +checkpoint.ckpt_path=null