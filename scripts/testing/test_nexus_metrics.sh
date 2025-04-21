#!usr/bin/env bash
SAVE_DIR="/cpfs01/user/yenaisheng/SceneGen/logs"
EXPERIMENT="caching"
JOB_NAME="train_test"

CACHE_DIR=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_tokenized_log
CACHE_META_PATH=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_tokenized_log/metadata/cache_tokenized_log_metadata_node_0.csv

export NUPLAN_DEVKIT_PATH=/cpfs01/user/yenaisheng/Nexus/third_party/nuplan-devkit
export NUPLAN_SENSOR_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/nuplan-v1.1/sensor_blobs
export NUPLAN_DATA_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/nuplan-v1.1/splits/trainval
export NUPLAN_MAPS_ROOT=/nas/shared/opendrivelab/dataset/datasets/nuplan/maps

NUM_GPUS=8
BATCH_SIZE_PER_GPU=64
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))
NUM_WORKERS=12

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

export WANDB_PROJECT="nexus"
export WANDB_EXP_NAME="constrain"
export WANDB_ENTITY="opendrivelab"


CHECKPOINT_PATH=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_nuPlan/training_world_model/training_world_model/2025.02.27.23.58.37/best_model/last.ckpt

# choose different constrain modes from :keep, clip, velocity, sma, collision, map (please refer to the nexus paper for more details)
# you can also set the force for each constrain mode in range of [0,1]
# example usage: +model.diffuser.constrain_mode=[keep,clip], +model.diffuser.constrain_gamma=[0.5,0.5] (or model.diffuser.constrain_gamma=0.5 for short)
# model will execute all constrains in the order of the constrain_mode list with the corresponding force 

# choose different scheduling_matrix from: pyramid, full_sequence, half_half_sequence, autoregressive, trapezoid (please refer to the nexus paper for more details)
# full_sequence will be used as default if you do not specify the scheduling_matrix
# example usage: model.diffuser.scheduling_matrix=pyramid

# choose which split of waymo motion dataset to evaluate (val or test)
# example usage: model.downstream_task=sim_agent_val for validation split, model.downstream_task=sim_agent_test for test split

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