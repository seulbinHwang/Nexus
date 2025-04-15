#! /usr/bin/env bash
SAVE_DIR="/cpfs01/user/yenaisheng/SceneGen/logs"
EXPERIMENT="caching"
JOB_NAME="train_test"

# waymo motion dataset cache path
CACHE_META_PATH=/cpfs01/shared/opendrivelab/yenaisheng/cache_waymo/trainval/metadata/trainval_metadata_node_0.csv
CACHE_DIR="/cpfs01/shared/opendrivelab/yenaisheng/cache_waymo"
CACHE_META_PATH=/cpfs01/shared/opendrivelab/yenaisheng/wod_test_keep/metadata/test.csv
CACHE_DIR=/cpfs01/shared/opendrivelab/yenaisheng/wod_test_keep
NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))


WOD_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl
# set the token_list.txt path (validation_token_list or test_token_list) here to evaluate the model 
# later offline simulation only works for validation_token_list (as we only have gt future trajectories for validation set)
EVAL_TOKEN_LIST_PATH=/cpfs01/shared/opendrivelab/datasets/Waymo_motion/scenario_pkl/testing_token_list.txt

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

export USE_WANDB=False # False
export WANDB_PROJECT="scenegen_project"
export WANDB_EXP_NAME="waymo_val"
export WANDB_ENTITY="opendrivelab"

CHECKPOINT=/nas/shared/opendrivelab/zhouyunsong/nuplan/trainval/cache_nuPlan/training_world_model/training_world_model/2025.02.27.23.58.37/best_model/last.ckpt

# inference results which constain the 32 future trajectories for each sample will be saved under PKL_PATH, you can then run `./offline_sim_agents.py` to offline evaluate the results
PKL_PATH=/cpfs01/user/yenaisheng/test

# choose different constrain modes from :keep, clip, velocity, sma, collision, map (please refer to the nexus paper for more details)
# you can also set the force for each constrain mode in range of [0,1]
# example usage: model.diffuser.constrain_mode=[keep,clip] model.diffuser.constrain_gamma=[0.5,0.5] (or model.diffuser.constrain_gamma=0.5 for short)
# model will execute all constrains in the order of the constrain_mode list with the corresponding force 

# choose different scheduling_matrix from: pyramid, full_sequence, half_half_sequence, autoregressive, trapezoid (please refer to the nexus paper for more details)
# full_sequence will be used as default if you do not specify the scheduling_matrix
# example usage: model.diffuser.scheduling_matrix=pyramid

# choose which split of waymo motion dataset to evaluate (val or test)
# example usage: model.downstream_task=sim_agent_val for validation split, model.downstream_task=sim_agent_test for test split


python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.cache_metadata_path=$CACHE_META_PATH \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    experiment_name=$EXPERIMENT \
    py_func=test \
    seed=0 \
    +training=testing_wod_nexus \
    scenario_builder=wod_v1_1 \
    scenario_builder.data_root=$WOD_PATH \
    scenario_builder.subsample_ratio=1 \
    scenario_builder.start_index=0 \
    splitter.training_token_list_path=null \
    splitter.validation_token_list_path=$EVAL_TOKEN_LIST_PATH \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=1 \
    lightning.trainer.params.max_time=07:32:00:00\
    lightning.trainer.params.precision=16 \
    aggregated_metric.sim_agents_metrics.basepath=$PKL_PATH \
    data_loader.params.batch_size=$BATCH_SIZE_PER_GPU \
    data_loader.params.num_workers=$NUM_WORKERS \
    worker=single_machine_thread_pool \
    model=nexus_wod \
    +model.diffuser.constrain_mode=[] \
    +model.diffuser.constrain_gamma=1.0 \
    model.downstream_task=sim_agents_test \
    optimizer.lr=0.0002 \
    optimizer.weight_decay=0.0 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[8,10] \
    lightning.trainer.checkpoint.resume_training=false \
    scenario_filter=all_scenarios\
    +checkpoint.ckpt_path=$CHECKPOINT \
    +checkpoint.resume=True \
    +checkpoint.strict=True  