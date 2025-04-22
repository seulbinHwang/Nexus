#!/usr/bin/env bash

# ========== Logging & Experiment Settings ==========
SAVE_DIR="YOUR_SAVE_DIR"                    # Where logs and checkpoints will be saved
EXPERIMENT="YOUR_EXPERIMENT_NAME"           # Experiment name
JOB_NAME="YOUR_JOB_NAME"                    # Job name used for logging/checkpoints

# ========== Waymo Motion Dataset Cache ==========
CACHE_META_PATH="YOUR_CACHE_META_PATH"      # Path to metadata CSV
CACHE_DIR="YOUR_CACHE_DIR"                  # Path to cached data

# ========== Training Configuration ==========
NUM_GPUS=YOUR_NUM_GPUS
BATCH_SIZE_PER_GPU=YOUR_BATCH_SIZE
NUM_ACCUM_BATCHES=$((1024 / BATCH_SIZE_PER_GPU / NUM_GPUS))
NUM_WORKERS=$((BATCH_SIZE * NUM_GPUS))      # Adjust based on hardware

# ========== Waymo Data Path ==========
WOD_PATH="YOUR_WAYMO_SCENARIO_PATH"         # Path to Waymo .pkl scenario files

# ========== Evaluation Token List ==========
EVAL_TOKEN_LIST_PATH="YOUR_EVAL_TOKEN_LIST_PATH"  # Validation or test token list

# ========== Environment Variables ==========
export NUPLAN_DEVKIT_PATH="YOUR_NUPLAN_DEVKIT_PATH"    # nuplan-devkit path
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1       # Avoid OpenBLAS thread explosion
export OMP_NUM_THREADS=1            # Control OpenMP threads

# ========== Weights & Biases ==========
export WANDB_PROJECT="YOUR_WANDB_PROJECT"
export WANDB_EXP_NAME="YOUR_WANDB_EXP_NAME"
export WANDB_ENTITY="YOUR_WANDB_ENTITY"

# ========== Model Checkpoint ==========
CHECKPOINT="YOUR_MODEL_CHECKPOINT_PATH"     # Path to trained model checkpoint

# ========== Output Path ==========
PKL_PATH="YOUR_OUTPUT_PKL_SAVE_PATH"        # Where inference results will be stored

# ========== Constrain Mode Settings ==========
# Choose different constrain modes from: keep, clip, velocity, sma, collision, map (please refer to the nexus paper for more details)
# You can also set the force for each constrain mode in the range of [0,1]
# Example usage: model.diffuser.constrain_mode=[keep,clip], model.diffuser.constrain_gamma=[0.5,0.5] (or model.diffuser.constrain_gamma=0.5 for short)
# The model will execute all constraints in the order of the constrain_mode list with the corresponding force 

# ========== Scheduling Matrix ==========
# Choose different scheduling_matrix from: pyramid, full_sequence, half_half_sequence, autoregressive, trapezoid (please refer to the nexus paper for more details)
# Full_sequence will be used as default if you do not specify the scheduling_matrix
# Example usage: model.diffuser.scheduling_matrix=pyramid

# ========== Waymo Motion Dataset Split ==========
# Choose which split of Waymo Motion dataset to evaluate (val or test)
# Example usage: model.downstream_task=sim_agent_val for validation split, model.downstream_task=sim_agent_test for test split

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
    model.downstream_task=sim_agents_val \
    optimizer.lr=0.0002 \
    optimizer.weight_decay=0.0 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[8,10] \
    lightning.trainer.checkpoint.resume_training=false \
    scenario_filter=all_scenarios\
    +checkpoint.ckpt_path=$CHECKPOINT \
    +checkpoint.resume=True \
    +checkpoint.strict=True  