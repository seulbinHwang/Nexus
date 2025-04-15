# Training and Evaluation Guide

## Training
To train on the Nuplan Dataset, execute the following command:
```bash
sh ./scripts/training/train_nexus_nuplan.sh
```

To train on the Waymo Open Dataset, execute the following command:
```bash
sh ./scripts/training/train_nexus_wod.sh
```

## Evaluation

### Waymo Sim Agents Benchmark

#### Steps to Fix Bugs in Waymo Open Dataset Package

1. **Install the Waymo Open Dataset Package**
   Make sure you have installed the Waymo Open Dataset package by running:
   ```bash
   pip install waymo-open-dataset-tf-2-12-0==1.6.4
   ```

2. **Modify the File**
   Edit the file located at `/usr/local/lib/python3.9/dist-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py`. Change the code at line 47 from:
   ```python
   config_path = '{pyglib_resource}waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto'.format(pyglib_resource='')
   ```
   to:
   ```python
   import os
   # Get the resource path from an environment variable
   pyglib_resource = os.getenv('PYGLIB_RESOURCE_PATH', '')

   # Construct the full config path
   config_path = f'{pyglib_resource}/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_config.textproto'
   ```

3. **Set Environment Variable**
   Set the environment variable for the resource path by running:
   ```bash
   export PYGLIB_RESOURCE_PATH=/usr/local/lib/python3.9/dist-packages
   ```

By following these steps, you will ensure that the configuration path is correctly set and the package functions properly.



#### Start Evaluation
To evaluate on the Waymo Sim Agents Benchmark, follow these steps:

1. Running the testing script:
    ```bash
    sh ./scripts/testing/test_nexus_wod_rollouts.sh
    ```

2. Running the offline evaluation script with multiprocessing
    ```bash
    python ./scripts/testing/offline_sim_agents.py --pkl_dir ${YOUR_PKL_DUMP_PATH} --nprocess 32 --output_dir ${OUTPUT_DIR}
    ```

### Metrics(e.g. FDE, ADE, ...) Evaluation
To evaluate specific metrics like FDE, ADE, Collision rate, etc., run the following command:
   ```bash
   sh ./scripts/testing/test_nexus_metrics.sh
   ```

## Checkpoints

We provide several checkpoints for model reproduction. To use a checkpoint, download it and replace the checkpoint path in the bash script:

```bash
+checkpoint.ckpt_path=PATH_TO_CHECKPOINT \
+checkpoint.strict=True \
```

### Checkpoint List
We provided the following CKPT:
| Model         | Dataset | Checkpoint                                                                                  |
|---------------|---------|---------------------------------------------------------------------------------------------|
| Nexus-nuplan     | NuPlan  |   [Google Cloud](https://storage.googleapis.com/93935945854-us-central1-blueprint-config/epoch_llama_sm.ckpt)                                                                  
| Nexus-wod  | Waymo Motion dataset  | [Google Cloud](https://storage.googleapis.com/93935945854-us-central1-blueprint-config/epoch_llama_sm.ckpt) |
