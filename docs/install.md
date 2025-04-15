# Installation

## Setup conda environment

    ```python
    conda create -n nexus python=3.9
    conda activate nexus

    git clone https://github.com/OpenDriveLab/Nexus.git && cd Nexus
    pip install -r requirements.txt
    ```

## Install Nuplan Devkit

1. Navigate to the `third_party` directory:
   ```bash
   cd Nexus/third_party
   ```

2. Clone the Nuplan Devkit repository:
   ```bash
   git clone https://github.com/motional/nuplan-devkit.git
   ```

3. Install the Nuplan Devkit:
   ```bash
   cd nuplan-devkit
   pip install -e .
   ```

4. Set the `NUPLAN_DEVKIT_PATH` environment variable:
   ```bash
   export NUPLAN_DEVKIT_PATH=$YOUR_PATH_TO_Nexus/Nexus/third_party/nuplan-devkit
   ```
5. install following packages:
   ```bash
   pip install aiofiles aioboto3 flatdict adjustText loralib easydict einops_exts
   pip install waymo-open-dataset-tf-2-12-0==1.6.4
   ```

## Install ALF

Follow these steps to install ALF while ignoring the dependencies of torch:

1. Navigate to the `third_party` directory:

    ```bash
    cd Nexus/third_party
    ```

2. Clone the ALF repository:

    ```bash
    git clone https://github.com/HorizonRobotics/alf.git
    ```

3. Edit the `setup.py` file to ignore the dependencies of torch:

    Comment out lines 52, 53, and 54 in the `setup.py` file:

    ```python
    # 'torch==2.2.0',
    # 'torchvision==0.17.0',
    # 'torchtext==0.17.0',
    ```

4. Install ALF:

    ```bash
    pip install -e .
    ```