# Installation

## Setup conda environment

    ```python
    conda create -n nexus python=3.9
    conda activate nexus

    git clone https://github.com/OpenDriveLab/Nexus.git && cd Nexus
    pip install -r requirements.txt
    ```

## Install ALF

To simplify installation, we directly provide a stable version of ALF here. Follow these steps to install [ALF](https://github.com/HorizonRobotics/alf):

1. Navigate to the `third_party/alf` directory:

    ```bash
    cd Nexus/third_party/alf
    ```

2. Install ALF:

    ```bash
    pip install -e .
    ```

## Install Nuplan Devkit

1. Navigate to the `third_party` directory:
   ```bash
   cd Nexus/third_party
   ```

2. Clone the Nuplan Devkit repository:
   ```bash
   git clone --branch feat-v1.3_gump https://github.com/HorizonRobotics/nuplan-devkit.git
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
   
## Install MTR
1. Navigate to the `third_party` directory:
   ```bash
   cd Nexus/third_party
   ```

2. Clone the MTR repository:
   ```bash
   git clone https://github.com/sshaoshuai/MTR.git
   ```

3. Install the MTR:
   ```bash
   cd MTR
   pip install -e .
   ```