import logging
import os
import warnings
from typing import Optional

import cv2
import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.training_builder import (
    build_lightning_datamodule,
    build_lightning_module,
    build_trainer,
)
from nuplan.planning.script.builders.utils.utils_config import (
    update_config_for_training,
)
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan_extent.planning.training.caching.caching import cache_data
from nuplan.planning.training.experiments.training import (
    TrainingEngine,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger


cv2.setNumThreads(1)
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy("file_system")
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and
# experiment paths
set_default_path()

# Add a new resolver that supports eval
OmegaConf.register_new_resolver("eval", eval)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config")

CONFIG_NAME = os.getenv("DEFAULT_CONFIG", "horizon_training")


def build_custom_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info("Building training engine...")

    # Build trainer
    trainer = build_trainer(cfg)

    # Force creating a model directly on the target device with the desired precision
    with trainer.init_module():
        # Create model
        torch_module_wrapper = build_torch_module_wrapper(cfg.model)
        # Build lightning module
        model = build_lightning_module(cfg, torch_module_wrapper)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine
print(CONFIG_PATH, CONFIG_NAME)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    # Build plugins (compatible with mmdet)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_path = _module_dir.replace("/", ".")
            if _module_path.startswith("."):
                _module_path = _module_path[1:]
            logger.info(f"Plugin directory: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

    if cfg.py_func == "train":
        # Build training engine
        engine = build_custom_training_engine(cfg, worker)
        # import pdb; pdb.set_trace()
        # Run training
        logger.info("Starting training...")
        # OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
        if os.environ.get("USE_WANDB", 'true').lower() == 'true':
            # check if wandb is enabled
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                training_logger = WandbLogger(
                    save_dir=os.environ.get("WANDB_DIR", "wandb/"),
                    project=os.environ.get("WANDB_PROJECT", "your_project_name"),
                    name=os.environ.get("WANDB_EXP_NAME", "your_experiment_name"),
                    entity=os.environ.get("WANDB_ENTITY", "your_wandb_entity"),
                    log_model=False,
                    offline=os.environ.get("WANDB_OFFLINE", 'false').lower() == 'true',
                )

                engine.trainer.logger = training_logger

        my_ckpt_path = cfg.checkpoint.ckpt_path
        if my_ckpt_path is not None:
            assert isinstance(my_ckpt_path, str), "Checkpoint path must be a string"
            assert os.path.exists(
                my_ckpt_path
            ), f"Checkpoint path {my_ckpt_path} does not exist"

            my_ckpt = torch.load(my_ckpt_path, map_location="cpu")
            engine.model.load_state_dict(my_ckpt["state_dict"], strict=True)
        # Load model state dict 
        engine.trainer.fit(
            model=engine.model,
            datamodule=engine.datamodule,
        )
        return engine
    
    elif cfg.py_func == "distill":
        # Build training engine
        engine = build_custom_training_engine(cfg, worker)

        # Run training
        logger.info("Starting distill...")
        # OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

        if os.environ.get("USE_WANDB", 'true').lower() == 'true':
            # check if wandb is enabled
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                training_logger = WandbLogger(
                    save_dir=os.environ.get("WANDB_DIR", "wandb/"),
                    project=os.environ.get("WANDB_PROJECT", "your_project_name"),
                    name=os.environ.get("WANDB_EXP_NAME", "your_experiment_name"),
                    entity=os.environ.get("WANDB_ENTITY", "your_wandb_entity"),
                    log_model=False,
                    offline=os.environ.get("WANDB_OFFLINE", 'false').lower() == 'true',
                )

                engine.trainer.logger = training_logger

        my_ckpt_path = cfg.checkpoint.ckpt_path
        assert isinstance(my_ckpt_path, str), "Checkpoint path must be a string"
        assert os.path.exists(
            my_ckpt_path
        ), f"Checkpoint path {my_ckpt_path} does not exist"

        my_ckpt = torch.load(my_ckpt_path, map_location="cpu")
        # Load model state dict with strict=False to ignore LoRA parameters
        missing_keys=engine.model.load_state_dict(my_ckpt["state_dict"], strict=False).missing_keys
        # only activate LoRA parameters
        for name, param in engine.model.named_parameters():
            if name in missing_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        engine.trainer.fit(
            model=engine.model,
            datamodule=engine.datamodule,
        )
        return engine

    elif cfg.py_func == "test":
        # Build training engine
        engine = build_custom_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")

        my_ckpt_path = cfg.checkpoint.ckpt_path
        assert isinstance(my_ckpt_path, str), "Checkpoint path must be a string"
        assert os.path.exists(
            my_ckpt_path
        ), f"Checkpoint path {my_ckpt_path} does not exist"

        my_ckpt = torch.load(my_ckpt_path, map_location="cpu")
        engine.model.load_state_dict(my_ckpt["state_dict"])

        if os.environ.get("USE_WANDB", 'true').lower() == 'true':
            # check if wandb is enabled
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                training_logger = WandbLogger(
                    save_dir=os.environ.get("WANDB_DIR", "wandb/"),
                    project=os.environ.get("WANDB_PROJECT", "your_project_name"),
                    name=os.environ.get("WANDB_EXP_NAME", "your_experiment_name"),
                    entity=os.environ.get("WANDB_ENTITY", "your_wandb_entity"),
                    log_model=False,
                    offline=os.environ.get("WANDB_OFFLINE", 'false').lower() == 'true',
                )

                engine.trainer.logger = training_logger
 
        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine

    elif cfg.py_func == "cache":
        # Precompute and cache all features
        cache_data(cfg=cfg, worker=worker)
        return None

    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()