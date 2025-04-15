import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from pathlib import Path
from hydra.utils import instantiate
import torch 
import numpy as np

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry,CacheResult,save_cache_metadata
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """
    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())
        cfg: DictConfig = args[0]["cfg"]
        if cfg.get('cache_adv',False):
            for arg in args:
                arg['scenario'].adv_info=arg['adv_info']
        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        model = build_torch_module_wrapper(cfg.model)
        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()

        # Now that we have the feature and target builders, we do not need the model any more.
        # Delete it so it gets gc'd and we can save a few system resources.
        del model

        # Create feature preprocessor
        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
        preprocessor = FeaturePreprocessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                "Processing scenario %s / %s in thread_id=%s, node_id=%s",
                idx + 1,
                len(scenarios),
                thread_id,
                node_id,
            )
            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)

            scenario_num_failures = sum(
                0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())
            )
            scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata

        logger.info("Finished processing scenarios for thread_id=%s, node_id=%s", thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result

def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker) 

def generate_adv_scenario_chunk(cfg,scenarios):
    adv_generator = instantiate(cfg.adv_generator)
    return scenarionet_cat(adv_generator,scenarios)

# process parallel version
def generate_adv_scenarios_v0(cfg , scenarios):
    # split the scenarios into chunks and assign them to different workers
    workers = cfg.convert_workers
    scenarios_chunk = chunk_list(scenarios, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        adv_scenarios = []
        logger.info(f"Generating adversarial scenarios for {len(scenarios)} scenarios on {workers} workers.")
        for scenario_chunk in scenarios_chunk:
            futures.append(executor.submit(generate_adv_scenario_chunk,cfg,scenario_chunk))
        logger.info(f"Waiting for adversarial scenario generation to complete...")
        for future in as_completed(futures):
            if future.result() is not None:
                adv_scenarios.extend(future.result())
    logger.info(f"Generated {len(adv_scenarios)}/{len(scenarios)} adversarial scenarios.")
    return adv_scenarios

def generate_adv_scenarios(cfg , scenarios):
    choice = cfg.get('gen_adv_mode','1')
    if choice == '0':
        return generate_adv_scenarios_v0(cfg,scenarios)
    elif choice == '1':
        return generate_adv_scenarios_v1(cfg,scenarios)
    else:
        raise NotImplementedError(f"Unsupported choice {choice} for adversarial scenario generation. 0: process parallel version, 1: thread parallel version")

# thread parallel version and sequential verison
def generate_adv_scenarios_v1(cfg , scenarios):
    # better be ThreadPoolExecutor since the CAT use pretrained model
    # import pdb; pdb.set_trace()
    devices = cfg.get('devices') if cfg.get('devices') else [0]
    if devices == 'all':
        devices = list(range(torch.cuda.device_count()))
    logger.info(f"Building AdvGenerator ...")
    adv_generators = []
    for device in devices:
        adv_generator = instantiate(cfg.adv_generator)
        adv_generator.device = device
        adv_generators.append(adv_generator)
    logger.info(f'Building AbstractScenarioBuilder...DONE! AdvGenerators:{devices}')

    # parallel version
    # split the scenarios into chunks and assign them to different workers
    workers = cfg.convert_workers
    scenarios_chunk = chunk_list(scenarios, workers)
    with ThreadPoolExecutor(max_workers=cfg.convert_workers) as executor:
        futures = []
        adv_scenarios = []
        num = len(adv_generators)
        logger.info(f"Generating adversarial scenarios for {len(scenarios)} scenarios on {cfg.convert_workers} workers.")
        for i,scenarios_ in enumerate(scenarios_chunk):
            futures.append(executor.submit(scenarionet_cat,adv_generators[i%num],scenarios_))
        logger.info(f"Waiting for adversarial scenario generation to complete...")
        for future in as_completed(futures):
            if future.result() is not None:
                adv_scenarios.extend(future.result())
    logger.info(f"Generated {len(adv_scenarios)}/{len(scenarios)} adversarial scenarios.")

    # sequential version
    # adv_scenarios = []
    # num = len(adv_generators)
    # for i,scenario in tqdm(enumerate(scenarios)):
    #     adv_scenarios.extend(scenarionet_cat(adv_generators[i%num],[scenario]))
    # logger.info(f"Generated {len(adv_scenarios)}/{len(scenarios)} adversarial scenarios.")

    return adv_scenarios

def scenarionet_cat(adv_generator,scenarios,toleration=1,dataset_version='v1.1'):
    adv_scenarios=[]
    pbar=tqdm(total=len(scenarios))
    pbar.set_description(f"[Success rate: {len(adv_scenarios)}/{len(scenarios)}]")
    for scenario in scenarios: 
        toleration_ = toleration
        env = convert_nuplan_scenario(scenario,dataset_version)
        with torch.no_grad():
            while toleration_>0:
                try:
                    adv_generator.before_episode(env)  # initialization before each episode
                    adv_info,collision=adv_generator.generate(env)  # Adversarial scenario generation with the logged history corresponding to the current env
                except Exception as e:
                    logger.error(f"Error processing: {e}")
                    break
                if collision:
                    scenario.adv_info = adv_info
                    adv_scenarios.append(scenario)
                    break 
                toleration_-=1        
        del env
        # display progress bar and success rate len(adv_scenarios)/len(scenarios)
        pbar.update(1)
        pbar.set_description(f"Success rate: {len(adv_scenarios)}/{len(scenarios)}")
    # gc.collect()
    # torch.cuda.empty_cache()
    return adv_scenarios if adv_scenarios else None
    
def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
    scenario_builder = build_scenario_builder(cfg)
     
    logger.debug(
        "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
        "accidentally caching each scenario multiple times!"
    )
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    
    # import pdb; pdb.set_trace()
    split = cfg.get('split',None)
    if split:  # split is a string, e.g. '3/5'
        split = split.split('/')
        split = [int(split[0]),int(split[1])]
        scenarios = np.array_split(scenarios,split[1])[split[0]-1].tolist()
    
    logger.info(f"{len(scenarios)} scenarios have been built.")

    if cfg.get('cache_adv',False):
        from third_party.scenarionet.scenarionet.converter.nuplan.utils import convert_nuplan_scenario
        from third_party.CAT.advgen.adv_generator_nuplan import AdvGenerator
        scenarios = generate_adv_scenarios(cfg,scenarios)
        data_points = [{"scenario": scenario, "cfg": cfg, "adv_info":scenario.adv_info} for scenario in scenarios]
    else:
        data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]

    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    logger.info("Completed dataset caching! Successed features and targets: %s out of %s", str(num_success), str(num_total))
    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]            
    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info("Done storing metadata csv file.")