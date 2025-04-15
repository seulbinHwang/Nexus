from typing import List
import copy

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class OverfitSplitter(AbstractSplitter):
    """
    Splitter that splits database to lists of samples for each of train/val/test
    sets based on the split attribute of the scenario.
    """

    def __init__(
        self, number_of_train_samples: int = 1000, scenario_num: int = 0
    ) -> None:
        """
        Initializes the class.

        """
        self._number_of_train_samples = number_of_train_samples
        self._scenario_num = scenario_num

    def get_train_samples(
        self, scenarios: List[AbstractScenario], worker: WorkerPool
    ) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [
            copy.deepcopy(scenarios[self._scenario_num])
            for _ in range(self._number_of_train_samples)
        ]

    def get_val_samples(
        self, scenarios: List[AbstractScenario], worker: WorkerPool
    ) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [scenarios[self._scenario_num]]

    def get_test_samples(
        self, scenarios: List[AbstractScenario], worker: WorkerPool
    ) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [scenarios[self._scenario_num]]
