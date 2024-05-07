import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import config
from utils.data_processor import DataProcessor
from utils.common_utils import (
    get_scenario_map,
    get_filter_parameters
)

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

if __name__ == '__main__':
    # nuplan arguments
    scenarios_per_type = 1000
    total_scenarios = None
    shuffle_scenarios = False

    sensor_root = None
    db_files = None

    # create folder for processed data
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    # get scenarios
    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(), 
        subsample_ratio_override=0.5
    )

    print("Building scenario ...")
    builder = NuPlanScenarioBuilder(
        config.DATA_PATH, 
        config.MAP_PATH, 
        sensor_root, 
        db_files, 
        config.MAP_VERSION, 
        scenario_mapping=scenario_mapping
    )

    # scenarios for training
    print("Filtering scenario ...")
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            scenarios_per_type, 
            total_scenarios, 
            shuffle_scenarios
        )
    )

    # enable parallel process
    print("Enabling parallel executor ...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    # get scenarios
    print("Getting scenarios ...")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    
    # Process scenarios
    print("Processing ...")
    print("Number of scenarios: ", len(scenarios))
    
    count = 0
    for scenario in tqdm(scenarios):
        # process data
        data_processor = DataProcessor(scenario)
        data, features = data_processor.process(debug=False)
        
        count = count + 1