import os
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm

from . import config
from .common_utils import (
    get_scenario_map,
    get_filter_parameters
)

from .data_utils import (
    sampled_past_ego_states_to_tensor,
    get_neighbor_vector_set_map,
    sampled_tracked_objects_to_tensor_list,
    map_process,
    agent_past_process,
    agent_future_process,
    encoding_vectornet_features,
    save_features,
    create_map_raster,
    create_ego_raster,
    create_agents_raster,
    draw_trajectory
)

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    AgentFeatureIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states
)

class DataProcessor(object):
    def __init__(self, scenario):
        self.scenario = scenario
        self.map_api = scenario.map_api        
        
        self.discrete_size       = config.DISCRETE_SIZE
        self.past_time_horizon   = config.PAST_TIME_HORIZON
        self.num_past_poses      = config.NUM_PAST_POSES
        self.future_time_horizon = config.FUTURE_TIME_HORIZON
        self.num_future_poses    = config.NUM_FUTURE_POSES
        self.num_agents          = config.NUM_AGENTS

        # name of map features to be extracted.
        self._map_features = [
            'LANE', 
            'ROUTE_LANES', 
            'CROSSWALK'
        ] 
        
        # maximum number of elements to extract per feature layer.
        self._max_elements = {
            'LANE'       : config.LANE_NUM, 
            'ROUTE_LANES': config.ROUTE_LANES_NUM, 
            'CROSSWALK'  : config.CROSSWALKS_NUM
        } 
        
        # maximum number of points per feature to extract per feature layer.
        self._max_points = {
            'LANE'       : config.LANE_POINTS_NUM, 
            'ROUTE_LANES': config.ROUTE_LANES_POINTS_NUM, 
            'CROSSWALK'  : config.CROSSWALKS_POINTS_NUM
        } 
        
        # [m] query radius scope relative to the current pose.
        self._radius = config.QUERY_RADIUS 
        
        self._interpolation_method = 'linear' 

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, 
            num_samples=self.num_past_poses, 
            time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, 
            self._map_features, 
            ego_coords, 
            self._radius, 
            route_roadblock_ids, 
            traffic_light_data
        )

        vector_map = map_process(
            ego_state.rear_axle, 
            coords, 
            traffic_light_data, 
            self._map_features, 
            self._max_elements, 
            self._max_points,
            self._interpolation_method
        )

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, 
            num_samples=self.num_future_poses, 
            time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle,
            [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, 
                time_horizon=self.future_time_horizon, 
                num_samples=self.num_future_poses
            )
        ]
        
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects        
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(
            current_ego_state, 
            future_tracked_objects_tensor_list, 
            self.num_agents, 
            agent_index
        )
                
        return agent_futures
    
    def plot_scenario(self, data):
        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'], alpha=0.5, linewidth=2)

        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def process(self, debug=False):
        map_name = self.scenario._map_name
        token = self.scenario.token

        # get agent past tracks
        ego_agent_past, time_stamps_past = self.get_ego_agent()
        neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
        
        ego_agent_past, neighbor_agents_past, neighbor_indices = \
            agent_past_process(
                ego_agent_past, 
                time_stamps_past, 
                neighbor_agents_past, 
                neighbor_agents_types, 
                self.num_agents
            )

        # get vector set map
        vector_map = self.get_map()

        # get agent future tracks
        ego_agent_future = self.get_ego_agent_future()
        neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

        # gather data
        data = {
            "map_name": map_name, 
            "token": token, 
            "ego_agent_past": ego_agent_past, 
            "ego_agent_future": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past, 
            "neighbor_agents_future": neighbor_agents_future
        }
        
        # update vector map info
        data.update(vector_map)

        # encoding features
        features = encoding_vectornet_features(
            neighbor_agents_past,
            neighbor_agents_future,
            ego_agent_past,
            ego_agent_future,
            vector_map
        )
    
        save_features(
            df=features,
            name=str(data['token']),
            dir_=config.SAVE_PATH
        )    
    
        # visualization
        if debug:
            self.plot_scenario(data)
            
        return data, features

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

    builder = NuPlanScenarioBuilder(
        config.DATA_PATH, 
        config.MAP_PATH, 
        sensor_root, 
        db_files, 
        config.MAP_VERSION, 
        scenario_mapping=scenario_mapping
    )

    # scenarios for training
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            scenarios_per_type, 
            total_scenarios, 
            shuffle_scenarios
        )
    )

    # enable parallel process
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    # get scenarios
    scenarios = builder.get_scenarios(scenario_filter, worker)
    
    # pick a scenario
    scenario = scenarios[11]

    # process data
    data_processor = DataProcessor(scenario)
    data, features = data_processor.process(debug=True)