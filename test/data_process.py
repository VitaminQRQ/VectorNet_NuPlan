import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame

from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
	AgentInternalIndex,
	EgoInternalIndex,
	sampled_past_ego_states_to_tensor,
	sampled_past_timestamps_to_tensor,
	compute_yaw_rate_from_state_tensors,
	filter_agents_tensor,
	pack_agents_tensor,
	pad_agent_states
)

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines,
	LaneSegmentTrafficLightData,
	VectorFeatureLayer, 
 	VectorFeatureLayerMapping,
	get_lane_polylines,
	get_traffic_light_encoding,
 	get_route_lane_polylines_from_roadblock_ids,
	get_map_object_polygons
)

from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap

from typing import List, Dict, Tuple

def get_filter_parameters(num_scenarios_per_type=20, limit_total_scenarios=None, shuffle=True):
	# nuplan challenge
	scenario_types = [
		'starting_left_turn',
		'starting_right_turn',
		'starting_straight_traffic_light_intersection_traversal',
		'stopping_with_lead',
		'high_lateral_acceleration',
		'high_magnitude_speed',
		'low_magnitude_speed',
		'traversing_pickup_dropoff',
		'waiting_for_pedestrian_to_cross',
		'behind_long_vehicle',
		'stationary_in_traffic',
		'near_multiple_vehicles',
		'changing_lane',
		'following_lane_with_lead',
	]

	scenario_tokens = None              # List of scenario tokens to include
	log_names = None                     # Filter scenarios by log names
	map_names = None                     # Filter scenarios by map names

	num_scenarios_per_type               # Number of scenarios per type
	limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
	timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
	ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

	expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
	remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
	shuffle                             # Whether to shuffle the scenarios

	ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
	ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
	speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

	return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
		   expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

def get_scenario_map():
	scenario_map = {
		'accelerating_at_crosswalk': [15.0, -3.0],
		'accelerating_at_stop_sign': [15.0, -3.0],
		'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
		'accelerating_at_traffic_light': [15.0, -3.0],
		'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
		'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
		'behind_bike': [15.0, -3.0],
		'behind_long_vehicle': [15.0, -3.0],
		'behind_pedestrian_on_driveable': [15.0, -3.0],
		'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
		'changing_lane': [15.0, -3.0],
		'changing_lane_to_left': [15.0, -3.0],
		'changing_lane_to_right': [15.0, -3.0],
		'changing_lane_with_lead': [15.0, -3.0],
		'changing_lane_with_trail': [15.0, -3.0],
		'crossed_by_bike': [15.0, -3.0],
		'crossed_by_vehicle': [15.0, -3.0],
		'following_lane_with_lead': [15.0, -3.0],
		'following_lane_with_slow_lead': [15.0, -3.0],
		'following_lane_without_lead': [15.0, -3.0],
		'high_lateral_acceleration': [15.0, -3.0],
		'high_magnitude_jerk': [15.0, -3.0],
		'high_magnitude_speed': [15.0, -3.0],
		'low_magnitude_speed': [15.0, -3.0],
		'medium_magnitude_speed': [15.0, -3.0],
		'near_barrier_on_driveable': [15.0, -3.0],
		'near_construction_zone_sign': [15.0, -3.0],
		'near_high_speed_vehicle': [15.0, -3.0],
		'near_long_vehicle': [15.0, -3.0],
		'near_multiple_bikes': [15.0, -3.0],
		'near_multiple_pedestrians': [15.0, -3.0],
		'near_multiple_vehicles': [15.0, -3.0],
		'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
		'near_pedestrian_on_crosswalk': [15.0, -3.0],
		'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
		'near_trafficcone_on_driveable': [15.0, -3.0],
		'on_all_way_stop_intersection': [15.0, -3.0],
		'on_carpark': [15.0, -3.0],
		'on_intersection': [15.0, -3.0],
		'on_pickup_dropoff': [15.0, -3.0],
		'on_stopline_crosswalk': [15.0, -3.0],
		'on_stopline_stop_sign': [15.0, -3.0],
		'on_stopline_traffic_light': [15.0, -3.0],
		'on_traffic_light_intersection': [15.0, -3.0],
		'starting_high_speed_turn': [15.0, -3.0],
		'starting_left_turn': [15.0, -3.0],
		'starting_low_speed_turn': [15.0, -3.0],
		'starting_protected_cross_turn': [15.0, -3.0],
		'starting_protected_noncross_turn': [15.0, -3.0],
		'starting_right_turn': [15.0, -3.0],
		'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
		'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
		'starting_u_turn': [15.0, -3.0],
		'starting_unprotected_cross_turn': [15.0, -3.0],
		'starting_unprotected_noncross_turn': [15.0, -3.0],
		'stationary': [15.0, -3.0],
		'stationary_at_crosswalk': [15.0, -3.0],
		'stationary_at_traffic_light_with_lead': [15.0, -3.0],
		'stationary_at_traffic_light_without_lead': [15.0, -3.0],
		'stationary_in_traffic': [15.0, -3.0],
		'stopping_at_crosswalk': [15.0, -3.0],
		'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
		'stopping_at_stop_sign_with_lead': [15.0, -3.0],
		'stopping_at_stop_sign_without_lead': [15.0, -3.0],
		'stopping_at_traffic_light_with_lead': [15.0, -3.0],
		'stopping_at_traffic_light_without_lead': [15.0, -3.0],
		'stopping_with_lead': [15.0, -3.0],
		'traversing_crosswalk': [15.0, -3.0],
		'traversing_intersection': [15.0, -3.0],
		'traversing_narrow_lane': [15.0, -3.0],
		'traversing_pickup_dropoff': [15.0, -3.0],
		'traversing_traffic_light_intersection': [15.0, -3.0],
		'waiting_for_pedestrian_to_cross': [15.0, -3.0]
	}

	return scenario_map

def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data

def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param ego_pose: the current pose of the ego vehicle.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data: Optional traffic light status corresponding to map elements at given index in coords.
        [num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options: 'linear' and 'area'.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    :raise ValueError: If coordinates and traffic light data size do not match.
    """
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
    
        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor

def map_process(anchor_state, coords, traffic_light_data, map_features, max_elements, max_points, interpolation_method):
    """
    This function process the data from the raw vector set map data.
    :param anchor_state: The current state of the ego vehicle.
    :param coords: The input data of the vectorized map coordinates.
    :param traffic_light_data: The input data of the traffic light data.
    :return: dict of the map elements.
    """

    # convert data to tensor list
    anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32)
    list_tensor_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float32))
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
            list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data is available for point at given index or if it is zero-padded.
    """
    
    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )

            coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                    anchor_state_tensor,
                    feature_coords,
                    feature_tl_data,
                    max_elements[feature_name],
                    max_points[feature_name],
                    traffic_light_encoding_dim,
                    interpolation=interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                        VectorFeatureLayer.CROSSWALK.name
                    ]
                    else None,
            )

            coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state_tensor)

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data

    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features
    """

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_route_lanes = polyline_process(polylines, avails)

        else:
            pass

    vector_map_output = {'lanes': vector_map_lanes, 'crosswalks': vector_map_crosswalks, 'route_lanes': vector_map_route_lanes}

    return vector_map_output

def polyline_process(polylines, avails, traffic_light=None):
    dim = 3 if traffic_light is None else 7
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    for i in range(polylines.shape[0]):
        if avails[i][0]: 
            polyline = polylines[i]
            polyline_heading = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]))
            polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])[:, np.newaxis]
            if traffic_light is None:
                new_polylines[i] = np.concatenate([polyline, polyline_heading], axis=-1)
            else:
                new_polylines[i] = np.concatenate([polyline, polyline_heading, traffic_light[i]], axis=-1)  

    return new_polylines

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

class DataProcessor:
	def __init__(self, scenario):
		# map api
		self.scenario = scenario
		
		# discrete 1 second to several points
		self.discrete_size = 10

		# history observation horizon
		self.past_time_horizon = 2

		# prediction horizon
		self.future_time_horizon = 8

		# number of discrete trajectory points
		self.num_past_poses = self.discrete_size * self.past_time_horizon

		# number of discrete trajectory points in future horizon
		self.num_future_poses = self.discrete_size * self.future_time_horizon

		# deal with top 20 closest agents around the ego vehicle
		self.num_agents = 20

		# [m] query radius scope relative to the current pose.
		self.query_radius = 60

		# Interpolation method to apply when interpolating to maintain fixed size map elements.
		self.interpolation_method = 'linear'

		# name of map features to be extracted.
		self.map_features = [
			'LANE', 
			'ROUTE_LANES', 
			'CROSSWALK'
		] 

		# maximum number of elements to extract per feature layer.
		self.max_elements = {
			'LANE': 40, 
			'ROUTE_LANES': 10, 
			'CROSSWALK': 5
		}      
		
		# maximum number of points per feature to extract per feature layer.
		self.max_points = {
			'LANE': 50, 
			'ROUTE_LANES': 50, 
			'CROSSWALK': 30
		}

		# initial state of ego vehicle
		self.anchor_ego_state = self.scenario.initial_ego_state
  
		# map related
		self.map_name = self.scenario._map_name
		self.map_api = scenario.map_api

	def process_ego_info(self):
		pass

	def process_agent_info(self):
		pass

	def process_map_info(self):
		ego_coords = Point2D(
      		self.anchor_ego_state.rear_axle.x, 
        	self.anchor_ego_state.rear_axle.y
        )

		route_roadblock_ids = self.scenario.get_route_roadblock_ids()
		traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)
		
		coords, traffic_light_data = get_neighbor_vector_set_map(
			self.map_api,
			self.map_features, 
			ego_coords,
			self.query_radius,
			route_roadblock_ids,
			traffic_light_data
		)
  
		vector_map = map_process(
      		self.anchor_ego_state.rear_axle, 
        	coords, 
         	traffic_light_data, 
          	self.map_features, 
            self.max_elements, 
            self.max_points, 
            self.interpolation_method
        )

		plot_vector_map(vector_map['lanes'], vector_map['crosswalks'], vector_map['route_lanes'])
  
		return vector_map
  
def plot_vector_map(lanes, crosswalks, route_lanes):
	for i in range(lanes.shape[0]):
		lane = lanes[i]
		if lane[0][0] != 0:
			plt.plot(lane[:, 0], lane[:, 1], 'c', linewidth=1) # plot centerline

	for j in range(crosswalks.shape[0]):
		crosswalk = crosswalks[j]
		if crosswalk[0][0] != 0:
			plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=1) # plot crosswalk

	for k in range(route_lanes.shape[0]):
		route_lane = route_lanes[k]
		if route_lane[0][0] != 0:
			plt.plot(route_lane[:, 0], route_lane[:, 1], 'g', linewidth=1) # plot route_lanes

def plot_agents(self):
	pass

def plot_ego(self):
	pass

if __name__ == '__main__':
	# nuplan arguments
	data_path = '/root/nuplan/dataset/nuplan-v1.1/splits/mini'
	map_path = '/root/nuplan/dataset/maps'
	save_path = '/root/workspace/GameFormer-Planner/ProcessedData'
	scenarios_per_type = 1000
	total_scenarios = None
	shuffle_scenarios = False
	debug = False

	map_version = "nuplan-maps-v1.0"    
	sensor_root = None
	db_files = None

	# create folder for processed data
	os.makedirs(save_path, exist_ok=True)
 
	# get scenarios
	scenario_mapping = ScenarioMapping(
		scenario_map=get_scenario_map(), 
		subsample_ratio_override=0.5
	)

	builder = NuPlanScenarioBuilder(
		data_path, 
		map_path, 
		sensor_root, 
		db_files, 
		map_version, 
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

	# delete useless variables, only deal with scenarios
	del worker, builder, scenario_filter, scenario_mapping

	# pick a scenario and process data
	scenario = scenarios[100]
 
	data_processor = DataProcessor(scenario)
	vector_map = data_processor.process_map_info()

	