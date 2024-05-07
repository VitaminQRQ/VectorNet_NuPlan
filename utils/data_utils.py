import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

from nuplan.database.nuplan_db.nuplan_scenario_queries import *
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local

from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *

from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points
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

def _extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects.
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = _extract_agent_tensor(past_tracked_objects[i], track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)


def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
    else:
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state


def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input tensor data of the ego past.
    :param past_time_stamps: The input tensor data of the past timestamps.
    :param past_time_stamps: The input tensor data of other agents in the past.
    :return: ego_agent_array, other_agents_array.
    """
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    """
    Model input feature representing the present and past states of the ego and agents, including:
    ego: <np.ndarray: num_frames, 7>
        The num_frames includes both present and past frames.
        The last dimension is the ego pose (x, y, heading) velocities (vx, vy) acceleration (ax, ay) at time t.
    agents: <np.ndarray: num_frames, num_agents, 8>
        Agent features indexed by agent feature type.
        The num_frames includes both present and past frames.
        The num_agents is padded to fit the largest number of agents across all frames.
        The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) and size (length, width) at time t.
    """

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
    
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    '''
    Post-process the agents tensor to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
        The num_frames includes both present and past frames.
    '''
    agents = np.zeros(shape=(num_agents, agents_tensor.shape[0], agents_tensor.shape[-1]+3), dtype=np.float32)

    # sort agents according to distance to ego
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    for i, j in enumerate(indices):
        agents[i, :, :agents_tensor.shape[-1]] = agents_tensor[:, j, :agents_tensor.shape[-1]].numpy()
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_tensor.shape[-1]:] = [1, 0, 0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 1, 0]
        else:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 0, 1]

    return ego_tensor.numpy().astype(np.float32), agents, indices


def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    anchor_ego_state = torch.tensor([
        anchor_ego_state.rear_axle.x, 
        anchor_ego_state.rear_axle.y, 
        anchor_ego_state.rear_axle.heading, 
        anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
        anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
        anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
    ])
    
    agent_future = filter_agents_tensor(future_tracked_objects)
        
    local_coords_agent_states = []
    for agent_state in agent_future:
        local_coords_agent_states.append(
            convert_absolute_quantities_to_relative(
                agent_state, 
                anchor_ego_state, 
                'agent'
            )
        )
            
    padded_agent_states = pad_agent_states_with_zeros(local_coords_agent_states)
    
    # fill agent features into the array
    agent_futures = np.zeros(
        shape=(num_agents, padded_agent_states.shape[0]-1, 3), 
        dtype=np.float32
    )
        
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[
            1:, 
            j, 
            [
                AgentInternalIndex.x(), 
                AgentInternalIndex.y(), 
                AgentInternalIndex.heading()
            ]
        ].numpy()
    
    return agent_futures

def pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = torch.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx]==row_idx]

    return pad_agent_trajectories


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

def agent_ground_truth_traj_to_increments(last_observed_agent, agent_future):
    agent_increments = np.zeros_like(agent_future[:, 0:2])

    agent_increments[1:] = agent_future[1:, 0:2] - agent_future[:-1, 0:2]
    agent_increments[0] = agent_future[0, 0:2] - last_observed_agent    
    
    return agent_increments

def restore_agent_traj_from_increments(origin, increments):
    """
    Restore the full trajectory of an agent from incremental changes.
    
    :param origin: The starting absolute position (x, y) of the agent.
    :param increments: A numpy array of incremental changes (delta_x, delta_y).
    :return: A numpy array representing the absolute trajectory (x, y) of the agent.
    """
    # 创建一个数组，用于存储恢复的轨迹点，初始点为原点
    trajectory = np.zeros_like(increments)
    trajectory[0] = origin + increments[0]

    # 累加增量，还原出完整的轨迹
    for i in range(1, len(increments)):
        trajectory[i] = trajectory[i - 1] + increments[i]

    return trajectory

def encoding_vectornet_features(
    agents_past, 
    agents_future, 
    ego_past,
    ego_future,
    vector_map
    ):
    # each subgraph has its only ID
    subgraph_ID = 0
    
    # each subgraph ID has a series of node. These nodes are stacked 
    # vertically in a list and there has a start node and end node.  
    trajectory_ID_to_indices = {}
    map_ID_to_indices = {}
    
    # create an empty trajectory subgraph
    # each node is a 1 * 8 tensor 
    # [start_x, start_y, end_x, end_y, x, x, x, subgraph_ID]
    # agent type is one hot encoded
    trajectory_subgraph = np.empty((0, 8))
    
    # create an empty map subgraph
    # each node is a 1 * 8 tensor
    # [start_x, start_y, end_x, end_y, x, x, x, subgraph_ID]
    # lane type is one hot encoded
    map_subgraph = np.empty((0, 8))
    
    # encoding ego features
    ego_subgraph = np.hstack((
        ego_past[ :-1, 0].reshape(-1, 1),   
        ego_past[ :-1, 1].reshape(-1, 1),      
        ego_past[1:  , 0].reshape(-1, 1),
        ego_past[1:  , 1].reshape(-1, 1),  
        np.zeros((len(ego_past) - 1, 1)),
        np.zeros((len(ego_past) - 1, 1)), 
        np.zeros((len(ego_past) - 1, 1)),  
        np.ones ((len(ego_past) - 1, 1)) * subgraph_ID
    ))
    
    trajectory_subgraph = np.vstack((
        trajectory_subgraph,
        ego_subgraph
    ))
    
    start_ID = 0
    end_ID = len(ego_past) - 2
    trajectory_ID_to_indices[subgraph_ID] = (start_ID, end_ID)
    subgraph_ID += 1
    
    # encoding ego ground truth
    ground_truth = np.empty((0, 2))
    ego_ground_truth = agent_ground_truth_traj_to_increments(
        ego_past[-1, 0:2], 
        ego_future[:, 0:2]
    )
    
    ground_truth = np.vstack((
        ground_truth,
        ego_ground_truth
    ))
 
    # encoding agent features
    for i in range(len(agents_past)):
        agent = agents_past[i]
  
        agent_subgraph = np.hstack((
            agent[ :-1, 0].reshape(-1, 1),    
            agent[ :-1, 1].reshape(-1, 1),     
            agent[1:  , 0].reshape(-1, 1),
            agent[1:  , 1].reshape(-1, 1),  
            np.zeros((len(agent) - 1, 1)),
            np.zeros((len(agent) - 1, 1)), 
            np.zeros((len(agent) - 1, 1)),  
            np.ones ((len(agent) - 1, 1)) * subgraph_ID            
        ))
        
        trajectory_subgraph = np.vstack((
            trajectory_subgraph,
            agent_subgraph
        ))
        
        start_ID = end_ID + 1
        end_ID += len(agent_subgraph)

        trajectory_ID_to_indices[subgraph_ID] = (start_ID, end_ID)
        subgraph_ID += 1

        # ground truth
        origin = agent[-1, 0:2]
        future = agents_future[i]
        
        agent_ground_truth = agent_ground_truth_traj_to_increments(
            origin,
            future[:, 0:2]
        )

        ground_truth = np.vstack((
            ground_truth,
            agent_ground_truth
        ))
  
    # encoding lane features
    start_ID = -1
    end_ID = -1

    # encoding centerline features
    for lane in vector_map['lanes']:
        lane_subgraph = np.hstack((
            lane[ :-1, 0].reshape(-1, 1),  
            lane[ :-1, 1].reshape(-1, 1),       
            lane[1:  , 0].reshape(-1, 1),
            lane[1:  , 1].reshape(-1, 1),  
            np.zeros((len(lane) - 1, 1)),
            np.zeros((len(lane) - 1, 1)), 
            np.zeros((len(lane) - 1, 1)),  
            np.ones ((len(lane) - 1, 1)) * subgraph_ID                  
        ))

        map_subgraph = np.vstack((
            map_subgraph,
            lane_subgraph
        ))
  
        start_ID = end_ID + 1
        end_ID += len(lane_subgraph)
  
        map_ID_to_indices[subgraph_ID] = (start_ID, end_ID)
        subgraph_ID += 1
  
    # encoding crosswalk features
    for lane in vector_map['crosswalks']:
        lane_subgraph = np.hstack((
            lane[ :-1, 0].reshape(-1, 1),  
            lane[ :-1, 1].reshape(-1, 1),       
            lane[1:  , 0].reshape(-1, 1),
            lane[1:  , 1].reshape(-1, 1),  
            np.zeros((len(lane) - 1, 1)),
            np.zeros((len(lane) - 1, 1)), 
            np.zeros((len(lane) - 1, 1)),  
            np.ones ((len(lane) - 1, 1)) * subgraph_ID                  
        ))

        map_subgraph = np.vstack((
            map_subgraph,
            lane_subgraph
        ))
  
        start_ID = end_ID + 1
        end_ID += len(lane_subgraph)
  
        map_ID_to_indices[subgraph_ID] = (start_ID, end_ID)
        subgraph_ID += 1

    # encoding route lane features
    for lane in vector_map['route_lanes']:
        lane_subgraph = np.hstack((
            lane[ :-1, 0].reshape(-1, 1),  
            lane[ :-1, 1].reshape(-1, 1),       
            lane[1:  , 0].reshape(-1, 1),
            lane[1:  , 1].reshape(-1, 1),  
            np.zeros((len(lane) - 1, 1)),
            np.zeros((len(lane) - 1, 1)), 
            np.zeros((len(lane) - 1, 1)),  
            np.ones ((len(lane) - 1, 1)) * subgraph_ID                  
        ))
  
        map_subgraph = np.vstack((
            map_subgraph,
            lane_subgraph
        ))
  
        start_ID = end_ID + 1
        end_ID += len(lane_subgraph)
  
        map_ID_to_indices[subgraph_ID] = (start_ID, end_ID)
        subgraph_ID += 1

    # data frame
    feature_data = [[
        np.vstack((trajectory_subgraph, map_subgraph)),
        ground_truth,
        trajectory_ID_to_indices,
        map_ID_to_indices,
        trajectory_subgraph.shape[0],
        map_subgraph.shape[0]
    ]]
    
    return pd.DataFrame(
        feature_data,
        columns=[
            "POLYLINE_FEATURES", 
            "GROUND_TRUTH",
            "TRAJ_ID_TO_INDICES", 
            "LANE_ID_TO_INDICES", 
            "TARJ_SIZE", 
            "LANE_SIZE"
        ]
    )

def save_features(df, name, dir_=None):
    """
    Save a pandas DataFrame to a pickle file in the specified directory.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        name (str): The base name of the file to which the DataFrame will be saved.
        dir_ (str, optional): The directory where the pickle file will be saved.
                              If not specified, './input_data' will be used.

    Returns:
        None
    """
    # Set default directory if not provided
    dir_ = './input_data' if dir_ is None else dir_
    
    # Create the directory if it doesn't exist
    os.makedirs(dir_, exist_ok=True)

    # Define the complete path for the pickle file
    file_path = os.path.join(dir_, f"features_{name}.pkl")

    # Save the DataFrame as a pickle file
    df.to_pickle(file_path)

def decoding_features(observations):
    agent_index_end        = (config.NUM_AGENTS + 1) * config.NUM_PAST_POSES
    lane_index_start       = agent_index_end
    lane_index_end         = lane_index_start + config.LANE_NUM * (config.LANE_POINTS_NUM - 1)

    crosswalk_index_start  = lane_index_end
    crosswalk_index_end    = crosswalk_index_start + config.CROSSWALKS_NUM * (config.CROSSWALKS_POINTS_NUM - 1)

    route_lane_index_start = crosswalk_index_end

    ego_past = None
    agent_past_list = []
    agent_current_pose_list = []
    
    lane_list = []
    crosswalk_list = []
    route_lane_list = []

    # plot agents
    observations = observations.cpu()
    for i in range(config.NUM_AGENTS + 1):
        agent_data = observations[i * config.NUM_PAST_POSES : 
                                (i + 1) * config.NUM_PAST_POSES, 
                                :4]
        
        agent_start_x = agent_data[:, 0]
        agent_start_y = agent_data[:, 1]
        agent_end_x = agent_data[:, 2]
        agent_end_y = agent_data[:, 3]
        
        last_end_x = agent_end_x[-1].unsqueeze(0)
        last_end_y = agent_end_y[-1].unsqueeze(0)
        
        agent_x = torch.cat((agent_start_x, last_end_x), dim=0)
        agent_y = torch.cat((agent_start_y, last_end_y), dim=0)
        
        agent_past = torch.stack((agent_x, agent_y), dim=1)
        
        if i == 0:
            ego_past = agent_past.numpy()
        else:
            agent_past_list.append(agent_past.numpy())

        agent_current_pose_list.append([last_end_x.item(), last_end_y.item()])
    
    agent_current_pose_list = torch.tensor(agent_current_pose_list)
    
    for i in range(config.LANE_NUM):
        lane_data = observations[
            lane_index_start + i * (config.LANE_POINTS_NUM - 1):
            lane_index_start + (i + 1) * (config.LANE_POINTS_NUM - 1), 
            :4]
            
        lane_start_x = lane_data[:, 0]
        lane_start_y = lane_data[:, 1]
        lane_end_x = lane_data[:, 2]
        lane_end_y = lane_data[:, 3]
        
        last_end_x = lane_end_x[-1].unsqueeze(0)
        last_end_y = lane_end_y[-1].unsqueeze(0)
        
        lane_x = torch.cat((lane_start_x, last_end_x), dim=0)
        lane_y = torch.cat((lane_start_y, last_end_y), dim=0)
        
        lane = torch.stack((lane_x, lane_y), dim=1)
        lane_list.append(lane.numpy())

    for i in range(config.CROSSWALKS_NUM):
        crosswalk_data = observations[
            crosswalk_index_start + i * (config.CROSSWALKS_POINTS_NUM - 1):
            crosswalk_index_start + (i + 1) * (config.CROSSWALKS_POINTS_NUM - 1), 
            :4]
            
        crosswalk_start_x = crosswalk_data[:, 0]
        crosswalk_start_y = crosswalk_data[:, 1]
        crosswalk_end_x = crosswalk_data[:, 2]
        crosswalk_end_y = crosswalk_data[:, 3]
        
        last_end_x = crosswalk_end_x[-1].unsqueeze(0)
        last_end_y = crosswalk_end_y[-1].unsqueeze(0)
        
        crosswalk_x = torch.cat((crosswalk_start_x, last_end_x), dim=0)
        crosswalk_y = torch.cat((crosswalk_start_y, last_end_y), dim=0)
        
        crosswalk = torch.stack((crosswalk_x, crosswalk_y), dim=1)
        crosswalk_list.append(crosswalk.numpy())

    for i in range(config.ROUTE_LANES_NUM):
        route_lane_data = observations[
            route_lane_index_start + i * (config.ROUTE_LANES_POINTS_NUM - 1):
            route_lane_index_start + (i + 1) * (config.ROUTE_LANES_POINTS_NUM - 1), 
            :4]
            
        route_lane_start_x = route_lane_data[:, 0]
        route_lane_start_y = route_lane_data[:, 1]
        route_lane_end_x = route_lane_data[:, 2]
        route_lane_end_y = route_lane_data[:, 3]
        
        last_end_x = route_lane_end_x[-1].unsqueeze(0)
        last_end_y = route_lane_end_y[-1].unsqueeze(0)
        
        route_lane_x = torch.cat((route_lane_start_x, last_end_x), dim=0)
        route_lane_y = torch.cat((route_lane_start_y, last_end_y), dim=0)
        
        route_lane = torch.stack((route_lane_x, route_lane_y), dim=1)
        route_lane_list.append(route_lane.numpy())
    
    return ego_past, agent_past_list, lane_list, crosswalk_list, route_lane_list, agent_current_pose_list

def increment_to_trajectories(predictions, agent_current_pose_list):
    increments = predictions.view(config.NUM_AGENTS + 1, config.NUM_FUTURE_POSES, 2)
    predicted_path_list = torch.zeros(config.NUM_AGENTS + 1, config.NUM_FUTURE_POSES, 2)
    predicted_path_list[:, 0, :] = agent_current_pose_list + increments[:, 0, :]

    for i in range(1, config.NUM_FUTURE_POSES):
        predicted_path_list[:, i, :] = predicted_path_list[:, i-1, :] + increments[:, i, :]

    return predicted_path_list

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

COLOR_DICT = {
    'centerline' : '#6495ED',
    'crosswalk'  : '#778899',
    'route_lane' : '#191970',
    'agent'      : '#007672',
    'ego'        : '#d33e4c',
    'others'     : '#d3e8ef',
}

def create_ego_raster(vehicle_state):
    # Extract ego vehicle dimensions
    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    ego_bottom_right = (x_center - ego_rear_length, y_center - ego_width/2)

    # Paint the rectangle
    rect = plt.Rectangle(
        ego_bottom_right, 
        ego_front_length + ego_rear_length, 
        ego_width, 
        linewidth=2, 
        color=COLOR_DICT['ego'], 
        alpha=0.6, 
        zorder=3,
        transform=mpl.transforms.Affine2D().rotate_around(
            *(x_center, y_center), heading
        ) + plt.gca().transData)
    
    plt.gca().add_patch(rect)


def create_agents_raster(agents):
    for i in range(agents.shape[0]):
        if agents[i, 0] != 0:
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i, 2]
            agent_length, agent_width = agents[i, 6],  agents[i, 7]
            agent_bottom_right = (x_center - agent_length/2, y_center - agent_width/2)

            rect = plt.Rectangle(
                agent_bottom_right, 
                agent_length, 
                agent_width, 
                linewidth=1, 
                color=COLOR_DICT['agent'], 
                alpha=0.6, 
                zorder=3,
                transform=mpl.transforms.Affine2D().rotate_around(
                    *(x_center, y_center), heading
                ) + plt.gca().transData
            )
            
            plt.gca().add_patch(rect)


def create_map_raster(lanes, crosswalks, route_lanes):
    for lane in lanes:
        if lane[0][0] != 0:
            plt.plot(
                lane[:, 0], 
                lane[:, 1], 
                color=COLOR_DICT['centerline'], 
                linewidth=1
            ) # plot centerline

    for crosswalk in crosswalks:
        if crosswalk[0][0] != 0:
            plt.plot(
                crosswalk[:, 0], 
                crosswalk[:, 1], 
                color=COLOR_DICT['crosswalk'], 
                linewidth=1
            ) # plot crosswalk

    for route_lane in route_lanes:
        if route_lane[0][0] != 0:
            plt.plot(
                route_lane[:, 0], 
                route_lane[:, 1], 
                color=COLOR_DICT['route_lane'], 
                linewidth=1
            ) # plot route_lanes

def retrieve_prediction_result(initial_positions, increments):
    """
    根据初始位置和位置增量重建完整的轨迹。

    参数:
    - initial_positions: 一个形状为 (num_agents, 2) 的张量，包含每个代理的初始 x 和 y 坐标。
    - increments: 一个形状为 (num_agents, 80) 的张量，每行包含 40 个 x 增量和 40 个 y 增量。
    
    返回:
    - paths: 一个形状为 (num_agents, 41, 2) 的张量，包含每个代理的完整轨迹，从初始位置开始。
    """

    num_agents = initial_positions.shape[0]
    # 确保增量张量有正确的形状 (num_agents, 40, 2)
    increments = increments.view(num_agents, 40, 2)
    
    # 初始化路径张量，每个代理多一个初始点所以是 41 个点
    paths = torch.zeros(num_agents, 41, 2)
    
    # 设置每个代理的初始位置
    paths[:, 0, :] = initial_positions
      
    # 累加位置增量以构造完整的路径
    for i in range(1, 41):
        paths[:, i, :] = paths[:, i-1, :] + increments[:, i-1, :]

    return paths

def draw_trajectory(ego_trajectory, agent_trajectories, alpha=1, linewidth=3):
    # plot ego 
    plt.plot(
        ego_trajectory[:, 0], 
        ego_trajectory[:, 1], 
        color=COLOR_DICT['ego'], 
        linewidth=linewidth, 
        zorder=3,
        alpha=alpha
    )

    for trajectory in agent_trajectories:
        if trajectory[-1, 0] != 0:
            plt.plot(
                trajectory[:, 0], 
                trajectory[:, 1], 
                color=COLOR_DICT['agent'], 
                linewidth=linewidth, 
                zorder=3,
                alpha=alpha
            )            
