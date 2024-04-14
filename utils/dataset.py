import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from . import config

import logging

def get_fully_connected_edge_index(num_nodes, start=0):
    """
    Generate a fully connected graph's edge indices with no self-loops.

    Args:
        num_nodes (int): The number of nodes in the graph.
        start (int, optional): A starting index for node numbering.

    Returns:
        np.ndarray: The edge indices of the fully connected graph.
    """
    # create edge indices
    edge_indices = np.array(
        [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], 
        dtype=np.int64
    ).T
    
    # update edge indices to meet start index
    edge_indices += start
    
    return edge_indices

def get_sequential_edge_index(num_nodes, start=0):
    """
    Generate edge indices for a sequential graph representing an agent's trajectory.

    Args:
        num_nodes (int): The number of nodes in the graph representing the trajectory.
        start (int, optional): A starting index for node numbering, allowing sequential
                               graphs to be combined without index collisions.

    Returns:
        np.ndarray: The edge indices of the sequential graph.
    """
    # Create edge indices such that each node (except the last) points to the next
    edge_indices = np.array(
        [[i, i + 1] for i in range(start, start + num_nodes - 1)], 
        dtype=np.int64
    ).T

    return edge_indices

class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass
    
    def process(self):
        data_path_list = sorted(
            os.path.join(self.root, data_path)
            for data_path in os.listdir(self.root)
            if data_path.endswith('pkl')
        )

        node_size_list = []
        graph_data_list = []

        for data_path in tqdm(data_path_list):
            data = pd.read_pickle(data_path)

            features = data['POLYLINE_FEATURES'].values[0]
            cluster = features[:, -1].astype(np.int32)
            
            node_size_list.append(cluster.max())
            
            y = data['GROUND_TRUTH'].values[0].astype(np.float32)
            
            trajectory_ID_to_indices = data['TRAJ_ID_TO_INDICES'].values[0]
            lane_ID_to_indices = data['LANE_ID_TO_INDICES'].values[0]
            
            x_list = []
            edge_index_list = []
            cluster_list = []
            
            # process agent trajectories
            for id, mask in trajectory_ID_to_indices.items():
                trajectory_features = features[mask[0] : mask[1] + 1]

                edge_index = get_sequential_edge_index(
                    trajectory_features.shape[0],
                    start=mask[0]
                )
                
                x_list.append(trajectory_features)
                edge_index_list.append(edge_index)
            
            accumulate_edge_index = mask[1] + 1

            # process lanes
            for id, mask in lane_ID_to_indices.items():
                lane_features = features[
                    accumulate_edge_index + mask[0] : 
                    accumulate_edge_index + mask[1] + 1
                ]
                
                print((accumulate_edge_index + mask[0], accumulate_edge_index + mask[1] + 1))
                
                edge_index = get_fully_connected_edge_index(
                    lane_features.shape[0],
                    start=mask[0] + accumulate_edge_index
                )

                x_list.append(lane_features)
                edge_index_list.append(edge_index) 
            
            # concat subgraph data to global graph
            x_combined = np.vstack(x_list).astype(np.float32)
            edge_index_combined = np.hstack(edge_index_list)
            
            graph_data = GraphData(
                x=torch.from_numpy(x_combined).to(torch.float32),
                y=torch.from_numpy(y).to(torch.float32),
                cluster=torch.from_numpy(cluster).to(torch.long),
                edge_index=torch.from_numpy(edge_index_combined).to(torch.long),
                valid_len=torch.tensor([cluster.max()], dtype=torch.long),
                time_step_len=torch.tensor([trajectory_features.shape[0]], dtype=torch.long)
            )
            
            graph_data_list.append(graph_data)

        data, slices = self.collate(graph_data_list)
        torch.save(
            (data, slices),
            self.processed_paths[0]
        )

if __name__== "__main__":
    processed_data_path = config.SAVE_PATH
    dataset = GraphDataset(processed_data_path)
    batch_iter = DataLoader(dataset, batch_size=1)

    # Check if CUDA is available and set the device accordingly
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Get a single batch from the iterator and move it to the selected device    
    batch = next(iter(batch_iter))    
    
    # debug
    print(batch)