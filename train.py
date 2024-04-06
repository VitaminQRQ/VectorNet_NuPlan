import torch
import numpy as np

from torch_geometric.loader import DataLoader

from utils import config
from utils.dataset import GraphDataset
from vectornet.vectornet import VectornetGNN



if __name__ == "__main__":
    # Set seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    # Get training set
    train_data = GraphDataset(config.TRAIN_PATH).shuffle()
    
    # Get validation set
    validate_data = GraphDataset(config.TRAIN_PATH)
    
    # Load training data
    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    # Load validation data
    validate_loader = DataLoader(
        validate_data,
        batch_size=config.BATCH_SIZE
    )

    # Create predictor
    device = 
    model = VectornetGNN(
        in_channels=8,
        out_channels=(config.NUM_AGENTS + 1) * config.NUM_FUTURE_POSES,
    ).to(device)