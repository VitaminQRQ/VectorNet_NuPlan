import os
import time

import torch
import numpy as np

import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from utils import config
from utils.dataset import GraphDataset, GraphData
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
    device = device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    model = VectornetGNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.DECAY_LR_EVERY, 
        gamma=config.DECAY_LR_FACTOR
    )
    
    global_step = 0
    model.train()
    for epoch in range(config.EPOCHS):
        print(f"start training at epoch:{epoch}")
        
        acc_loss = .0
        num_samples = 1
        start_tic = time.time()
        
        for data in train_loader:
            data = data.to(device)
            y = data.y.to(torch.float32).view(-1, config.OUT_CHANNELS)
            
            optimizer.zero_grad()
            out = model(data)
            
            loss = F.mse_loss(out, y)
            loss.backward()
            
            acc_loss += config.BATCH_SIZE * loss.item()
            num_samples += y.shape[0]
            
            optimizer.step()
                        
            if (global_step + 1) % config.SHOW_EVERY == 0:
                loss_value = loss.item()
                learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                elapsed_time = time.time() - start_tic

                # Print training info
                print(f"epoch-{epoch}, step-{global_step}： "
                      f"loss: {loss_value:.3f}, "
                      f"lr:   {learning_rate:.6f}, "
                      f"time: {elapsed_time:.4f} sec")
            
            global_step += 1
            
        scheduler.step()
        
        # Print every epoch
        loss_value = acc_loss / num_samples
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        elapsed_time = time.time() - start_tic
        print(f"finished epoch {epoch}: "
              f"loss: {loss_value:.3f}, "
              f"lr:   {learning_rate:.6f}, "
              f"time: {elapsed_time:.4f} sec")
    
        # Save params to local
        model_filename = f"model_epoch_{epoch+1:03d}.pth"
        model_path = os.path.join(config.WEIGHT_PATH, model_filename)
        
        torch.save(model.state_dict(), model_path)