import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import os
from PIL import Image
import glob
import pdb
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import random
from pathlib import Path
from natsort import natsorted
from torch.utils.data.distributed import DistributedSampler

class tseDataset(Dataset):
    def __init__(self, image_paths, transform=None, type='img'):
        self.image_paths = image_paths
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        t1_path = self.image_paths[index]
        tse_path = self.image_paths[index].replace('/t1/','/tse/')
        if self.type == 'img':
            t1_image = Image.open(t1_path)
            t1_image = torch.tensor(np.array(t1_image)).unsqueeze(0).unsqueeze(-1)
            
            tse_image = Image.open(tse_path)
            tse_image = torch.tensor(np.array(tse_image)).unsqueeze(0).unsqueeze(-1)
            
            if self.transform:
                t1_image = self.transform(t1_image).squeeze()
                tse_image = self.transform(tse_image).squeeze()
                
            t1_image = t1_image/t1_image.max()
            tse_image = tse_image/tse_image.max()

            return t1_image, tse_image
        
        elif self.type == 'latent':
            t1_latent = torch.load(t1_path)
            tse_latent = torch.load(tse_path)
            
            return t1_latent, tse_latent

def get_dataset(data_root, crop_size, size, sample=1, type='img', split_ratio=0.95):
    """Create and return the dataset without creating DataLoader"""
    target_shape = (crop_size, crop_size, 1)
    resize_transform = tio.CropOrPad(target_shape)

    transform = tio.Compose([resize_transform, tio.transforms.Resize((size,size,1))])
    if type == 'img':
        all_files = natsorted(glob.glob(f'{data_root}/*.png'))
    elif type == 'latent':
        all_files = natsorted(glob.glob(f'{data_root}/*.pt'))
        
    sample_size = max(1, np.int32(len(all_files) * sample/100))  # Use max to ensure at least one file is sampled
    
    # Using fixed seed for reproducibility across ranks
    random.seed(42)  # Fixed seed for consistent sampling
    sampled_files = random.sample(all_files, sample_size)
    random.seed(None)  # Reset seed for other random operations
    
    dataset = tseDataset(image_paths=sampled_files, transform=transform, type=type)
    
    # Create train/val split with fixed seed for reproducibility across ranks
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        dataset, 
        [int(len(dataset)*split_ratio), len(dataset)-int(len(dataset)*split_ratio)],
        generator=generator
    )
    
    return train_set, val_set

def getloader(batch_size, data_root, crop_size, size, sample=1, type='img', distributed=False, rank=0, world_size=1, split_ratio=0.95, train_shuffle=True):
    """Create and return dataloaders with optional distributed training support"""
    train_set, val_set = get_dataset(data_root, crop_size, size, sample, type, split_ratio)
    
    if distributed:
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=train_shuffle,
            seed=42  # Fixed seed for reproducibility
        )
        
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # No shuffling for validation
            seed=42
        )
        
        # Create DataLoaders with distributed samplers
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Avoid issues with uneven batch sizes in DDP
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
    else:
        # Non-distributed DataLoaders
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=train_shuffle, 
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False,  # No need to shuffle validation
            num_workers=8,
            pin_memory=True
        )
    
    return train_loader, val_loader