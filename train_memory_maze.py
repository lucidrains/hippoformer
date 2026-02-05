# /// script
# dependencies = [
#   "torch",
#   "accelerate",
#   "einops",
#   "gym==0.25.2",
#   "memory-maze",
#   "dm-control",
#   "matplotlib",
#   "numpy<2",
#   "beartype",
#   "pillow",
#   "scipy",
#   "assoc-scan",
#   "einx",
#   "x-mlps-pytorch",
# ]
# ///

import os
os.environ['MUJOCO_GL'] = 'glfw'

from pathlib import Path

import torch
from torch import nn, Tensor, pi, stack
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from einops import rearrange
from accelerate import Accelerator

import gym
import memory_maze

from hippoformer.hippoformer import mmTEM, maze_sensory_enc_dec

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import correlate2d

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# MemoryMaze environment wrapper

def find_physics(env):
    curr = env
    for _ in range(20): # depth limit
        if hasattr(curr, '_physics'): return curr._physics
        if hasattr(curr, 'physics'): return curr.physics
        if hasattr(curr, 'env'): curr = curr.env
        elif hasattr(curr, 'unwrapped'): curr = curr.unwrapped
        else: break
    return None

class MemoryMazeEnv:
    def __init__(self, env_name = 'MemoryMaze-9x9-v0'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.physics = None

    def reset(self):
        obs = self.env.reset()
        self.physics = find_physics(self.env)
        return obs

    def step(self, action):
        return self.env.step(action)

    def get_pos(self):
        if self.physics is None:
            self.physics = find_physics(self.env)
        try:
            return self.physics.data.qpos[:2].copy()
        except Exception:
            return np.array([0., 0.])

    def generate_trajectory(self, steps = 100, skip_obs = False):
        obs = self.reset()
        observations, actions, positions = [], [], []
        
        for _ in range(steps):
            action = self.action_space.sample()
            
            if not skip_obs:
                obs_t = torch.from_numpy(obs.copy()).float()
                obs_t = rearrange(obs_t, 'h w c -> c h w') / 255.0
                observations.append(obs_t)
            
            v_w = torch.zeros(2, dtype = torch.float32)
            if action == 1: v_w[0] = 0.5   # Move forward
            elif action == 2: v_w[1] = -0.5 # Rotate right
            elif action == 3: v_w[1] = 0.5  # Rotate left
            
            actions.append(v_w)
            positions.append(torch.from_numpy(self.get_pos()).float())
            
            step_res = self.step(action)
            obs, done = step_res[0], step_res[2]
            if done: obs = self.reset()
            
        return stack(observations) if not skip_obs else None, stack(actions), stack(positions)

# dataset

class TrajectoryDataset(Dataset):
    def __init__(self, world, num_trajectories = 32, steps = 100):
        self.data = [world.generate_trajectory(steps) for _ in range(num_trajectories)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

# grid cell visualization

def get_sac(rate_map: Tensor):
    """Spatial Autocorrelogram (SAC) using Torch"""
    # rate_map: (res, res)
    mask = ~rate_map.isnan()
    if not mask.any():
        return torch.zeros_like(rate_map)

    m = rate_map.clone()
    mean = rate_map[mask].mean()
    m[mask] -= mean
    m[~mask] = 0.

    # 2D correlation via conv2d
    # correlate2d(m, m, mode='full')
    h, w = m.shape
    m_batch = rearrange(m, 'h w -> 1 1 h w')
    
    sac = F.conv2d(
        F.pad(m_batch, (w - 1, w - 1, h - 1, h - 1)),
        m_batch
    )
    return rearrange(sac, '1 1 h w -> h w')

def gaussian_blur_2d(img: Tensor, sigma: float = 1.0):
    """2D Gaussian Blur in Torch"""
    # img: (c, h, w)
    ksize = int(2 * 3 * sigma + 1)
    if ksize % 2 == 0: ksize += 1
    
    x = torch.linspace(-3 * sigma, 3 * sigma, ksize)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    
    c = img.shape[0]
    kernel2d = rearrange(kernel2d, 'h w -> 1 1 h w').to(img.device)
    kernel2d = kernel2d.expand(c, 1, -1, -1)
    
    padding = ksize // 2
    # reflect pad to avoid edge artifacts
    img_padded = F.pad(rearrange(img, 'c h w -> 1 c h w'), (padding, padding, padding, padding), mode = 'reflect')
    blurred = F.conv2d(img_padded, kernel2d, groups = c)
    return rearrange(blurred, '1 c h w -> c h w')

class GridCellVisualizer:
    def __init__(
        self,
        world: MemoryMazeEnv,
        resolution: int = 40,
        spatial_range: tuple[float, float] = (-5.0, 5.0)
    ):
        self.world = world
        self.resolution = resolution
        self.spatial_range = spatial_range

    @torch.no_grad()
    def get_rate_maps(self, model: nn.Module, steps: int = 5000):
        model.eval()
        device = next(model.parameters()).device

        # Probing trajectory (skip observations for speed)
        _, actions, positions = self.world.generate_trajectory(steps = steps, skip_obs = True)
        
        actions = actions.to(device)
        positions = positions.to(device)
        
        actions_in = rearrange(actions, 't d -> 1 t d')
        structural_codes = model.path_integrator(actions_in)
        structural_codes = rearrange(structural_codes, '1 t d -> t d')

        # Vectorized binning in Torch
        res = self.resolution
        p_min, p_max = self.spatial_range
        
        # Map positions to [0, resolution - 1]
        indices = ((positions - p_min) / (p_max - p_min + 1e-5) * (res - 1)).long()
        indices = torch.clamp(indices, 0, res - 1)

        num_cells = structural_codes.shape[-1]
        activations = torch.zeros((num_cells, res, res), device = device)
        counts = torch.zeros((res, res), device = device)

        # Flat indices for index_add_
        flat_indices = indices[:, 0] * res + indices[:, 1]
        
        activations_flat = rearrange(activations, 'd h w -> d (h w)')
        activations_flat.index_add_(1, flat_indices, structural_codes.T)
        
        counts_flat = counts.view(-1)
        counts_flat.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype = torch.float32))

        # Occupancy normalization
        rate_maps = activations / rearrange(counts.clamp(min = 1), 'h w -> 1 h w')
        mask = counts < 1
        
        # Fill NaNs before smoothing
        has_visits = (~mask).any()
        if has_visits:
            # For each cell, fill unvisited with its own mean
            # rate_maps: (c, h, w)
            # mask: (h, w)
            for i in range(num_cells):
                rmap = rate_maps[i]
                rmap[mask] = rmap[~mask].mean()
        
        # Smoothing
        rate_maps = gaussian_blur_2d(rate_maps, sigma = 1.0)
        
        # Normalize to [0, 1] per cell
        rm_min = rearrange(rate_maps.amin(dim = (1, 2)), 'c -> c 1 1')
        rm_max = rearrange(rate_maps.amax(dim = (1, 2)), 'c -> c 1 1')
        
        rate_maps = (rate_maps - rm_min) / (rm_max - rm_min).clamp(min = 1e-5)
        
        # Restore NaNs for visualization transparency
        rate_maps[:, mask] = float('nan')
            
        return rate_maps

    def visualize(
        self,
        model: nn.Module,
        epoch: int,
        path_to_save: str | Path,
        probing_steps: int = 5000
    ):
        path_to_save = Path(path_to_save)
        rate_maps = self.get_rate_maps(model, steps = probing_steps)
        rate_maps_cpu = rate_maps.cpu()

        # Sort by spatial variance to find high-information cells
        # variance handling NaNs
        variances = torch.from_numpy(np.nanvar(rate_maps_cpu.numpy(), axis = (1, 2)))
        top_indices = torch.argsort(variances, descending = True)[:8]

        fig, axes = plt.subplots(4, 4, figsize = (14, 14), facecolor = 'white')
        
        cmap_rate = plt.get_cmap('rainbow').copy()
        cmap_rate.set_bad('white')

        for i, idx in enumerate(top_indices):
            # Rate Map
            ax_rate = axes[i // 2, (i % 2) * 2]
            rate_map = rate_maps_cpu[idx]
            
            ax_rate.imshow(rate_map.numpy(), cmap = cmap_rate, interpolation = 'nearest', origin = 'lower')
            ax_rate.axis('off')
            ax_rate.set_title(f'Rate Map {idx}')

            # Spatial Autocorrelogram
            ax_sac = axes[i // 2, (i % 2) * 2 + 1]
            sac = get_sac(rate_map)
            
            ax_sac.imshow(sac.numpy(), cmap = 'jet', interpolation = 'gaussian', origin = 'lower')
            ax_sac.axis('off')
            ax_sac.set_title(f'SAC {idx}')

        plt.tight_layout()
        plt.suptitle(f'Grid Cell Discovery (Epoch {epoch})', fontsize = 18)
        
        plt.savefig(path_to_save)
        plt.close()

# main simulation

def run_simulation():
    accelerator = Accelerator()
    accelerator.print(f"Using device: {accelerator.device}")
    
    world = MemoryMazeEnv('MemoryMaze-9x9-v0')
    visualizer = GridCellVisualizer(world)
    
    model = mmTEM(
        dim = 32,
        sensory_encoder_decoder = maze_sensory_enc_dec,
        dim_sensory = (3, 64, 64),
        dim_action = 2,
        dim_encoded_sensory = 32,
        dim_structure = 64
    )
    
    optimizer = Adam(model.parameters(), lr = 1e-3)
    
    accelerator.print("Generating training dataset (scale: 64x100)...")
    dataset = TrajectoryDataset(world, num_trajectories = 64, steps = 100)
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)
    
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    results_folder = Path('results')
    results_folder.mkdir(parents = True, exist_ok = True)
    
    accelerator.print("Starting extended training on MemoryMaze3D...")
    for epoch in range(1, 16):
        model.train()
        total_loss = 0
        for obs, actions, _ in loader:
            obs = rearrange(obs, 'b t c h w -> b c t h w')
            loss = model(obs, actions)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            
        accelerator.print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")
        
        if divisible_by(epoch, 5):
            visualizer.visualize(accelerator.unwrap_model(model), epoch, path_to_save = results_folder / f'grid_cells_epoch_{epoch}.png', probing_steps = 5000)
            accelerator.print(f"Grid cell visualization (epoch {epoch}) saved.")
    
    visualizer.visualize(accelerator.unwrap_model(model), 15, path_to_save = results_folder / 'grid_cells_final.png', probing_steps = 10000)
    
    obs, _, _ = world.generate_trajectory(steps = 1)
    sample_img = rearrange(obs[0], 'c h w -> h w c').numpy()
    Image.fromarray((sample_img * 255).astype(np.uint8)).save(results_folder / 'sample_view.png')
    accelerator.print("Extended simulation complete. Results saved to 'results/' folder.")

if __name__ == "__main__":
    run_simulation()
