# /// script
# dependencies = [
#   "fire",
#   "hippoformer",
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
# [tool.uv.sources]
# hippoformer = { path = "." }
# ///

# env setup

import os
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['PYOPENGL_PLATFORM'] = 'glfw'

# imports

from pathlib import Path

import torch
from torch import nn, Tensor, stack
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from einops import rearrange
from accelerate import Accelerator

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import fire
import gym
import memory_maze
import matplotlib.pyplot as plt
from PIL import Image

from hippoformer.hippoformer import Hippoformer, maze_sensory_enc_dec

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# memory maze environment wrapper

def find_physics(env):
    curr = env
    for _ in range(20):
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
        if not exists(self.physics):
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
            if action == 1:
                v_w[0] = 0.5
            elif action == 2:
                v_w[1] = -0.5
            elif action == 3:
                v_w[1] = 0.5

            actions.append(v_w)
            positions.append(torch.from_numpy(self.get_pos()).float())

            step_res = self.step(action)
            obs, done = step_res[0], step_res[2]
            if done:
                obs = self.reset()

        return (stack(observations) if not skip_obs else None), stack(actions), stack(positions)

# dataset

class TrajectoryDataset(Dataset):
    def __init__(self, world, num_trajectories = 32, steps = 100):
        self.data = [world.generate_trajectory(steps) for _ in range(num_trajectories)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# spatial cell (grid & place cell) visualization

def get_sac(rate_map: Tensor):
    mask = ~rate_map.isnan()
    if not mask.any():
        return torch.zeros_like(rate_map)

    m = rate_map.clone()
    mean = rate_map[mask].mean()
    m[mask] -= mean
    m[~mask] = 0.

    h, w = m.shape
    m_batch = rearrange(m, 'h w -> 1 1 h w')

    sac = F.conv2d(
        F.pad(m_batch, (w - 1, w - 1, h - 1, h - 1)),
        m_batch
    )
    return rearrange(sac, '1 1 h w -> h w')

def gaussian_blur_2d(img: Tensor, sigma: float = 1.0):
    ksize = int(2 * 3 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    x = torch.linspace(-3 * sigma, 3 * sigma, ksize)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]

    c = img.shape[0]
    kernel2d = rearrange(kernel2d, 'h w -> 1 1 h w').to(img.device)
    kernel2d = kernel2d.expand(c, 1, -1, -1)

    padding = ksize // 2
    img_padded = F.pad(rearrange(img, 'c h w -> 1 c h w'), (padding, padding, padding, padding), mode = 'reflect')
    blurred = F.conv2d(img_padded, kernel2d, groups = c)
    return rearrange(blurred, '1 c h w -> c h w')

class SpatialCellVisualizer:
    def __init__(
        self,
        world: MemoryMazeEnv,
        resolution: int = 35,
        spatial_range: tuple[float, float] = (-5.0, 5.0)
    ):
        self.world = world
        self.resolution = resolution
        self.spatial_range = spatial_range

    @torch.no_grad()
    def get_rate_maps(self, model: nn.Module, steps: int = 3000):
        model.eval()
        device = next(model.parameters()).device

        observations, actions, positions = self.world.generate_trajectory(steps = steps, skip_obs = False)

        observations = observations.to(device)
        actions = actions.to(device)
        positions = positions.to(device)

        actions_in = rearrange(actions, 't d -> 1 t d')
        obs_in = rearrange(observations, 't c h w -> 1 c t h w')

        target_model = model.mm_tem if hasattr(model, 'mm_tem') else model

        structural_codes = target_model.path_integrator(actions_in)
        structural_codes = rearrange(structural_codes, '1 t d -> t d')

        encoded_sensory = target_model.sensory_encoder(obs_in)
        encoded_sensory = rearrange(encoded_sensory, '1 t d -> t d')

        def compute_maps_for_codes(codes):
            res = self.resolution
            p_min, p_max = self.spatial_range

            indices = ((positions - p_min) / (p_max - p_min + 1e-5) * (res - 1)).long()
            indices = torch.clamp(indices, 0, res - 1)

            num_cells = codes.shape[-1]
            activations = torch.zeros((num_cells, res, res), device = device)
            counts = torch.zeros((res, res), device = device)

            flat_indices = indices[:, 0] * res + indices[:, 1]

            activations_flat = rearrange(activations, 'd h w -> d (h w)')
            activations_flat.index_add_(1, flat_indices, codes.T)

            counts_flat = counts.view(-1)
            counts_flat.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype = torch.float32))

            rate_maps = activations / rearrange(counts.clamp(min = 1), 'h w -> 1 h w')
            mask = counts < 1

            has_visits = (~mask).any()
            if has_visits:
                for i in range(num_cells):
                    rmap = rate_maps[i]
                    rmap[mask] = rmap[~mask].mean()

            rate_maps = gaussian_blur_2d(rate_maps, sigma = 1.0)

            rm_min = rearrange(rate_maps.amin(dim = (1, 2)), 'c -> c 1 1')
            rm_max = rearrange(rate_maps.amax(dim = (1, 2)), 'c -> c 1 1')

            rate_maps = (rate_maps - rm_min) / (rm_max - rm_min).clamp(min = 1e-5)
            rate_maps[:, mask] = float('nan')
            return rate_maps

        grid_rate_maps = compute_maps_for_codes(structural_codes)
        place_rate_maps = compute_maps_for_codes(encoded_sensory)

        return grid_rate_maps, place_rate_maps

    def visualize(
        self,
        model: nn.Module,
        epoch: int,
        path_to_save: str | Path,
        probing_steps: int = 3000
    ):
        path_to_save = Path(path_to_save)
        grid_maps, place_maps = self.get_rate_maps(model, steps = probing_steps)

        grid_maps_cpu = grid_maps.cpu()
        place_maps_cpu = place_maps.cpu()

        grid_vars = torch.from_numpy(np.nanvar(grid_maps_cpu.numpy(), axis = (1, 2)))
        top_grid_idx = torch.argsort(grid_vars, descending = True)[:4]

        place_vars = torch.from_numpy(np.nanvar(place_maps_cpu.numpy(), axis = (1, 2)))
        top_place_idx = torch.argsort(place_vars, descending = True)[:4]

        fig, axes = plt.subplots(4, 4, figsize = (14, 14), facecolor = 'white')

        cmap_rate = plt.get_cmap('rainbow').copy()
        cmap_rate.set_bad('white')

        for i, idx in enumerate(top_grid_idx):
            row = (i // 2)
            col_rate = (i % 2) * 2
            col_sac = col_rate + 1

            ax_rate = axes[row, col_rate]
            rate_map = grid_maps_cpu[idx]
            ax_rate.imshow(rate_map.numpy(), cmap = cmap_rate, interpolation = 'nearest', origin = 'lower')
            ax_rate.axis('off')
            ax_rate.set_title(f'grid cell {idx} (rate map)')

            ax_sac = axes[row, col_sac]
            sac = get_sac(rate_map)
            ax_sac.imshow(sac.numpy(), cmap = 'jet', interpolation = 'gaussian', origin = 'lower')
            ax_sac.axis('off')
            ax_sac.set_title(f'grid cell {idx} (sac)')

        for i, idx in enumerate(top_place_idx):
            row = 2 + (i // 2)
            col_place = (i % 2) * 2
            col_field = col_place + 1

            ax_p = axes[row, col_place]
            pmap = place_maps_cpu[idx]
            ax_p.imshow(pmap.numpy(), cmap = 'viridis', interpolation = 'gaussian', origin = 'lower')
            ax_p.axis('off')
            ax_p.set_title(f'place cell {idx} (rate map)')

            ax_f = axes[row, col_field]
            ax_f.imshow(pmap.numpy(), cmap = 'hot', interpolation = 'nearest', origin = 'lower')
            ax_f.axis('off')
            ax_f.set_title(f'place field {idx}')

        plt.tight_layout()
        plt.suptitle(f'in-silico grid cells (mec) & place cells (hpc) - epoch {epoch}', fontsize = 16)

        plt.savefig(path_to_save)
        plt.close()

# main simulation

def run_simulation(
    num_trajectories: int = 32,
    steps: int = 80,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    dim: int = 32,
    dim_structure: int = 64,
    dim_encoded_sensory: int = 32,
    dim_action: int = 2,
    probing_steps: int = 3000,
    probing_freq: int = 2,
    resolution: int = 35,
    spatial_range_min: float = -5.0,
    spatial_range_max: float = 5.0
):
    accelerator = Accelerator()
    accelerator.print(f"using device: {accelerator.device}")

    world = MemoryMazeEnv('MemoryMaze-9x9-v0')
    visualizer = SpatialCellVisualizer(
        world,
        resolution = resolution,
        spatial_range = (spatial_range_min, spatial_range_max)
    )

    model = Hippoformer(
        dim = dim,
        sensory_encoder_decoder = maze_sensory_enc_dec,
        dim_action = dim_action,
        dim_encoded_sensory = dim_encoded_sensory,
        dim_structure = dim_structure
    )

    optimizer = Adam(model.parameters(), lr = learning_rate)

    accelerator.print("generating 3d memory maze training trajectories...")
    dataset = TrajectoryDataset(world, num_trajectories = num_trajectories, steps = steps)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    results_folder = Path('results')
    results_folder.mkdir(parents = True, exist_ok = True)

    accelerator.print("starting training of hippoformer on 3d memory maze...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for obs, actions, _ in loader:
            obs = rearrange(obs, 'b t c h w -> b c t h w')
            loss = model(obs, actions)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

        accelerator.print(f"epoch {epoch}/{epochs}, loss: {total_loss / len(loader):.4f}")

        if divisible_by(epoch, probing_freq):
            visualizer.visualize(
                accelerator.unwrap_model(model),
                epoch,
                path_to_save = results_folder / f'grid_and_place_cells_epoch_{epoch}.png',
                probing_steps = probing_steps
            )
            accelerator.print(f"grid & place cell visualization (epoch {epoch}) saved.")

    visualizer.visualize(
        accelerator.unwrap_model(model),
        epochs,
        path_to_save = results_folder / 'grid_and_place_cells_final.png',
        probing_steps = probing_steps
    )

    obs, _, _ = world.generate_trajectory(steps = 1)
    sample_img = rearrange(obs[0], 'c h w -> h w c').numpy()
    Image.fromarray((sample_img * 255).astype(np.uint8)).save(results_folder / 'sample_view.png')
    accelerator.print("3d maze simulation complete! visualizations saved to 'results/' directory.")

if __name__ == "__main__":
    fire.Fire(run_simulation)

