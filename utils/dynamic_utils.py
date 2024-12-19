import numpy as np
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
import os
import roma


class unicycle(torch.nn.Module):

    def __init__(self, train_timestamp, centers, eulers, heights=None):
        super(unicycle, self).__init__()
        self.train_timestamp = train_timestamp
        self.delta = torch.diff(self.train_timestamp)

        self.input_a = centers[:, 0].clone()
        self.input_b = centers[:, 1].clone()

        self.a = nn.Parameter(centers[:, 0])
        self.b = nn.Parameter(centers[:, 1])
        
        diff_a = torch.diff(centers[:, 0]) / self.delta
        diff_b = torch.diff(centers[:, 1]) / self.delta
        v = torch.sqrt(diff_a ** 2 + diff_b**2)
        
        self.v = nn.Parameter(F.pad(v, (0, 1), 'constant', v[-1].item()))
        self.pitchroll = eulers[:, :2]
        self.yaw = nn.Parameter(eulers[:, -1] + (torch.pi / 2))

        if heights is None:
            self.h = torch.zeros_like(train_timestamp).float()
        else:
            self.h = heights

    def acc_omega(self):
        acc = torch.diff(self.v) / self.delta
        omega = torch.diff(self.yaw) / self.delta
        acc = F.pad(acc, (0, 1), 'constant', acc[-1].item())
        omega = F.pad(omega, (0, 1), 'constant', omega[-1].item())
        return acc, omega

    def forward(self, timestamp):
        if timestamp < self.train_timestamp[0]:
            delta_t = self.train_timestamp[0] - timestamp
            a = self.a[0] - delta_t * torch.cos(self.yaw[0]) * self.v[0]
            b = self.b[0] - delta_t * torch.sin(self.yaw[0]) * self.v[0]
            return a, b, self.v[0], self.pitchroll[0], self.yaw[0] - (np.pi / 2), self.h[0]
        elif timestamp > self.train_timestamp[-1]:
            delta_t = timestamp - self.train_timestamp[-1]
            a = self.a[-1] + delta_t * torch.cos(self.yaw[-1]) * self.v[-1]
            b = self.b[-1] + delta_t * torch.sin(self.yaw[-1]) * self.v[-1]
            return a, b, self.v[-1], self.pitchroll[-1], self.yaw[-1] - (np.pi / 2), self.h[-1]
        idx = torch.searchsorted(self.train_timestamp, timestamp, side='left')
        if self.train_timestamp[idx] == timestamp:
            return self.a[idx], self.b[idx], self.v[idx], self.pitchroll[idx], self.yaw[idx] - (np.pi / 2), self.h[idx]
        else:
            prev_timestamps = self.train_timestamp[idx-1]
            delta_t = timestamp - prev_timestamps
            prev_a, prev_b = self.a[idx-1], self.b[idx-1]
            prev_v, prev_yaw = self.v[idx-1], self.yaw[idx-1]
            acc, omega = self.acc_omega()
            v = prev_v + acc[idx-1] * delta_t
            yaw = prev_yaw + omega[idx-1] * delta_t
            a = prev_a + prev_v * ((torch.sin(yaw) - torch.sin(prev_yaw)) / (omega[idx-1] + 1e-6))
            b = prev_b - prev_v * ((torch.cos(yaw) - torch.cos(prev_yaw)) / (omega[idx-1] + 1e-6))
            h = self.h[idx-1]
        return a, b, v, self.pitchroll[idx-1], yaw - (np.pi / 2), h

    def capture(self):
        return (
            self.a,
            self.b,
            self.v,
            self.pitchroll,
            self.yaw,
            self.h,
            self.train_timestamp,
            self.delta
        )
    
    def restore(self, model_args):
        (
            self.a,
            self.b,
            self.v,
            self.pitchroll,
            self.yaw,
            self.h,
            self.train_timestamp,
            self.delta
        ) = model_args

    def visualize(self, save_path, noise_centers=None, gt_centers=None):
        a = self.a.detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()
        yaw = self.yaw.detach().cpu().numpy()
        plt.scatter(a, b, marker='x', color='b')
        plt.quiver(a, b, np.ones_like(a) * np.cos(yaw), np.ones_like(b) * np.sin(yaw), scale=20, width=0.005)
        if noise_centers is not None:
            noise_centers = noise_centers.detach().cpu().numpy()
            plt.scatter(noise_centers[:, 0], noise_centers[:, 1], marker='o', color='gray')
        if gt_centers is not None:
            gt_centers = gt_centers.detach().cpu().numpy()
            plt.scatter(gt_centers[:, 0], gt_centers[:, 1], marker='v', color='g')
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()

    def reg_loss(self):
        reg = 0
        acc, omega = self.acc_omega()
        reg += torch.mean(torch.abs(torch.diff(acc))) * 0.01
        reg += torch.mean(torch.abs(torch.diff(omega))) * 0.1
        reg_a_motion = self.v[:-1] * ((torch.sin(self.yaw[1:]) - torch.sin(self.yaw[:-1])) / (omega[:-1] + 1e-6)) 
        reg_b_motion = -self.v[:-1] * ((torch.cos(self.yaw[1:]) - torch.cos(self.yaw[:-1])) / (omega[:-1] + 1e-6))
        reg_a = self.a[:-1] + reg_a_motion
        reg_b = self.b[:-1] + reg_b_motion
        reg += torch.mean((reg_a - self.a[1:])**2 + (reg_b - self.b[1:])**2) * 1
        return reg
    
    def pos_loss(self):
        return torch.mean((self.a - self.input_a) ** 2 + (self.b - self.input_b) ** 2) * 10


def create_unicycle_model(train_cams, model_path, opt_iter=0, data_type='kitti'):
    unicycle_models = {}
    if data_type == 'kitti':
        cameras = [cam for cam in train_cams if 'cam_0' in cam.image_name]
    elif data_type == 'waymo':
        cameras = [cam for cam in train_cams if 'cam_1' in cam.image_name]
    elif data_type == 'nuscenes':
        cameras = [cam for cam in train_cams if (('CAM_FRONT' in cam.image_name) and ('LEFT' not in cam.image_name) and ('RIGHT' not in cam.image_name))]
    else:
        raise NotImplementedError    

    all_centers, all_heights, all_eulers, all_timestamps = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    seq_timestamps = []
    for cam in cameras:
        t = cam.timestamp
        seq_timestamps.append(t)
        for track_id, b2w in cam.dynamics.items():
            all_centers[track_id].append(b2w[[0, 2], 3])
            all_heights[track_id].append(b2w[1, 3])
            eulers = roma.rotmat_to_euler('xzy', b2w[:3, :3])
            all_eulers[track_id].append(eulers)
            all_timestamps[track_id].append(t)

    for track_id in all_centers.keys():
        centers = torch.stack(all_centers[track_id], dim=0).cuda()
        timestamps = torch.tensor(all_timestamps[track_id]).cuda()
        heights = torch.tensor(all_heights[track_id]).cuda()
        eulers = torch.stack(all_eulers[track_id]).cuda()

        model = unicycle(timestamps, centers.clone(), eulers.clone(), heights.clone())
        l = [
            {'params': [model.a], 'lr': 1e-3, "name": "a"},
            {'params': [model.b], 'lr': 1e-3, "name": "b"},
            {'params': [model.v], 'lr': 1e-3, "name": "v"},
            {'params': [model.yaw], 'lr': 1e-4, "name": "yaw"},
        ]

        optimizer = optim.Adam(l, lr=0.0)

        t_range = tqdm(range(opt_iter), desc=f"Fitting {track_id}")
        for iter in t_range:
            loss = 5e-3 * model.reg_loss() + 1e-3 * model.pos_loss()
            t_range.set_postfix({'loss': loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        unicycle_models[track_id] = {'model': model, 
                                    'optimizer': optimizer,
                                    'input_centers': centers}
    
    os.makedirs(os.path.join(model_path, "unicycle"), exist_ok=True)
    for track_id, unicycle_pkg in unicycle_models.items():
        model = unicycle_pkg['model']
        optimizer = unicycle_pkg['optimizer']
        
        model.visualize(os.path.join(model_path, "unicycle", f"{track_id}_init.png"))

    return unicycle_models