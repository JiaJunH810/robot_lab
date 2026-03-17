from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import numpy as np
import torch
import utils
import bisect
import os

body_indexes = torch.tensor([ 0,  5, 15, 23,  6, 16, 24,  4, 13, 21, 25, 14, 22, 26])

class VQDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        mean_std_path = os.path.join(args.meta_folder, "mean_std.pt")
        if os.path.exists(mean_std_path):
            print(f"Loading mean/std from {mean_std_path}")
            stats = torch.load(mean_std_path, map_location='cpu')
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            print("[Warning] No mean_std.pt found! Data will NOT be normalized.")
            self.mean = None
            self.std = None

        self.files = glob.glob(f"{self.args.data_folder}/**/*.npz", recursive=True)
        self.lengths = []
        self.cum_lengths = [0]
        self.root_pos_height = []
        self.root_lin_vel_r = []
        self.root_ang_vel_r = []
        self.projected_gravity = []
        
        self.body_pos_r = []
        self.body_ori_r = []
        self.joint_pos = []
        self.joint_vel = []
        for file in tqdm(self.files, desc="Loading and Processing npz files..."):
            data = np.load(file, allow_pickle=True)

            root_pos = self.ToTensor(data["body_pos_w"][:, self.args.anchor_index, :])
            root_rot = self.ToTensor(data["body_quat_w"][:, self.args.anchor_index, :])
            root_lin_vel_w = self.ToTensor(data["body_lin_vel_w"][:, self.args.anchor_index, :])
            root_ang_vel_w = self.ToTensor(data["body_ang_vel_w"][:, self.args.anchor_index, :])
            body_pos_w = self.ToTensor(data["body_pos_w"][:, body_indexes, :])
            body_rot_w = self.ToTensor(data["body_quat_w"][:, body_indexes, :])
            joint_pos = self.ToTensor(data["joint_pos"])
            joint_vel = self.ToTensor(data["joint_vel"])

            self.root_pos_height.append(root_pos[..., -1:])
            self.root_lin_vel_r.append(utils.root_lin_vel_r(root_rot, root_lin_vel_w))
            self.root_ang_vel_r.append(utils.root_ang_vel_r(root_rot, root_ang_vel_w))
            self.projected_gravity.append(utils.projected_gravity(root_rot))
            self.body_pos_r.append(utils.body_pos_r(body_pos_w, root_pos, root_rot))
            self.body_ori_r.append(utils.body_ori_r(body_rot_w, root_rot))
            self.joint_pos.append(joint_pos)
            self.joint_vel.append(joint_vel)
            self.lengths.append(joint_pos.shape[0])
            self.cum_lengths.append(self.lengths[-1] + self.cum_lengths[-1])
    
    def ToTensor(self, array, dtype=torch.float32):
        return torch.tensor(array, dtype=dtype, device='cpu')

    def normalize(self, array):
        return (array - self.mean) / self.std
    
    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        frame_idx = idx - self.cum_lengths[file_idx]
        timesteps = torch.arange(frame_idx, frame_idx + self.args.window).clamp(max=self.lengths[file_idx] - 1)

        root_pos_height = self.root_pos_height[file_idx][timesteps]
        root_lin_vel_r = self.root_lin_vel_r[file_idx][timesteps]
        root_ang_vel_r = self.root_ang_vel_r[file_idx][timesteps]
        projected_gravity = self.projected_gravity[file_idx][timesteps]
        body_pos_r = self.body_pos_r[file_idx][timesteps]
        body_ori_r = self.body_ori_r[file_idx][timesteps]
        joint_pos = self.joint_pos[file_idx][timesteps]
        joint_vel = self.joint_vel[file_idx][timesteps]

        feature = torch.cat([
            root_pos_height,
            root_lin_vel_r,
            root_ang_vel_r,
            projected_gravity,
            body_pos_r.view(self.args.window, -1),
            body_ori_r.view(self.args.window, -1),
            joint_pos, joint_vel
        ], dim=-1)

        if self.mean is not None and not self.args.data_process:
            return self.normalize(feature)
        else:
            return feature


def DATALoader(args, batch_size = 512, num_workers = 8):
    dataset = VQDataset(args)
    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x