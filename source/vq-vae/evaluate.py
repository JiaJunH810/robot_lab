from parser import get_args_parser
from data_loader import VQDataset
from model import VQ_VAE
import os
import torch
from tqdm import tqdm
import numpy as np
import yaml

class Evaluating:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset = VQDataset(args)
        self.net = VQ_VAE(args).to(args.device)

        model_path = os.path.join(args.log_dir, "vq_vae.pt")
        print(model_path)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            raise FileExistsError(f"No find {model_path}")
        
        self.net.eval()

    def run(self):
        file_code_idx = dict()
        file_code_idx["codebook"] = self.net.quantizer.codebook.detach().cpu().numpy()
        with torch.no_grad():
            for i, file_path in enumerate(tqdm(self.dataset.files, desc="Extracting Codes")):
                T = self.dataset.joint_pos[i].shape[0]

                feature_seq = torch.cat([
                    self.dataset.root_pos_height[i],
                    self.dataset.root_lin_vel_r[i],
                    self.dataset.root_ang_vel_r[i],
                    self.dataset.projected_gravity[i],
                    self.dataset.body_pos_r[i].view(T, -1),
                    self.dataset.body_ori_r[i].view(T, -1),
                    self.dataset.joint_pos[i],
                    self.dataset.joint_vel[i]
                ], dim=-1)
                feature_seq = (feature_seq - self.dataset.mean) / self.dataset.std
                padding_size = self.args.window - 1
                last_frame = feature_seq[-1:]
                padding = last_frame.repeat(padding_size, 1)
                feature_seq_padded = torch.cat([feature_seq, padding], dim=0)

                all_windows = feature_seq_padded.unfold(0, self.args.window, 1).permute(0, 2, 1)
                batch_size = 1024
                all_indices = []

                for b in range(0, all_windows.shape[0], batch_size):
                    batch_obs = all_windows[b : b + batch_size].to(self.device).float()

                    x_in = self.net.preprocess(batch_obs)
                    z = self.net.encoder(x_in)
                    z_permuted = z.permute(0, 2, 1).contiguous()
                    z_flat = z_permuted.view(-1, z.shape[1])
                    code_index = self.net.quantizer.quantize(z_flat)
                    code_index = code_index.view(batch_obs.shape[0], -1)
                    all_indices.append(code_index.cpu().numpy().astype(np.int16))
                
                file_code_idx[file_path] = np.concatenate(all_indices, axis=0)
                print(f"Files: {file_path}, file shape: {feature_seq.shape}, code shape: {file_code_idx[file_path].shape}")
        
        save_path = os.path.join(self.args.log_dir, self.args.codebook)
        np.savez_compressed(save_path, **file_code_idx)

def main():
    args = get_args_parser()
    args.log_dir = "/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs/2026-02-18_14-06-53"
    config_path = os.path.join(args.log_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                setattr(args, key, value)

    args.training = False
    args.data_folder = "/home/ubuntu/projects/hjj-robot_lab/source/motion/"
    args.codebook = "codebook_interp.npz"
    eval = Evaluating(args)
    eval.run()

if __name__ == '__main__':
    main()