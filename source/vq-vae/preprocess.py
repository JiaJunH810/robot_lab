import torch
from tqdm import tqdm
import os
from parser import get_args_parser
import data_loader

def compute_mean_std(dataloader, args):
    print(f"Start calculating mean and std on device: {args.device}...")
    
    sum_x = None
    sum_sq_x = None
    total_samples = 0

    for batch in tqdm(dataloader, desc="Scanning dataset"):

        data = batch.view(-1, batch.shape[-1])
        
        if sum_x is None:
            feature_dim = data.shape[-1]
            print(f"Detected Feature Dim: {feature_dim}")
            if feature_dim != args.features:
                print(f"[Warning] Args.features ({args.features}) != Detected Dim ({feature_dim})")
            
            sum_x = torch.zeros(feature_dim, dtype=torch.float64, device=args.device)
            sum_sq_x = torch.zeros(feature_dim, dtype=torch.float64, device=args.device)

        data = data.to(args.device).double()
        
        sum_x += data.sum(dim=0)
        sum_sq_x += (data ** 2).sum(dim=0)
        total_samples += data.shape[0]

    mean = sum_x / total_samples

    variance = (sum_sq_x / total_samples) - (mean ** 2)
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)
    std[std < 1e-6] = 1.0

    return mean.float(), std.float()

def main():
    args = get_args_parser()
    args.window = 1
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    args.data_process = True
        
    dataloader = data_loader.DATALoader(args, batch_size=1024, num_workers=8)
    
    mean, std = compute_mean_std(dataloader, args)
    
    print("\nCalculation Done!")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape:  {std.shape}")
    print(f"First 5 Mean: {mean[:5]}")
    print(f"First 5 Std:  {std[:5]}")

    # 保存文件
    save_path = os.path.join(args.meta_folder, "mean_std.pt")
    save_dict = {
        "mean": mean.cpu(),
        "std": std.cpu()
    }
    torch.save(save_dict, save_path)
    print(f"Saved stats to: {save_path}")

if __name__ == "__main__":
    main()