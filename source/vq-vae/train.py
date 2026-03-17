
from parser import get_args_parser
import data_loader
from model import VQ_VAE
import torch.optim as optim
from torch import nn
import torch
from tqdm import tqdm
from datetime import datetime
import os
import yaml

class Train:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataloader = data_loader.DATALoader(self.args)
        self.net = VQ_VAE(args=args)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
        self.Loss = ReConsLoss(self.args.recons_loss)

        self.start_iter = 0
    
    def update_lr_warm_up(self, optimizer, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        return optimizer, current_lr

    def save(self, iter):
        model = {
            'iter': iter,
            'args': self.args,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, 'vq_vae.pt')
        torch.save(model, path)

    def load(self, pth_path):
        if os.path.exists(pth_path):
            print(f"Loading checkpoint from {pth_path} ...")
            checkpoint = torch.load(pth_path, map_location=self.device, weights_only=False)
            
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_iter = checkpoint['iter']
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"Resumed from iteration {self.start_iter}")
        else:
            print(f"Checkpoint {pth_path} not found! Starting from scratch.")

    def run(self,):
        if self.args.resume is not None:
            self.load(self.args.resume)

        train_loader_iter = data_loader.cycle(self.dataloader)

        self.net.train()
        self.net.cuda()
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        avg_loss = 100.
        best_iter = -1
        #### ----- Warming ----- ####
        if self.start_iter < self.args.warm_up_iter:
            for nb_iter in tqdm(range(self.start_iter + 1, self.args.warm_up_iter + 1)):
                self.optimizer, current_lr = self.update_lr_warm_up(self.optimizer, nb_iter, self.args.warm_up_iter, self.args.lr)

                motion = next(train_loader_iter)
                motion = motion.to(self.device).float()
                
                pred_motion, loss_commit, perplexity = self.net(motion)
                loss_motion = self.Loss(pred_motion, motion)
                
                loss = loss_motion + self.args.commit * loss_commit
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_recons += loss_motion.item()
                avg_perplexity += perplexity.item()
                avg_commit += loss_commit.item() * self.args.commit

                if nb_iter % self.args.print_iter ==  0 :
                    
                    avg_recons /= self.args.print_iter
                    avg_perplexity /= self.args.print_iter
                    avg_commit /= self.args.print_iter
                    
                    print(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}", flush=True)
                    
                    if avg_recons  < avg_loss:
                        avg_loss = avg_recons 
                        best_iter = nb_iter
                    
                    # 重置累计变量
                    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
            self.start_iter = self.args.warm_up_iter

        ##### ---- Training ---- #####
        # 保存参数
        os.makedirs(self.args.out_dir, exist_ok=True)
        config_path = os.path.join(self.args.out_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(vars(self.args), f, sort_keys=False)

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        best_iter = -1
        
        for nb_iter in tqdm(range(self.start_iter + 1, self.args.total_iter + 1)):

            motion = next(train_loader_iter)
            motion = motion.to(self.device).float()

            pred_motion, loss_commit, perplexity = self.net(motion)

            loss_motion = self.Loss(pred_motion, motion)
            loss = loss_motion + self.args.commit * loss_commit

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            avg_recons += loss_motion.item()
            avg_perplexity += perplexity.item()
            avg_commit += self.args.commit * loss_commit

            if nb_iter % self.args.print_iter ==  0 :
                
                avg_recons /= self.args.print_iter
                avg_perplexity /= self.args.print_iter
                avg_commit /= self.args.print_iter
                
                print(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}", flush=True)
                
                if avg_recons  < avg_loss:
                    best_iter = nb_iter
                    avg_loss = avg_recons 
                    self.save(iter=nb_iter)
                print(f"Best Iter: {best_iter}")
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., :], motion_gt[..., :])
        return loss
    

def main():
    args = get_args_parser()
    args.training = True
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.out_dir = os.path.join(args.out_dir, current_time)
    train = Train(args)
    train.run()

if __name__ == "__main__":
    main()