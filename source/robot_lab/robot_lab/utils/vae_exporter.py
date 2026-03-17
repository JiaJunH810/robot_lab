import torch
import torch.nn as nn
import copy
import os

class VaeJitPolicyExporter(nn.Module):
    """
    专门为 VaeActor 设计的 JIT 导出器，支持 (obs, code) 双输入。
    """
    def __init__(self, policy, normalizer=None):
        super().__init__()
        # 复制 Actor 网络
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        else:
            # 如果 policy 本身就是 actor
            self.actor = copy.deepcopy(policy)
            
        # 处理 Normalizer (通常只归一化 obs，不归一化 code)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()

    def forward(self, obs, code):
        # 1. 对 obs 进行归一化
        obs_norm = self.normalizer(obs)
        # 2. 将归一化后的 obs 和 code 一起传给 actor
        return self.actor(obs_norm, code)

    def export(self, path, filename="policy.pt"):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        self.to("cpu")
        
        # 使用 JIT Script 进行编译
        # JIT 会自动解析 forward 的签名，识别出需要 obs 和 code 两个输入
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(file_path)
        print(f"Successfully exported VAE policy to: {file_path}")
    
class VaeOnnxPolicyExporter(nn.Module):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        else:
            self.actor = copy.deepcopy(policy)
            
        if hasattr(self.actor, "code_dim"):
            self.code_dim = self.actor.code_dim
        else:
            raise ValueError(
                "Could not find 'code_dim' in actor. "
                "Please add `self.code_dim = num_vqvae` to your VaeActor __init__ method."
            )
            
        if hasattr(self.actor, "obs_dim"):
            self.obs_dim = self.actor.obs_dim
        else:
            raise ValueError("Actor must have 'obs_dim' attribute.")

        # 4. 复制 Normalizer
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()

    def forward(self, obs, code):
        return self.actor(self.normalizer(obs), code)

    def export(self, path, filename):
        # ... (和之前一样，使用 self.code_dim 即可) ...
        self.to("cpu")
        dummy_obs = torch.zeros(1, self.obs_dim)
        dummy_code = torch.zeros(1, self.code_dim) # 这里自动使用了读取到的维度
        
        full_path = os.path.join(path, filename)
        torch.onnx.export(
            self,
            (dummy_obs, dummy_code),
            full_path,
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs", "code"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "code": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )