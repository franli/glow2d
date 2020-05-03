import torch
from glow2d import Glow


"""
Package the model with two interfaces for our purpose: sample and log_prob evaluation.
"""
class Flow(Glow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  

    def sample(self, args, device):
        with torch.no_grad():
            sample, _ = self.inverse(batch_size=args.batch, z_std=1.2)
            sample = sample.to(device)
            return sample

    def log_prob(self, x):
        z, logdet = self.forward(x)
        log_prob = self.base_dist.log_prob(z) + logdet
        return log_prob.unsqueeze(1)