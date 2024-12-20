import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, lora_dim, a_dim, b_dim, alpha):
        super().__init__()
        
        self.alpha = alpha
        self.Aaux = nn.Parameter(torch.randn(in_dim, a_dim), requires_grad=False)
        # self.Aux_init(self.Aaux)
        self.Atrain = nn.Parameter(torch.randn(a_dim, lora_dim))
        self.Btrain = nn.Parameter(torch.randn(lora_dim, b_dim))
        # nn.init.kaiming_uniform_(self.Btrain, a=math.sqrt(5))
        self.Baux = nn.Parameter(torch.randn(b_dim, out_dim), requires_grad=False)
        # self.Aux_init(self.Baux)

    def Aux_init(self, tensor):
        nn.init.orthogonal_(tensor)

    def forward(self, x):
        A = self.Aaux @ (self.Atrain @ self.Btrain) @ self.Baux
        return x @ A

class Lidb(torch.nn.Module):
    def __init__(self, linear, lora_dim, a_dim, b_dim, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, lora_dim, a_dim, b_dim, alpha
        )
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )

    def forward(self, x):
        lora_weight = self.lora.forward(x)
        combined_weight = self.linear.weight + self.lora.alpha * lora_weight.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        return F.linear(x, new_weight, self.linear.bias)
