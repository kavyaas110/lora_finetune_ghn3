import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()

        if not isinstance(linear_layer, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")

        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.linear(x)

        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)

        return base + self.scaling * lora