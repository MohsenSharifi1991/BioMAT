import torch
from torch import nn


class WeightedRMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedRMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        se = (output - target) ** 2
        mse = se.mean(axis=1).mean(axis=0)
        rmse = torch.sqrt(mse)
        loss = rmse * self.weights
        loss = loss.mean()
        return loss