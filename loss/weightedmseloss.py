from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        se = (output - target) ** 2
        mse = se.mean(axis=1).mean(axis=0)
        loss = mse * self.weights
        loss = loss.mean()
        return loss