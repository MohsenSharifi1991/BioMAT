import torch
from torch import nn


class WeightedMultiOutputLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMultiOutputLoss, self).__init__()
        self.weights = weights
        self.regression = nn.MSELoss()
        self.classification = nn.CrossEntropyLoss()

    def forward(self, output, target):
        w_cls = 1
        w_reg = 1
        regression_loss = self.regression(output[0], target[0])
        classification_loss = self.classification(output[1], target[1])
        loss = w_reg*regression_loss + w_cls*classification_loss
        return loss