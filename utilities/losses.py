import torch
import torch.nn as nn

class ObjectiveHingeLoss(nn.Module):
    def __init__(self, loss_margin):
        super(ObjectiveHingeLoss, self).__init__()
        self.loss_margin = loss_margin
        self.relu = nn.ReLU()

    def forward(self, y_hat, y):
        ## y_hat are the predicted ones
        indexes_pos = (y > 0).nonzero().squeeze()
        indexes_neg = (y <= 0).nonzero().squeeze()

        return self.relu(self.loss_margin - y_hat[indexes_pos].max() + y_hat[indexes_neg].max())
