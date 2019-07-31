import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, weight, input, target):
        return (weight*F.mse_loss(input,target) + (1-weight)*(1/F.mse_loss(input, target)))

#Adds the new Loss component to the loss criterion. Checks if the current criterion is the single TripletLoss (no anchor), or if it is already a list of criterions (anchors already present)
def add_customLoss(criterions):
        criterions.append(CustomLoss())
        return criterions