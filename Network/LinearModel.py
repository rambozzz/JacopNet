import torch.nn as nn

def get_model(dimIn, dimOut):
    model = nn.Linear(dimIn, dimOut)
    return model

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        print("Initializing weights for: " + str(m))
        nn.init.xavier_normal_(m.weight.data, 1)