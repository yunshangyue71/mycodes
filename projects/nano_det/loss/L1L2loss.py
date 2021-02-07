import torch
def Regularization(model):
    L1=0
    L2=0
    for param in model.parameters():
        L1+=torch.sum(torch.abs(param))
        L2+=torch.norm(param,2)
    return L1,L2