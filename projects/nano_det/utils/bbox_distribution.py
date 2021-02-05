import torch
import torch.nn.functional as F
from torch import nn

#这个类是讲预测的regmex个bbox，按照某种分布得到一个bbox
class BoxesDistribution(nn.Module):
    def __init__(self, reg_max=16):
        super(BoxesDistribution, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max-1, self.reg_max))
    def forward(self, x):
        x_ = x.reshape(-1, self.reg_max)
        x1 = F.softmax(x_, dim=1)

        x2 = F.linear(x1, self.project.type_as(x)).reshape(-1, 4)
        return x2