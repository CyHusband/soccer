import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Soccer(nn.Module):
    def __init__(self):
        super(Soccer, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for p in self.parameters():
            p.requires_grad = False
        self.resnet.fc = nn.Linear(2048, 2, bias=True)

    def forword(self, x):
        out = self.resnet(x)
        return out
