import torch
from torch import nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            wider=False,
    ):
        super().__init__()
        self.net = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-4])
        if in_channels != 3 or wider:
            stride = (2,1) if wider else (2,2)
            new_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=stride, padding=(3, 3), bias=False)
            self.net[0] = new_conv
        self.out_dim = 128
        self.height_at_64px = 8

    def forward(self,x):
        return self.net(x)

if __name__ == "__main__":
    import torch
    x = torch.randn(10,1,64,1024)
    m = Resnet(wider=True)
    print(m(x).shape)
