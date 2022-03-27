import torch
from torch import nn
import torch.nn.functional as F


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        # 两层Block
        self.conv_unit = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 2, 1),  # [n,ch_in,32,32] --> [n,ch_out,16,16]
            nn.BatchNorm2d(ch_out),  # keeps shape
            nn.ReLU(),  # active
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),  # [n,ch_out,16,16] --> [n,ch_out,16,16]
            nn.BatchNorm2d(ch_out)  # keeps shape
        )

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 1, 2, 0),  # keep the shape of chin and chout same.
            nn.BatchNorm2d(ch_out)
        )

    def foward(self, x):
        out = self.conv_unit(x)
        # shorcut
        out = out + self.extra(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [2,64,32,32] to [2,64,32,32]
            nn.BatchNorm2d(64),  # [2,64,32,32] to [2,64,32,32]
            nn.ReLU()
        )

        # 添加4个ResNet模块
        self.blk1 = ResBlk(64, 128)  # [2,64,32,32] to [2,128,16,16]
        self.blk2 = ResBlk(128, 256)  # [2,256,8,8]
        self.blk3 = ResBlk(256, 512)  # [2,512,4,4]
        self.blk4 = ResBlk(512, 512)  # [2,512,2,2]
        self.flat = nn.Flatten()  # [2,512*2*2]
        self.outlayer = nn.Linear(512*2*2, 10)  # [2,10]

    def foward(self, x):
        x = self.conv1(x)
        #print("x1的shape",x.shape)
        x = self.blk1.foward(x)
        #print(x.shape)
        x = self.blk2.foward(x)
        x = self.blk3.foward(x)
        x = self.blk4.foward(x)  #
        x = self.flat(x)
        x = self.outlayer(x)

        return x


def main():
    res = ResNet18()
    tmp = torch.randn(2, 64, 32, 32)
    out = res.foward(tmp)
    print("resnet out shape", out.shape)


if __name__ == '__main__':
    main()
