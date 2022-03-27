import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv1:[2,3,224,224]=>[2,64,56,56]

        self.blk1 = ResBlk(64, 64)
        # [2, 64, 56, 56] => [2, 64, 56, 56]

        self.blk2 = ResBlk(64, 128, stride=2)
        # [2, 64, 56, 56]=>[2, 128, 28, 28]

        self.blk2_1 = ResBlk(128, 128)
        # [2, 128, 28, 28] =>[2, 128, 28, 28]

        self.blk3 = ResBlk(128, 256, stride=2)
        # [2, 128, 28, 28]=>[2, 256, 14, 14]

        self.blk3_1 = ResBlk(256, 256)
        # [2, 256, 14, 14]=>[2, 256, 14, 14]

        self.blk4 = ResBlk(256, 512, stride=2)
        # [2, 256, 14, 14]=>[2, 512, 7, 7]

        self.blk4_1 = ResBlk(512, 512)
        # [2, 512, 7, 7]=>[2, 512, 7, 7]

        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.outlayer = nn.Linear(512, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))  # conv1:[b,3,224,224]=>[b,64,56,56]
        # print(x.shape)

        x = self.blk1(x)  # [b, 64, 56, 56]=>[2, 64, 56, 56]

        x = self.blk1(x)  # [b, 64, 56, 56]=>[2, 64, 56, 56]

        x = self.blk2(x)  # [2, 64, 56, 56]=>[2, 128, 28, 28]

        x = self.blk2_1(x)  # [2, 128, 28, 28] =>[2, 128, 28, 28]

        x = self.blk3(x)  # [2, 128, 28, 28]=>[2, 256, 14, 14]

        x = self.blk3_1(x)  # [2, 256, 14, 14]=>[2, 256, 14, 14]

        x = self.blk4(x)  # [2, 256, 14, 14]=>[2, 512, 7, 7]

        x = self.blk4_1(x)  # [2, 512, 7, 7]=>[2, 512, 7, 7]

        x = self.pool2(x)  # [2, 512, 7, 7]=>[2,512,1,1]

        x = x.view(x.size(0), -1)  # flatten

        x = self.outlayer(x)

        return x


def main():
    model = ResNet18(7)
    tmp = torch.randn(1, 3, 224, 224)
    out = model(tmp)
    print('resnet:', out.shape)


if __name__ == '__main__':
    main()
