import torch
from torch import nn
import torch.nn.functional as F

class Lenet5(nn.Module):
    '''
    for cifar10 dataset
    '''

    def __init__(self): #在这里完成对神经网络的构建
        super(Lenet5, self).__init__() #调用父类的方法来初始化类

        #卷积层
        self.conv_unit = nn.Sequential( #调用神经网络层操作序列
            #确定x的输入 x : [n,3,32,32]
            nn.Conv2d(3,6,kernel_size=(5,5),stride=(1,1),padding=(0,0)), #定义第一层2维卷积,x.shape --> [n,6,28,28]
            nn.AvgPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0)), #定义第一层池化层,x.shape --> [n,6,12,12]
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),  #定义第二层2维卷积层,x.shape --> [n,16,12,12]
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),  # 定义第一层池化层,x.shape --> [n,16,5,5]
            nn.Flatten(1,3) #自动将从参数1到参数2的维度数据相乘然后打平
        )

        #全连接层，进行分类操作
        self.fc_unit = nn.Sequential(
            #创建全连接层
            nn.Linear(16*5*5,120), #第一层全连接层
            #创建激活函数
            nn.ReLU(),
            nn.Linear(120,84), #第二层全连接层
            nn.ReLU(),
            nn.Linear(84,10) #第三层全连接层
        )

        #评价标准,用交叉熵来进行评价。通常分类问题都使用交叉熵
        self.criteon = nn.CrossEntropyLoss()

    #正向传播
    def foward(self,x):

        batchsz = x.size(0) #返回的是x.shape[0] 这里写x.size(0)是因为batchsz需要接收的是list数据？
        x = self.conv_unit(x)
        #输出logits logits.shape = [n,10]
        logits = self.fc_unit(x)
        #输出预测值
        #pred = F.softmax(logits, dim=1) #第二参数表示在第1维上做softmax
        #loss = self.criteon(logits,label)
        return logits

def main():

    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)  # fake batch
    logits = net.foward(tmp)
    print('LeNet Output:', logits.shape)

if __name__ == '__main__':
    main()