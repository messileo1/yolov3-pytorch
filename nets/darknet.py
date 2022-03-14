'''定义主干特征提取网络Darknet53'''
import math
from collections import OrderedDict
import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数减少参数量，加快运算。然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module): # 定义了一个Res unit
    def __init__(self, inplanes, planes):
        '''初始化的时候需要传入inplanes输入通道数，planes列表，包含两个元素，因为一个Res unit有两个CBL结构第一个CBL的out_channel是
        第二个CBL的in_channel，即planes[0]。而planes[1]是第二个CBL的out_channel'''
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        #经过这个卷积宽高不变。由于会进过BN层，bias一般设置为False
        self.bn1    = nn.BatchNorm2d(planes[0]) #BN层的输入为conv之后的输出通道数。Output: same shape as the input
        self.relu1  = nn.LeakyReLU(0.1) #0.1是x<0时候的斜率。Output: same shape as the input
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual #add操作必须保证out与residual的宽高通道数都相同
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):#只需传入一个列表layers，layers存放着ResX的个数。对darknet53，layers=[1, 2, 8, 8, 4]
        super(DarkNet, self).__init__()
        self.inplanes = 32 #416x416x3输入在经过Darknet53第一个CBL块之后的输出通道数为32
        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024] #记录一下，备用。

        # 进行权值初始化（就是一个盒子，不需理解，会用即可）
        '''在梯度下降的过程中，极易出现梯度消失或者梯度爆炸。因此，对权重w的初始化则显得至关重要，
        一个好的权重初始化虽然不能完全解决梯度消失和梯度爆炸的问题，但是对于处理这两个问题是有很大的帮助的，
        并且十分有利于模型性能和收敛速度'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks): #定义函数名前面有一个下划线，这个函数只能在DarkNet这个class内部调用
        '''plans是一个含两个元素的列表，分别代表ResX的输入与输出通道数，blocks代表有多少个残差组件，即X'''
        layers = []

        '''ResX结构中，通过CBL结构来改变宽高与通道数，Res unit并不改变宽高与通道数'''
        # 下采样，步长为2，卷积核大小为3。self.inplane=32、64、128等，即网络经过最开始的那个CBL层后的输出通道数
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        '''宽高减半，通道数由传入的plans知，加倍。注意，往空列表里面加的是元组而不是字典'''

        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]#一个ResX的Res unit的输出通道数等于前面那个CBL的输出通道数。
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers)) #实现对对象中元素的排序。有序字典一般用于动态添加并需要按添加顺序输出的时候

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # 由于要进行concat操作，这三个层要单独拿出来
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
