'''定义整个YOLOv3结构'''

from collections import OrderedDict
import torch
import torch.nn as nn
from nets.darknet import darknet53
'''在nets文件夹下面含有__init__.py程序，程序内容可以为空。但程序必须存在，那么nets文件夹就变成了一个package，
   其他程序就可以导入这个package里的模块或函数'''

def conv2d(filter_in, filter_out, kernel_size): #定义了一个CBL结构
    pad = (kernel_size - 1) // 2 if kernel_size else 0 #两个斜杠是整除运算
    # 其实就是pad = 1 if (kernel_size == 3) else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   看YOLOv3结构图，不管是哪个尺度，最后的部分都有7个卷积层
#   make_last_layers里面一共有七个卷积，前五个用于提取特征,即CBL*5。
#   后两个用于获得yolo网络的预测结果,即CBL+Conv
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    '''filters_list传入含两个元素的列表，比如对第一个尺度，在经过最后一个Res4之后，输出通道数为1024，
    则传入的filters_list为[512, 1024]。in_filters为1024，out_filters根据数据集中的num_class和anchor数计算得到最最最后输出的特征层，
    比如COCO数据集的话，out_filters为255。在前5次卷积中，kernel_size为1时作用为降低通道数，为3时则提取特征。
    输出通道数在1024和512之间反复横跳。最后两次卷积用于获得预测结果'''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)#bias=True，因为后面没有BN层了。
    )
    return m
'''为什么卷积之后有BN层的话，卷积时的偏置可以不要？因为执行BN操作的时候，会减去均值。加上偏置反而会占内存'''

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256 out3
        #   26,26,512  out4
        #   13,13,1024  out5
        #---------------------------------------------------#
        self.backbone = darknet53() #获取darknet53并保存到网络结构里
        '''运行darknet53函数，返回构建的darknet53网络，而在darknet53网络的正向传播过程中会依次返回out3 out4 out5'''
        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #   YOLOv3一个grid cell产生3个anchor，所以len(anchors_mask[0])=len(anchors_mask[1])=len(anchors_mask[2])=3
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1) #定义一个CBL块
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest') #采用最邻近插值算法进行上采样，输出宽高为输入的2倍
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        '''out_filters[-2] + 256，是要传入最后包含7次的层的通道数'''


        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   这是正向传播过程，因此返回的是out3,out4,out5。shape依次为52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x) #self.backbone()是一个骨干特征网络，当传入输入的时候，自动调用里面的forward函数，因此就会返回out3,out4,out5
        '''https://blog.csdn.net/u011501388/article/details/84062483'''

        #---------------------------------------------------#
        #   第一个特征层  out0 = (batch_size,255,13,13)
        #---------------------------------------------------#

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0) #self.last_layer0一共有7层，这里取前5层，即01234层，不包括5
        out0        = self.last_layer0[5:](out0_branch) #这里包括5，即56层。经过这一层的两次卷积操作，通道数由512变成255
        '''python对列表的分片操作，小甲鱼书本P32'''

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1) #dim=1代表沿通道数方向叠加

        #---------------------------------------------------#
        #   第二个特征层  out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        #---------------------------------------------------#
        #   第三个特征层  out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2

