import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox(): #DecodeBox是用于对先验框anchor进行调整。因为对于input进行预测时，网络输出的是偏移量，所以还需利用偏移量进行对先验框anchor的调整。
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):#enumerate(inputs)返回inputs里面的索引以及索引对应的元素
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #   decode_box每一次只能对一个特征层的预测，因此要进行三次循环才能对3个特征层完成解码
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   计算步长以确定每一个特征点对应原图上多少个像素点。如果输入为416×416，对于13×13的特征图就对应416/13=32个像素点。
            #   输入为416x416时，stride_h = stride_w = 32、16、8（其实就是下采样倍数）
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   聚类得到的anchor的尺寸是相对于原图而言的，通过下面这样的操作把anchor的尺寸调整成相对于特征层的。
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   进行通道转换。比如原始的shape是batch_size，3*(5+num_classes),13,13。转换为batch_size,3,13,13,(5+num_classes)
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            #-----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
            #view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。

            '''13*13个grid cell,每个grid cell产生3个anchor，每个anchor深度方向对于coco数据集（80个类）而言是85，
            即tx,ty,tw,th,p,c1,c2,c3,......,c80。其中前四个是由网络预测出的用于调整先验框中心与宽高的输出，p是先验框置信度，c1--c80是类别置信度'''
            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])  #取到tx，作为sigmoid的输入，得到先验框中心横坐标的调整参数x
            y = torch.sigmoid(prediction[..., 1])  #取到ty，作为sigmoid的输入，得到先验框中心纵坐标的调整参数y
            '''python中冒号省略号的使用见：https://blog.csdn.net/weixin_40522801/article/details/106458186'''

            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #-----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4]) #预测出的85个数，除了预测的宽高调整参数外，都要进行sigmoid激活
            #-----------------------------------------------#
            #   种类置信度
            #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，首先是获取网格左上角的坐标
            #   格式：batch_size,3,13,13
            #----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #----------------------------------------------------------#
            #   再获取anchor的宽高
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从网格左上角向右下角偏移
            #   再调整先验框的宽高。
            #----------------------------------------------------------#
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs #存放的就是预测结果，但是是在特征图上，还需要映射到原图上去


    '''YOLO的前向过程：任意尺寸的原图（image_shape），先将长边缩放到416×416，得到new_shape，再加灰条得到416×416图像（input_shape）。
    经过YOLO网络得到3个特征图，比如13×13，解码得到为偏移值，解码得到的中心点的坐标是相对于网格尺寸的位置，宽高是相对于416*416尺寸的位置，都在0~1之间。
    yolo_correct_boxes的作用：对模型输出的box信息(x, y, w, h)进行校正,输出基于原图坐标系的box信息(x_min, y_min, x_max, y_max),实际坐标值，非比值。
    有两步，直接用用特征图解码的（x,y,w,h）对得到416×416上的box，再得到原图image_shape上的box。
    参考：https://blog.csdn.net/qq_41011242/article/details/108276552 
         https://www.cnblogs.com/monologuesmw/p/12794883.html'''
    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):

        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#

        #进行坐标翻转
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        #获得图像经过等比缩放以后的尺寸（没有灰边的）--- new_shape
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        # 将box的中心点、宽高调整至原图尺寸
        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        #将中心点、宽高信息转化为四个坐标点 xmin ymin xmax ymax
        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1],#y_min
                                 box_mins[..., 1:2],#x_min
                                 box_maxes[..., 0:1],#y_max
                                 box_maxes[..., 1:2]],#x_max
                                 axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes #此处的boxes已经是原图上的了（不带灰条的原图，任意尺寸的那个原图）

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        '''原图上的box，首先利用conf_thres进行第一轮筛选，再进行非极大值抑制。当然conf_thres=0.5, nms_thres=0.4
        都是默认值，是可以被修改的。nms_thres的值越小，nms越严格'''
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2 #中心y坐标
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2 #中心x坐标
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2 #高
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2 #宽
        prediction[:, :, :4] = box_corner[:, :, :4] #置信度

        output = [None for _ in range(len(prediction))] #for _ in range(n) 一般仅仅用于循环n次，不用设置变量，用_指代临时变量
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # torch.max()函数：https://blog.csdn.net/liuweiyuxiang/article/details/84668269
            #上一行代码在对第5个序号之后的内容取max，返回种类置信度以及类别的序号索引

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            #image_pred[:, 4]相当于条件，判断这个预测框里面包含物体的概率，再乘以种类置信度，获得总的置信度。判断他是否大于阈值conf_thres
            #----------------------------------------------------------#
            #   如果大于了阈值conf_thres，那么就获取其信息
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0): #image_pred.size(0)是num_anchor。如果没有框，直接进行下一张图片的处理
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            #堆叠的是预测的前5个序号的内容、种类置信度、种类的序号索引
            #------------------------------------------#
            #   获得预测结果中包含的所有种类。使得在for c in unique_labels循环里面只对包含的种类进行循环
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None: #yolo_correct_boxes得到最原始图像上的box信息。
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output #返回的是原图上box的信息，可以直接用来画图
