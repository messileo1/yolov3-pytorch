'''这个程序在定义训练时候需要用到的损失函数、正负样本与忽略样本（用iou判定）、权重初始化方法等'''

import torch
import torch.nn as nn
import math
import numpy as np

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        '''input_shape是指reshape到的形状，比如416×416，
        cuda是布尔类型，用于判断是否使用GPU'''
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.ignore_threshold = 0.5 #anchor与GT的iou小于0.5就为负样本，大于0.5但不是最大就是忽略样本，有与GT最大的iou的anchor就是正样本，负责拟合GT
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        '''该方法的作用是：输入一个张量t，把t中的每一个元素的值都压缩在t_min和t_max之间。小于t_min的让它等于t_min，大于t_max的元素值等于t_max。
        介于t_min, t_max之间的元素则保持不变。
        例子：https://blog.csdn.net/york1996/article/details/89434935'''
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2) #对输入的pred - target中的每个元素求二次幂，并返回一个带有结果的张量

    def BCELoss(self, pred, target): #二分类交叉熵损失函数，不过此时是多标签的。比如标签[1，0]代表有猫没有狗
        epsilon = 1e-7 #希腊字母，非常非常小的正数，当然不能等于0，因为要进行对数运算。因为标签值只有1（正样本）和0（负样本）
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon) #pred介于epsilon和（1.0 - epsilon）之间。target要么1要么0
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def forward(self, l, input, targets=None):
        '''forward的作用是：只要在实例化一个对象中传入对应的参数就可以自动调用forward函数
        比如 module = Module()
            module(data)
            而不是使用module.forward(data)'''
        #----------------------------------------------------#
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input是网络的输出，有3个，即三个特征图
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #   13和13
        #--------------------------------#
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #   对于13×13特征图，stride_h和stride_w都是32。
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors] #a_w、a_h是在原图上的anchor宽高
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3*(5+num_classes), 13, 13 => batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        #-----------------------------------------------#
        #   网络预测的先验框的中心位置的调整参数，要送入损失函数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0]) # 用sigmoid将x,y压缩到[0,1]区间內，可以有效的确保目标中心处于执行预测的网格单元中，防止偏移过多
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   网络预测的先验框的宽高调整参数，要送入损失函数
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   网络预测的置信度，有物体的概率，要送入损失函数
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   网络预测的，在有物体的条件下，每个种类的置信度，要送入损失函数
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-----------------------------------------------#
        #   获得网络应该有的预测结果.看get_target的源码。
        #-----------------------------------------------#
        # 这个函数的作用是返回y_true（正样本，但是是网络应该学习的offset值）, noobj_mask（哪些anchor是包含物体的哪些不含物体，包含则为0不含则为1，就是损失函数前面乘的非0即1的系数）, box_loss_scale
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        #---------------------------------------------------------------#
        #   noobj_mask只能说明哪些anchor是包含物体的哪些不含，但是在包含物体的anchor里面，iou与GT过大但是不是最大的那些anchor
        #   作为负样本不合适，需要被忽略掉，于是有了get_ignore函数，因此运行下面这一行代码，返回的noobj_mask就是说明哪些是负样本。
        #----------------------------------------------------------------#
        noobj_mask = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.cuda() #正样本的offset值
            noobj_mask      = noobj_mask.cuda() #代表哪些样本是负样本
            box_loss_scale  = box_loss_scale.cuda()
        #-----------------------------------------------------------#
        #   表示真实框的宽高，二者均在0-1之间
        #   真实框越大，比重越小，小框的比重更大。假如框很小，bbox_loss_scale趋向于2，那么就增大了小框loss的权重
        #-----------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale
        #-----------------------------------------------------------#
        #   计算中心偏移情况的loss，使用BCELoss效果好一些
        #-----------------------------------------------------------#
        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4]) #为什么还要再乘以置信度y_true[..., 4]？
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        #-----------------------------------------------------------#
        #   计算宽高调整值的loss
        #-----------------------------------------------------------#
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4]) #0.5是方便求导时约掉平方项产生的系数
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])
        #-----------------------------------------------------------#
        #   计算置信度的loss。即正样本的置信度损失加上负样本的置信度损失
        #-----------------------------------------------------------#
        loss_conf   = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                      torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)
        #   类别loss
        loss_cls    = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss        = loss_x  + loss_y + loss_w + loss_h + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos)) #正样本的数量
        return loss, num_pos

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def get_target(self, l, targets, anchors, in_h, in_w): #这个函数的作用是返回y_true（正样本），但是y_true不是正样本的坐标而是网络应该学习的offset。 noobj_mask（哪些anchor是包含物体的哪些不含物体，包含则为0不含则为1）、box_loss_scale
        # target:第一个维度是batch_size，第二个维度是每一张图片里面真实框的数量，第三个维度内部是真实框的信息，包括位置以及种类
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs              = len(targets)
        #-----------------------------------------------------#
        #   构造全1张量作为noobj_mask，表示不包含目标的先验框，[batch_size, 3, 特征层高，特征层宽]，发现正样本在指定位置填0
        #-----------------------------------------------------#
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        '''一个特征图有13×13×3个anchor。再乘以batch_size。得到元素1有多少个。
        noobj_mask表示没有物体落在特征图中某一个grid cell的索引,所以在初始化的时候置1。在后面进行调整：如果没有物体落在那个cell中，那个对应的位置会置0。
        同时，如果预测的IOU值过大，（大于阈值ignore_thres）时，那么可以认为这个cell是有物体的，要置0。'''
        #-----------------------------------------------------#
        #   让网络更加去关注小目标（小目标权重增强，大目标权重减弱）
        '''box_loss_scale 这个值，它等于2-groundtruth.w * groundtruth.h，这个值是用于平衡大小目标之间的损失不均的问题，
        因为小目标在中心与宽高的损失和大目标在中心与宽高的损失不相同，大的目标的检测框的预测相对偏移值由于和大的anchor相比，
        所以相对值较小，但是小目标和小的anchor相比，所以小目标对于坐标和高宽的精确程度更加严格，所以采用box_loss_scale 来加重其损失权重。
        https://blog.csdn.net/qq_34199326/article/details/84109828'''
        #-----------------------------------------------------#
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   构造全零张量作为y_true，表示包含目标的先验框，[batch_size, 3, 特征层宽，特征层高，5 + num_classes]，发现正样本在指定位置的[...,4]填1
        #-----------------------------------------------------#
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        '''torch.ones和torch.zeros都是创建一个tensor，tensor的shape由参数决定，但是里面的元素都是0或者1'''
        for b in range(bs):  #对每一个batch_size循环，所以这个for循环内部在对每张图片单独进行处理。
            if len(targets[b])==0: #如果target里的第b张图片没有ground truth，就跳过当前图片。
                continue
            batch_target = torch.zeros_like(targets[b]) #根据给定张量，生成与其形状相同的全0张量。用于获取每张图片GT的情况
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点以及宽高
            #-------------------------------------------------------#
            '''对打的标签，是给出的x_min、y_min、x_max、y_max，经过神经网络，预测的都是相对值，比如某个点预测的是（0.3，0.4），
            说明说明该点在整张图片的实际坐标为（0.3×原图宽，0.4×原图高）。同理如果想获得GT在13×13特征图上的坐标，需要对预测值分别乘以特征图的宽高，就像下面这样'''
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w #获取左上角和右下角点的横坐标
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h #获取左上角和右下角点的纵坐标
            batch_target[:, 4] = targets[b][:, 4] #获取标签，非0即1
            batch_target = batch_target.cpu() #tensor有CPU类型（tensor.cpu）和GPU（tensor.cuda）类型，可以相互转化
            '''对tensor/numpy的索引，可以自己用代码看索引的是啥：
                import numpy as np
                a = np.random.randn(5,5)
                print(a)
                print(a[:, [0,2]])
                print(a[:, [1,3]])
                print(a[:, 4])'''
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式,构成ground truth
            #   num_true_box, 4。其中num_true_box代表ground truth的类别（属于第几个类），4代表GT的xmin,xmanx,ymin,ymax
            #   bacth_target为target转换为对应特征层，前两位是gt转换后的中心点坐标，[2:4]是宽高，[4]是类别索引
            #-------------------------------------------------------#
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4。9个anchor，每个都有四个坐标参数。
            #-------------------------------------------------------#
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比。
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9],代表每一个真实框和9个anchor的重合情况
            #   best_ns有两个部分的内容:
            #   [每个真实框中最大的重合度max_iou, 每一个真实框中最重合的先验框的序号]
            #-------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            '''上一行代码的解释：一张特征图，假设上面有很多个标注的GT，比如有4个标注的。那么对每一个GT都会计算找到9个iou值，
            然后找到最大iou及其对应的那个anchor，只返回该anchor的索引。4个标注的GT就会返回4个anchor索引。比如返回的索引是1,4,6，7。'''

            for t, best_n in enumerate(best_ns): # 当l=0时，计算13×13这个特征图上的所有损失。
                if best_n not in self.anchors_mask[l]: #self.anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]。若当前特征图为13×13，则对应的anchor索引为6,7,8。表示上面的1,4,6，7只有6,7会进入下面的计算
                    continue
                #----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框。返回的是anchor的序号，比如对于13×13特征图来说，只可能是6,7,8中的一个。
                #----------------------------------------#
                k = self.anchors_mask[l].index(best_n) #self.anchors_mask[0]=[6,7,8]，通过这一步判断是6还是7还是8（返回的k分别是0,1,2，具体看index方法的用法）
                #----------------------------------------#
                #   获得真实框属于哪个网格点。因为对yolov3，如果GT的中心落在某个grid cell里面，那么应该由该cell左上角的那个网格点进行预测
                #   下面计算ij就是在取整，就是帮助找到那个网格点
                #   对于进入计算的anchor，把每个anchor（x,y,w,h,p）堆叠起来：
                #                     （x,y,w,h,c）
                #                         ......
                #                     （x,y,w,h,c）
                #                     （x,y,w,h,c）
                #   i依次把所有x取了出来，j依次把所有y取了出来
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long() #torch.floor的作用是返回一个新tensor，里面的元素是输入元素向下取整
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   取出真实框的种类，也是一个索引，比如coco数据集，该标签的类是boat，则c=8，代表是是第8个类。
                #----------------------------------------#
                c = batch_target[t, 4].long()

                #----------------------------------------#
                #   noobj_mask代表无目标的特征点，置信度标签为0，也就是负样本，即背景信息。
                #----------------------------------------#
                noobj_mask[b, k, j, i] = 0 #只将那么多个1中的一个变为了0。由于为1的时候代表不包含物体，那么变为0说明对应anchor是包含物体的
                #----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #----------------------------------------#
                '''参考博文：https://blog.csdn.net/qq_34199326/article/details/84109828'''
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float() #代表x方向应该调整多少，相当于真实值，供网络学习
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float() #代表y方向应该调整多少，相当于真实值
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0]) #代表h方向应该调整多少，相当于真实值
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1]) #代表w方向应该调整多少，相当于真实值
                y_true[b, k, j, i, 4] = 1 #说明当前先验框内部是有包含物体的，因此是置信度标签设置为1
                y_true[b, k, j, i, c + 5] = 1 # 先验框内部包含的物体的种类。比如对于coco的80个类，当标签是boat的时候，将代表类别的c1...c80中第八个置为1，其他为0。
                                                #这里c+5是因为前面会添加5个真实值，所以要往后挪5个位置

                #   y_true： 形式为[batch_size, 3, 特征层宽，特征层高，5 + num_classes]， 存放着正样本框的中心点和宽高、类别索引、各类别的编码。
                #----------------------------------------#
                #   box_loss_scale其实算的是特征图上的GT相对于整个特征图的面积，之后会重新赋值的，即box_loss_scale=2-box_loss_scale
                #   用于平衡大小目标之间的损失不均的问题。起到了L2正则化的作用
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale


    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        #-----------------------------------------------------#
        #   对anchors_mask[l]这个anchor，获取其左上角的坐标
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成anchor的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高。判断忽略样本的方法是计算GT与anchor的IOU，但是各版本代码实现方式不太一样。此处是以调整之后的预测框与GT计算IOU来判断。
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        
        for b in range(bs):
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0: # 如果图片target[b]上的真实框数量大于0
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   获取ground truth在特征图上的位置。即横纵坐标的最大最小共4个值。必然是获取的这四个值而不可能是中心坐标。
                #   因为下一步要计算iou，而iou计算用的是顶点坐标而不是中心坐标
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4]
                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)#返回最大值及索引
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask #noobj_mask是说明某个anchor内是否含物体，iou大于self.ignore_threshold说明有物体，但iou不是最大，所以就被忽略了，剩下的就是负样本，负样本被return回来计算损失

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
