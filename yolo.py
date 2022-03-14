'''YOLO这个class只在预测和获取mAP的时候用'''
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = { # 必修改：model_path，classes_path。剩下的是可调整参数
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path（用的哪个网络权重）和classes_path（所要检测的类别）！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : './logs/ep020-loss15.143-val_loss18.281.pth',
        "classes_path"      : 'model_data/my_classes.txt', #需要和自己的数据集一样的类别
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.3,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小，越小说明nms越严格。可以在进行预测之前调整这个值然后运行看效果
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.4, #如果对图片进行检测发现没有结果，可能是这两个参数值设置得不合理
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n): #使用classmethod会使得YOLO这个class读取默认值（上面的信息）
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO（YOLO is a class）
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path) #返回txt文件里面包含的类名，以及多少个类
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path) #返回9个anchor
        # 返回的是一个对象，包含属性与方法，可以通过方法对属性进行解码，获得原图上的预测结果。
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)] #获得hsv格式的不同色度
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) # 获得RGB格式的不同颜色
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors)) # 通过hsv格式来调整不同类别对应边框的色度
        #生成一个列表，含num_classes种颜色，颜色以RGB通道值表示，比如(255, 229, 0)
        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolov3模型，载入yolov3模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2]) #计算输入图片的宽高
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别。返回resize之后的图像。
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data) #numpy-->torch
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs) #对网络的输出进行反解，得到特征图上的预测框
            #---------------------------------------------------------#
            #   将预测框进行堆叠，利用置信度和iou阈值进行预测框的筛选，返回的就是原图上的坐标信息。
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: #没检测到物体就返回原图
                return image
            #检测到了就获取种类、置信度、坐标
            top_label   = np.array(results[0][:, 6], dtype = 'int32') #取到矩阵所有行的第6列元素（列数从0开始）
            top_conf    = results[0][:, 4] * results[0][:, 5] #置信度×种类置信度
            top_boxes   = results[0][:, :4] #坐标
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        '''图像绘制看这段代码'
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (200, 100)) #创建一张指定尺寸的黑色图片,宽200，高100
        shape = [(40, 40), (80, 80)] #一个顶点的坐标，以及矩形对角线另一个顶点的坐标
        img1 = ImageDraw.Draw(img) #对原图img创建一个绘制对象
        img1.rectangle(shape, fill=(255, 229, 0), outline=(0, 229, 0))#轮廓与填充
        img.show()'''

        #下面这个for循环可以干很多事情。如果只想检测人不想检测其他类别，那么就在for循环内获取类别之后，加上判断if predicted_class='person':
        #再把下面获取坐标，画图部分的代码全都放到这个if条件句的执行体内就行了。即只有检测到的是person才获取坐标画图
        #如果想获取person类的总个数，可以在for循环外先定义sum=0,获取类别之后进行判断，if predicted_class='person': sum += 1
        #当for循环执行完，就得到总的人数sum，注意要在for循环外面输出这个sum
        '''
        person_sum = 0
        '''
        for i, c in list(enumerate(top_label)): #对一张图片进行重复检测，可以用于统计某个类别个数
            predicted_class = self.class_names[int(c)]
            '''
            if predicted_class='person':
                person_sum += 1
            '''
            box             = top_boxes[i]
            score           = top_conf[i]
            top, left, bottom, right = box


            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score) #这里得到原图预测出的种类及其得分
            draw = ImageDraw.Draw(image) #对原图创建了一个绘制对象，可以用于绘制文本、矩形等
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            #在此处获得的便是原图（任意尺寸）上的坐标信息，如下所示。

            '''
                0,0 ------------------------> x (width)
                 |
                 |  (Left,Top)
                 |      *_________
                 |      |         |
                 |      |         |
                 y      |_________|
              (height)            *
                            (Right,Bottom)
            '''


            if top - label_size[1] >= 0:#为了画表示类别的那个框
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c]) #画把目标框起来的大框，当然要考虑框的厚度。
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])#画用于表示类别（比如person，0.96）的边框
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)#写文本，从text_origin坐标处开始
            del draw
        '''
        print('总人数:',person_sum)    注意要在for循环执行完之后获取
        draw1 = ImageDraw.Draw(image)
        draw1.text((150, 150), '检测到总人数：' + str(person_sum), fill=(0, 0, 0), font=font)
        '''
        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):#test_interval用于指定图片的检测次数。越大FPS越准确
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time #检测一次图片所用时间，越小则FPS越快

    def get_map_txt(self, image_id, image, class_names, map_out_path):#生成txt文件
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: #没检测到就返回原图
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
