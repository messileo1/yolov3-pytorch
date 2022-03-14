import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

#-----------------------------------------------------------
#   自己定义数据集，首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，
#   然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取。
#-----------------------------------------------------------
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape #如416×416
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train


    #父类中的两个私有成员函数必须被重载，否则将会触发错误提示。继承Dataset必须继承__init_()和__getitim__()
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1)) #变换坐标轴
        box         = np.array(box, dtype=np.float32)
        '''经过上面三行，返回的是image（如416×416）和相对于原图的box(坐标值)，而通过下面这一步获取的是box相对于原图的坐标。
        一个真实框在特征图上的宽高可以用其等比缩放相应下采样倍数后的宽高，相对于特征图的宽高的比例，来表示。
        比如原图416×416，真实框宽高为208×208，那么在下采样32倍的13×13特征图上，该真实框的宽高为208/416×13=6.5。
        同理，真实框的中心坐标用相对于一个grid cell的宽高来表示。
        吴恩达教程里面说的，具体哪一节忘了'''
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2] # 宽高
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2 # 中心
        return image, box

    def rand(self, a=0, b=1): #np.random.rand()返回[0，1）内的一个随机数，故rand函数返回[a，b）内的一个随机数
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''请看2007_train.txt文件'''
        """进行数据增强的随机预处理"""
        """
        数据增强的方式:
          数据增强其实就是让图片变得更加多样,数据增强是非常重要的提高目标检测算法鲁棒性的手段。
          可以通过改变亮度、图像扭曲等方式使得图像变得更加多种多样，改变后的图片放入神经网络进行训练可以提高网络的鲁棒性，降低各方面额外因素对识别的影响.

        param annotation_line: 数据集中的某一行对应的图片
        param input_shape: yolo网络输入图片的大小416 * 416
        param jitter: 控制图片的宽高的扭曲比率, jitter = .3，表示在0.3到1.3之间进行扭曲
        param hue: 代表hsv色域中三个通道中的色调进行扭曲，色调（H）=.1
        param sat: 代表hsv色域中三个通道中的饱和度进行扭曲，饱和度(S) = 1.5
        param val: 代表hsv色域中三个通道中的明度进行扭曲，明度（V）=1.5
        """

        line = annotation_line.split() #分割，返回列表。默认以换行符或者空格分割

        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size #原图尺寸
        h, w    = input_shape #缩放后的尺寸，比如416×416
        #------------------------------#
        #   由于一张图片可能含多个框，那么对该行的图片中的目标框进行一个划分，获得GT
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])


        # 对图像进行缩放并且进行长和宽的扭曲
        # 扭曲后的图片大小可能会大于416*416的大小，但是在加灰条的时候会修正为416*416
        # 调整图片大小
        # 原图片的宽高的扭曲比率，jitter=0,则原图的宽高的比率不变，否则对图片的宽和高进行一定的扭曲

        '''训练的时候，在进行数据增强的时候就已经把图像调整为416×416了，因此不需要进行不失真resize。而测试的时候不进行数据增强所以需要进行不失真resize'''
        if not random: # 是否进行不失真的Resize，分为训练和测试
            scale = min(w/iw, h/ih) # 缩放因子
            nw = int(iw*scale) # 初步调整后的尺寸
            nh = int(ih*scale)
            dx = (w-nw)//2 # 所加灰度条的尺寸
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))#(128,128,128)代表灰色
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   调整ground truth
            #---------------------------------#
            if len(box)>0:#如果该图片里面有目标框的话，还需要对目标框进行相应调整
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box # 返回的是不失真resize后的图片及真实框。
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)# 表原图片的宽高的扭曲比率，jitter=0,则原图的宽高的比率不变，否则对图片的宽和高进行一定的扭曲
        scale = self.rand(.25, 2)
        '''[0.25,2]。
        scale控制对原图片的缩放比率，rand(.25, 2)表示在0.25到2之间缩放，
        图片可能会放大可能会缩小。rand(.25, 1)会把原始的图片进行缩小，图片的边缘加上灰条，
        可以训练网络对我们小目标的检测能力。rand(1,2)则是一定放大图像'''

        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC) #nh,nw为图片扭曲后的高宽

        #------------------------------------------#
        #   将图像多余的部分加上灰条，保证为416×416
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))#(128,128,128)代表灰色
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   是否翻转图像，有50%几率发生翻转
        #------------------------------------------#
        flip = self.rand()<.5 #返回bool值
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT) #左右翻转

        #------------------------------------------#
        #   色域扭曲，发生在hsv这个色域上。hsv色域是有色调H、饱和度S、明度V三者控制，调整这3个值调整色域扭曲的比率。
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        #---------------------------------#
        #   对原图片进行扭曲后，也要对原图片中的框框也进行相应的调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            # 扭曲调整
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy

            # 旋转调整
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box # 返回数据增强后的图片和真实框（由坐标构成），当然只针对训练的时候
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


