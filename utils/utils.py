import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image): #定义转RGB图像
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
#---------------------------------------------------#
#   对输入图像进行不失真的resize
#   核心思想是将图片中最长的边等比例缩放到目标尺寸，然后再对短边加灰条
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size #未调整前的图像尺寸
    w, h    = size #目标尺寸
    if letterbox_image: #用于判断是否进行不失真的resize，否则可以直接resize到目标尺寸
        scale   = min(w/iw, h/ih) # 先把最长的边压缩为目标尺寸然后求出缩放比
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC) # 将图片进行尺寸调整
        new_image = Image.new('RGB', size, (128,128,128)) # 空白部分补灰条
        new_image.paste(image, ((w-nw)//2, (h-nh)//2)) # 将image覆盖到new_image的坐标为((w-nw)//2, (h-nh)//2)的位置。
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()#读取txt文件一行一行的数据,返回一个包含所有行的列表
    class_names = [c.strip() for c in class_names]
    '''Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
    https://www.runoob.com/python/att-string-strip.html'''
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()#一行一行的读，返回一个列表
    anchors = [float(x) for x in anchors.split(',')]#将读取到的列表中的元素以逗号为分割符进行分割，返回的值再强转成flaot类型，作为列表的元素
    #[10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
    anchors = np.array(anchors).reshape(-1, 2) #anchors是个列表，用np.array创建一个数组。再改变形状：2代表变成两列，-1代表行数自动调整
    '''[[ 10.  13.]
        [ 16.  30.]
        [ 33.  23.]
        [ 30.  61.]
        [ 62.  45.]
        [ 59. 119.]
        [116.  90.]
        [156. 198.]
        [373. 326.]]'''
    return anchors, len(anchors) #len(anchors)为9

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):#传入一个优化器，获取其学习率
    for param_group in optimizer.param_groups:
        return param_group['lr']
    '''甚至可以通过这种方法动态修改优化器的学习率，比如
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr'''

def preprocess_input(image):
    image /= 255.0
    return image

