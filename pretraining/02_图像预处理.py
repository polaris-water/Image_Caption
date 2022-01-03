# -*- coding: UTF-8 -*-  
# @Time : 2021/12/10 21:13
# @Author : GCR
# @File : 03_图像预处理.py
# @Software : PyCharm



'''
数据：获取3万张图片
得到：30000*2048   3 万张图片，每张图片2048维
'''
from torchvision.models import resnet50


'''
1、我们要用神经网络提取图像的特征，具体来说是利用 ResNet 提取图片在倒数第二层的2018维度的向量。
2、为了提取这一层的输入，我们的措施：
    第一种：修改torchvison中的ResNet源码，让他在倒数第二层就输出返回
    第二种：直接把最后一层删除替换成一个恒等映射
'''
# def new_forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     # x = self.fc(x)  # 第一种做法
#     return x
#
#
# #模型
# model = resnet50(pretrained=True)
# #用新的 forward函数覆盖旧的forward函数
# model.forward = lambda x:new_forward(model, x)
# model.cuda()

#利用残差网络来提取图像的特征。 每张图片的特征用 2048 维的向量来表示
model=resnet50(pretrained=True)
del model.fc  # 直接把最后一层删除
model.fc=lambda x:x # 替换成一个恒等映射
model.cuda()



import torchvision as tv  # 一般的图像转换操作类
from PIL import Image  # pillow库，PIL读取图片
import numpy as np
import torch
from torch.utils import data
import os
from torch.autograd import Variable

# torch.cuda.set_device(0)


# 使用 ImageNet的均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#具体来说，对每个通道而言，Normalize执行：iamge=（image-mean）/std
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

transforms = tv.transforms.Compose([
    tv.transforms.Resize(256), # 将图像较短的边放缩到256，长宽比保持不变
    tv.transforms.CenterCrop(256), # 中心切割。256x256大小的图片
    tv.transforms.ToTensor(), # 图片像素点单个通道为[0,255]--->[0,1]
    normalize  #[0,1] --> [-1,1],对每个通道执行：value_n = (value-mean)/std
])


#重新定义数据类
class Dataset(data.Dataset):
    def __init__(self, caption_data_path): # caption_data_path：图片存放路径
        data = torch.load(
            'caption_2.pth')
        self.ix2id = data['ix2id'] # 每个序号对应的图片的iamge_id

        # 返回一个列表，列表中每个元素表示一张图片的路径
        self.imgs = [os.path.join(caption_data_path, self.ix2id[_]) for _ in range(len(self.ix2id))]
        # print(self.imgs)

    def __getitem__(self, item): # 重写这个方法
        x = Image.open(self.imgs[item]).convert('RGB') # 打开一个图像，并进行通道转换（RGBA-->RGB）
        x = transforms(x)  # ([3, 256, 256])
        return x, item

    def __len__(self): #  重写这个方法
        return len(self.imgs)  # 返回数据的长度


batch_size = 16  # 每次处理 16 张图片
dataset = Dataset(
    r'F:\09_周任务\2021.12.5\数据集\ai_challenger_caption_validation_20170910\caption_validation_images_20170910')
'''
dataset是一个元组组成的集合，一个元组表示一张图片。
元组的第一个元素：图片的像素表示
元组的第二个元素：图片的序号
'''

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 16张图片为一组
# print(dataloader)

results = torch.Tensor(len(dataloader.dataset), 2048).fill_(0) # 30000x2048 的张量，存储每个图片的特征


for ii, (imgs, indexs) in  enumerate(dataloader):
    print(ii,(imgs,indexs)) # 一次处理16张图片
    assert indexs[0] == batch_size * ii
    imgs = imgs.cuda()
    imgs=Variable(imgs,volatile=True)
    features = model(imgs)
    results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()
    print(ii * batch_size)

torch.save(results, 'results_2048.pth')

