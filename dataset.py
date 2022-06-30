import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Garbage_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        
        self.train_tf = transforms.Compose([
                transforms.Resize(224),     #将图片压缩成224*224的大小
                transforms.RandomHorizontalFlip(),   #对图片进行随机的水平翻转
                transforms.RandomVerticalFlip(),     #随机的垂直翻转
                transforms.ToTensor(),               #图片改为Tensor格式，图像数据的像素归一化，像素值为0-255，除以255，把像素值归一化到0-1之间

            ])
        self.val_tf = transforms.Compose([     #把图片压缩编程tensor模式
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        
    def get_images(self, txt_path):       #加载得到的图片信息
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()    #返回文件中所有行
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))  #imgs_info中的每个元素作为参数x传递给lambda x:x.strip()，然后在其上调用strip()方法
        return imgs_info
     
    def padding_black(self, img):    #填充黑色操作，如果尺寸太小可以扩充，填充黑色使得图片224*224

        w, h  = img.size

        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg))    #将img_bg变为224*224大小的图片

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,    #将img_fg填充到尺寸不足的部分
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img
        
    def __getitem__(self, index):   #获取图片和标签
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')     #读出RGB四通道的图片
        img = self.padding_black(img)   #填充操作
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label
 
    def __len__(self):    #获取数据集的规模大小
        return len(self.imgs_info)
 
    
if __name__ == "__main__":
    train_dataset = Garbage_Loader("train.txt", True)    # True 为将 "train.txt" 文件数据看做训练集对待
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)