import cv2.dnn
import cv2
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch
import os
import numpy as np
from PIL import Image

def getLabel(line_num):
    f = open("dir_label.txt",encoding='utf-8')
    if line_num == 1:
        line = f.readline()
    else:
        line = f.readline()
        while line:
            line = f.readline()
            line_num -= 1
            if (line_num == 1):
                break
    return line
def padding_black( img):

    w, h  = img.size

    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

    size_fg = img_fg.size
    size_bg = 224

    img_bg = Image.new("RGB", (size_bg, size_bg))

    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))

    img = img_bg
    return img
def load():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()])
    num_classes = 214  # 加载数据
    map_location = "cpu"
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, num_classes)
    resume = 'model_best_checkpoint_resnet50.pth.tar'
    checkpoint = torch.load(resume, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return t,model
def getRubbish(img,t,model):
    image = t(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(image)
        _, pred = torch.max(pred, 1)
        pred = getLabel(pred.item()+1)
    return pred
def shibie(pathh,t,model):
    img = Image.open(pathh)
    img = img.convert('RGB')
    img = padding_black(img)
    resize1 = transforms.Resize(224)
    img=resize1(img)
    totnseor= transforms.ToTensor()
    img=totnseor(img)
    pre = getRubbish(img,t,model)
    return pre

if __name__ == '__main__':
    # img = cv2.imdecode(np.fromfile("D:/lesson-4/垃圾图片库/可回收物_木制玩具/img_木制玩具_163.jpeg", dtype=np.uint8), -1)
    # img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    # # img = cv2.imread('img1.jpeg')
    # print(img)
    # cv2.imshow('11',img)
    # # cv2.waitKey(0)
    # print(getRubbish(img))
    t,model = load()

    print(shibie("D:/lesson-4/垃圾图片库/可回收物_木制玩具/img_木制玩具_163.jpeg",t,model))

    # f = open("test.txt",encoding='utf-8')
    # line = f.readline()
    # rcnt=0
    # cnt=0
    # while line:
    #     # print(line.split())
    #     pathh = line.split()[0]
    #     if pathh:
    #         img = cv2.imdecode(np.fromfile(pathh, dtype=np.uint8), -1)
    #         img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    #         pre = getRubbish(img)
    #         print(pre)
    #         rr = line.split()[1]
    #         print(rr,pre)
    #         if rr == pre.split()[1]:
    #             rcnt+=1
    #         cnt+=1
    #     line = f.readline()
    #     print('准确率',rcnt/cnt)

