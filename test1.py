# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.models as models

import PIL.Image as Image
import os

# %%
image_size = (256, 256)
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),

    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# %%
train_data = dset.ImageFolder(root="train", transform=data_transform)
# 数据集长度
totallen = len(train_data)
print('train data length:', totallen)
# %%
train_data
# %%
trainlen = int(totallen * 0.9)
vallen = totallen - trainlen
train_db, val_db = torch.utils.data.random_split(train_data, [trainlen, vallen])
print('train:', len(train_db), 'validation:', len(val_db))
# %%
# batch size
bs = 8
# 训练集
train_loader = torch.utils.data.DataLoader(train_db, batch_size=bs, shuffle=True, num_workers=2)
# 验证集
val_loader = torch.utils.data.DataLoader(val_db, batch_size=bs, shuffle=True, num_workers=2)
# %%
train_data.classes


# %%
def get_num_correct(out, labels):
    return out.argmax(dim=1).eq(labels).sum().item()


# %%
batch = next(iter(train_loader))
# %%
batch[1]
# %%
resnext101 = models.resnet.resnext101_32x8d(pretrained=True)
# %%
model = resnext101
n_classes = len(train_data.classes)
model.fc = nn.Linear(2048, n_classes)
# %%
import torch.nn.init as init

for name, module in model._modules.items():
    if (name == 'fc'):
        # print(module.weight.shape)
        init.kaiming_uniform_(module.weight, a=0, mode='fan_in')
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_num = 10
model = model.to(device)
for epoch in range(epoch_num):
    total_loss = 0
    total_correct = 0
    val_correct = 0
    for batch in train_loader:  # GetBatch
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outs = model(images)  # PassBatch
        loss = F.cross_entropy(outs, labels)  # CalculateLoss
        optimizer.zero_grad()
        loss.backward()  # CalculateGradients
        optimizer.step()  # UpdateWeights
        total_loss += loss.item()
        total_correct += get_num_correct(outs, labels)
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outs = model(images)
        val_correct += get_num_correct(outs, labels)
    print("loss:", total_loss, "train_correct:", total_correct / trainlen, "val_correct:", val_correct / vallen)


# %%
class Laji(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(self.imgs[index]).convert('RGB')
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.imgs)


# %%
test_data = Laji(root="test", transform=data_transform)
# %%
len(test_data)
# %%
model.to('cpu')
pre = []
for image in test:
    output = model(image)
    for i in range(bs):
        print(output[i].shape)
        prob = F.softmax(output[i], dim=0)
        indexs = torch.argsort(-prob)
        print("index:", indexs[0].item(), " prob: ", prob[indexs[0]])
        pre.append(indexs[0].item())
# %%
import pandas as pd
import os
import csv

# 将id读取
path_test = 'test'
# 训练出来垃圾类别
dirs = os.listdir(path_test)
pa = []
count = 0
for img in dirs:
    pa.append(img + '\t' + train_data.classes[pre[count]])
    count = count + 1

# 输出submission文件
pa = pd.DataFrame(pa)
pa.to_csv('submission', quoting=csv.QUOTE_NONE, header=None, index=False)
#

