from dataset import Garbage_Loader
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
    Author : kangzhicheng yangshuoqiu
"""

from tensorboardX import SummaryWriter   #Google的tensorflow中的tensorboard是一个用于服务神经网络训练过程可视化的网络服务器（web server），其可以实现标量值、图像、文本等的可视化。

#TopK是一种计算准确率的方法，在多分类任务中经常出现，如 cifar100 这个100分类任务中，衡量预测是否准确可以：
#1、直接对比网络计算得到的output中概率最高的那个类别和标签类别是否一致，如果一致则判断正确，否则错误。这就是常见的计算准确率的方法，也叫 Top1 准确率。
#2、对比output中概率前五的类别中是否有标签的类别，有则判断正确，否则就是判断错误。称为 Top5 准确率。
#k越大对网络的评价标准越宽
def accuracy(output, target, topk=(1,)):  #output：模型的输出，即模型对不同类别的评分。shape: [batch_size, num_classes]
    """                                   #target真实类别标签,shape[batch_size]
        计算topk的准确率                    #topk：需要计算top_k准确率中的k值，元组类型。默认为(1, 5)，即函数返回top1和top5的分类准确率
    """
    with torch.no_grad():   #减少冗长，自动处理上下文环境产生的异常
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)

def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # 计算输出
        output = model(input)
        loss = criterion(output, target)   #模型输出和目标，然后计算一个值来评估输出距离目标有多远。

        # 测量准确性、记录损失
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # 计算梯度并执行和SGD步骤
        optimizer.zero_grad()   #梯度清0
        loss.backward()         #将损失函数backward进行方向梯度传播
        optimizer.step()        #使用优化器的step函数更新参数

        # 测量流逝的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)

def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()   #评估模式

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  #每隔10个保存一个模型
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              phase, i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=top1, top5=top5))

        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return top1.avg, top5.avg

class AverageMeter(object):
    """计算并储存平均值和当前值"""
    def __init__(self):
        self.reset()    #重置

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = 'train.txt'
    valid_dir_list = 'val.txt'
    batch_size = 32  #每批训练数据量的大小
    epochs = 20    #一个epoch指用训练集中的全部样本训练一次
    num_classes = 214   #类别
    train_data = Garbage_Loader(train_dir_list, train_flag=True)
    valid_data = Garbage_Loader(valid_dir_list, train_flag=False)
    train_loader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    model = models.resnet50(pretrained=True)    #加载resnet101模型
    fc_inputs = model.fc.in_features     #提取fc层中的固定参数
    model.fc = nn.Linear(fc_inputs, num_classes)    #修改类别为num_classes=214
    model = model.cuda()  #GUP,CUDA训练
    # ------------------------------------ step 3/4 : 定义损失函数和优化器adam等 -------------------------
    lr_init = 0.0001
    lr_stepsize = 20
    weight_decay = 0.001
    criterion = nn.CrossEntropyLoss().cuda()   #定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)   #learningrate=0.0001,权重衰减速率为0.001
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)  #阶梯调整学习速率，当达到第20个时候调整一个学习速率，*0.1
    
    writer = SummaryWriter('runs/resnet101')
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec1 = 0
    for epoch in range(epochs):
        scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet101',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best,
            filename='checkpoint_resnet101.pth.tar')
    writer.close()
