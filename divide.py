import os
def generate(dir,label):
    path = './' # 生成文件存放的位置
    files = os.listdir(dir)  #，遍历目录文件夹，返回目录下的文件夹和文件
    files.sort()     #对文件列表进行排序
    trainListText = open(path+'train.txt','a')    #创建并打开一个txt文件，a+表示打开一个文件并追加内容
    testListText = open(path + 'test.txt', 'a')
    index = 0
    proportion = 0.8 # 训练集所占比例，剩余的0.2作为验证集
    for file in files:   #遍历文件夹中的文件
        if index < int(len(files)*proportion):
            if(os.path.splitext(file)[1]=='.jpeg' or os.path.splitext(file)[1]=='.jpg'):
                name = dir + '/' + file + '**' + str(int(label))+'\n'
                trainListText.write(name)
                index +=1
        else:
            if(os.path.splitext(file)[1]=='.jpeg' or os.path.splitext(file)[1]=='.jpg'):
                name = dir + '/' + file + '**' + str(int(label)) + '\n'
                testListText.write(name)
                index += 1
    trainListText.close()
    testListText.close()


if __name__ == '__main__':
    dirs = os.listdir('data/train_data/train_data') # 数据集所在位置
    dirs.sort()
    label = 0
    labelText = open('./label.txt', 'a')
    for dir in dirs:
        if dir != '.DS_Store':
            labelText.write(dir + '**' + str(label) + '\n')
            generate('data/train_data/train_data/' + dir, label)
            label += 1
    labelText.close()
    print("down!")
