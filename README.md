# CASIA-HWDB
针对离线中文手写数据集的学习

# 使用库
python == 3.6

tensorflow == 2.0

matplotlib

numpy

# 数据集
数据集下载地址：http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

要下载的文件：
 - 1. HWDB1.1trn_gnt（1873MB）
 - 2. HWDB1.1tst_gnt（471MB）
 
# 模块介绍
### 路径 ‘HWDB1trn’ 与 ‘HWDB1tst’
保存训练和测试数据的文件夹，存放的是gnt文件

### config.py
一些超参数的定义

### data_loader.py
简单演示训练数据(\*.gnt文件)的读入，与训练和测试无关

### Word2Index.py
生成一个将文字转成数字的字典并保存成文件

### func.py
里面放了读取文字到ID的字典的函数，且做了逆变换

### model.py
定义模型，可换成不同的想要测试的模型看看效果

### train.py / test.py
训练/测试文件

### tfrecord_generate.py
将gnt文件转为TFrecord文件，方便tf.data处理

### tfrecord_loader.py
加载tfrecord

# 如何训练
首先将训练文件放入之前提到的文件夹，然后运行'python tfrecord_generate.py'

等待tfrecord生成完毕后，运行'python train.py'，如果有保存权重，每次训练前会优先读取权重，若要重新训练网络请先删除权重文件，默认为‘model.h5’

# 关于测试
测试准确率我这里大概是60%左右，但是文件我把计算准确率的部分改了，因为图片太多了算一次老长时间。test.py主要是循环读取并显示一张图片，并打印正确值和网络预测值。
