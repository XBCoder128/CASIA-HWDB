from config import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import func
import os


# 下面这个字典和存储时的定义顺序要一致
exam_feature = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'heigh': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
}


def trans_func(exam):  # 对每个数据做映射
    features = tf.io.parse_single_example(exam, exam_feature)  # 读入TFRecord要先用这个函数解析
    img = tf.image.decode_image(features['image'])  # 直接解码图片
    img = tf.image.convert_image_dtype(img, tf.float32)  # 归一化
    img = tf.image.resize_with_pad(img, resize_scale, resize_scale)  # 重定义尺寸
    img = tf.image.adjust_contrast(img, 1.25 + tf.abs(tf.random.normal((), 0.3)))  # 随机对比度
    return img, features['label']  # 返回图片和标签


def GetDataset(DatasetPath):  # 根据TFRecord路径生成数据集
    TFRecodeFiles = [os.path.join(DatasetPath, file) for file in os.listdir(DatasetPath)]
    # 取出所有的TFRecord文件

    dataset = tf.data.TFRecordDataset(TFRecodeFiles, num_parallel_reads=5)  # 同时读入5个文件，顺序随机
    dataset = dataset.map(trans_func)  # 映射一下
    dataset = dataset.shuffle(512)  # 随机打乱，打乱前512个
    dataset = dataset.batch(batch_size)  # 设置batch大小
    return dataset
