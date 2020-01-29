'''
这个文件演示如何读取原始数据
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dir_name = 'HWDB1trn'  # 训练文件夹
file_list = os.listdir(dir_name)
file_name = os.path.join(dir_name, file_list[0])  # 取第一个文件测试

resize_scale = 92

print('File: ', file_name)

with open(file_name, 'rb+') as f:
    while True:
        struct_size = int.from_bytes(f.read(4), byteorder='little')
        # 一个样本(包括该字段+label+宽+高+data字段)的字节数
        if struct_size == 0:  # 样本尺寸为0时为文件尾
            break
        print("Struct size:", struct_size, '\n')
        label = f.read(2).decode('gbk')  # 这个样本是什么字

        print('label:', label)

        width = int.from_bytes(f.read(2), 'little')
        heigh = int.from_bytes(f.read(2), 'little')
        print('Pic size: %d x %d' % (width, heigh))
        # 获取宽高

        img = np.frombuffer(f.read(struct_size - 10), dtype=np.uint8)
        img = 255 - np.reshape(img, [heigh, width, -1])  # 取反色，为后面缩放填充做准备
        img1 = 255 - np.reshape(img, [heigh, width])
        plt.imshow(img1, cmap='gray')
        plt.show()

        img = tf.image.convert_image_dtype(img, tf.float32)  # 将图片value转为0 ~ 1之间
        img = tf.image.resize_with_pad(img, resize_scale, resize_scale)  # 先按比例缩放，如果尺寸再不对就在周围填充0
        # 因为给定的训练数据尺寸都是不相同的，所以要缩放为统一尺寸，resize_scale 定义在 config.py
        img_ = tf.image.adjust_contrast(img, 2)  # 调对比度，不然原图文字太灰了

        plt.imshow(np.reshape(img_.numpy(), (resize_scale, resize_scale)), cmap='gray')
        plt.show()
