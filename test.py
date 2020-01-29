'''
在测试集上验证
'''
from config import *
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import model
import func

w2i, i2w = func.get_w2i_dict()

MyMod = model.GetModel_V1((resize_scale, resize_scale, 1), len(w2i), False)
if os.path.exists(weight_path):
    MyMod.load_weights(weight_path)

data_dir = 'HWDB1tst'
file_list = os.listdir(data_dir)
idx = 0

all = 0
acc_num = 0
for name in file_list:
    file_name = os.path.join(data_dir, name)
    with open(file_name, 'rb+') as f:
        while True:
            struct_size = int.from_bytes(f.read(4), byteorder='little')
            if struct_size == 0:
                break
            label = f.read(2).decode('gbk')
            width = int.from_bytes(f.read(2), 'little')
            heigh = int.from_bytes(f.read(2), 'little')
            img = np.frombuffer(f.read(struct_size - 10), dtype=np.uint8)
            img = 255 - np.reshape(img, [heigh, width, -1])
            img_show = np.reshape(img, [heigh, width])

            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_with_pad(img, resize_scale, resize_scale)
            img = tf.image.adjust_contrast(img, 1.4)

            output = MyMod(tf.expand_dims(img, 0))  # 扩充维度到[1(batch), 92, 92, 1]
            pred = i2w[tf.argmax(output, 1).numpy()[0]]
            print('正确文字: ', label, '; 网络预测: ', pred)

            plt.imshow(img_show, cmap='gray')
            plt.show()
