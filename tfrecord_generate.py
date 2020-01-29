'''
这个文件是把数据统一装到一起，一个叫TFRecord里的东西，这个是 tensorflow 专门用来记录数据的东西
'''
from tqdm import trange
import tensorflow as tf
import numpy as np
import json
import os


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


save_path = './TFRecord/'
save_name = 'HWDB_train_%03d.TFRecord'  # 这里一个 gnt 文件生成一个 TFRecord 文件

if os.path.exists('W2I.Label'):
    with open('W2I.Label', 'r+') as f:
        W2I = json.loads(f.read())
        # 读取 汉字到数字 的映射词典

print(len(W2I))  # 字的数量

if not os.path.exists(save_path):
    os.mkdir(save_path)

data_dir = 'HWDB1trn'  # 要转换的数据集的路径，这里是训练文件夹
file_list = os.listdir(data_dir)
idx = 0

for i in trange(len(file_list)):
    file_name = os.path.join(data_dir, file_list[i])
    with tf.io.TFRecordWriter(save_path + (save_name % i)) as writer:
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
                img_data = tf.image.encode_jpeg(img)  # encoder 转为字节集(bytes)

                exam = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': _int64_feature(W2I[label]),
                            'width': _int64_feature(width),
                            'heigh': _int64_feature(heigh),
                            'image': _bytes_feature(img_data)
                        }
                    )
                )
                writer.write(exam.SerializeToString())
                # 上面几行都是固定样式
