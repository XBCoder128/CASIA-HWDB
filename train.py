from config import *
import tensorflow as tf
import tfrecord_loader
import model
import func
import numpy as np
import os

isTraining = True
train_dataset_path = 'TFRecord'
# test_dataset_path  = 'HWDB1tst'

train_dataset = tfrecord_loader.GetDataset(train_dataset_path)
w2i, i2w = func.get_w2i_dict()

MyMod = model.GetModel_V1((resize_scale, resize_scale, 1), len(w2i), isTraining)  # 获取模型
optimizer = tf.keras.optimizers.RMSprop()  # 定义优化器

if os.path.exists(weight_path):
    MyMod.load_weights(weight_path)  # 加载权重


@tf.function()
def train(datas, labels):  # 训练函数
    with tf.GradientTape() as Tape:  # 自动记录梯度
        output = MyMod(datas)  # 前向传播
        losses = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, output))  # 计算 loss
    grad = Tape.gradient(losses, MyMod.trainable_variables)  # 获取梯度
    optimizer.apply_gradients(zip(grad, MyMod.trainable_variables))  # 传播梯度
    return losses  # 返回 loss 值

if isTraining:
    try:
        for eps in range(epochs):
            for i, (datas, labels) in enumerate(train_dataset):
                if i % 1000 == 0:  # 1000 次 保存一下权重
                    MyMod.save_weights(weight_path)
                loss = train(datas, labels)
                if i % 20 == 0:  # 20次输出一下loss和这一轮的准确率
                    y_pred = MyMod(datas)
                    acc = tf.reduce_sum(tf.cast(tf.equal(labels, tf.argmax(y_pred, 1)), tf.float64)) / len(labels)
                    print('epoch: %d, step: %d, loss:%.2f, acc = %.2f%%' % (eps, i, loss.numpy(), acc.numpy() * 100))
    except KeyboardInterrupt:
        MyMod.save_weights(weight_path)  # Ctrl + C 时保存模型