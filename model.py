'''
这个文件定义了网络的结构，可以随意更改测试不同模型的效果
'''
import tensorflow.keras as keras

def GetModel_V1(input_shape, class_num, is_training=True):
    input_ = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(64, 7, 2, 'SAME', activation='relu')(input_)
    pool1 = keras.layers.MaxPooling2D()(conv1)
    conv2 = keras.layers.Conv2D(256, 3, 2, 'SAME', activation='relu')(pool1)
    pool2 = keras.layers.MaxPooling2D()(conv2)
    conv3 = keras.layers.Conv2D(512, 3, 2, 'SAME', activation='relu')(pool2)
    pool3 = keras.layers.AveragePooling2D()(conv3)

    flat = keras.layers.Flatten()(pool3)
    # drop1 = keras.layers.Dropout(0.25)(flat, is_training)
    output = keras.layers.Dense(class_num, activation='softmax')(flat)

    model = keras.Model(inputs=input_, outputs=output)
    # keras.utils.plot_model(model, 'Model_V1.png')  # 将模型结构输出为文件，需要安装一个软件
    model.summary()  # 将模型信息输出到终端
    return model

