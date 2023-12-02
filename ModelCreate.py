import tensorflow as tf
 
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, \
     Dense, Reshape, Permute, Dot, Softmax, Lambda, Add, Flatten
from tensorflow.keras.optimizers import Adam

from DataGet import load_data


def create_model(learning_rate):
    inputs=Input(shape=(224, 224, 3), name="img")

    #特征提取模块(基于VGG的特征提取，激活函数使用 ReLU 函数，填补0)
    block_1_C1=Conv2D(filters=32, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(224,224,3), data_format="channels_last")(inputs)
    block_1_C2=Conv2D(filters=32, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(224,224,32), data_format="channels_last")(block_1_C1)
    block_1_M=MaxPooling2D(pool_size=2, strides=2, padding='valid',
                           input_shape=(224,224,32), data_format="channels_last")(block_1_C2)
    block_2_C1=Conv2D(filters=64, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(112,112,32), data_format="channels_last")(block_1_M)
    block_2_C2=Conv2D(filters=64, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(112,112,64), data_format="channels_last")(block_2_C1)
    block_2_M=MaxPooling2D(pool_size=2, strides=2, padding='valid',
                           input_shape=(112,112,64), data_format="channels_last")(block_2_C2)
    block_3_C1=Conv2D(filters=128, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(56,56,64), data_format="channels_last")(block_2_M)
    block_3_C2=Conv2D(filters=128, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(56,56,128), data_format="channels_last")(block_3_C1)
    block_3_C3=Conv2D(filters=128, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(56,56,128), data_format="channels_last")(block_3_C2)
    block_3_M=MaxPooling2D(pool_size=2, strides=2, padding='valid',
                           input_shape=(56,56,128), data_format="channels_last")(block_3_C3)
    block_4_C1=Conv2D(filters=256, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(28,28,128), data_format="channels_last")(block_3_M)
    block_4_C2=Conv2D(filters=256, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(28,28,256), data_format="channels_last")(block_4_C1)
    block_4_C3=Conv2D(filters=256, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(28,28,256), data_format="channels_last")(block_4_C2)
    block_4_M=MaxPooling2D(pool_size=2, strides=2, padding='valid',
                           input_shape=(28,28,256), data_format="channels_last")(block_4_C3)
    block_5_C1=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(14,14,256), data_format="channels_last")(block_4_M)
    block_5_C2=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(14,14,512), data_format="channels_last")(block_5_C1)
    block_5_C3=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(14,14,512), data_format="channels_last")(block_5_C2)
    F=MaxPooling2D(pool_size=2, strides=2, padding='valid',
                   input_shape=(14,14,512), data_format="channels_last")(block_5_C3)
    
    #特征融合模块(基于自注意力的特征融合)
    a=0.001#可学习参数
    Fc=Reshape((49,512),input_shape=(7,7,512))(F)
    Fc_T=Permute((2,1),input_shape=(49,512))(Fc)
    G=Dot(axes=(2,1))([Fc_T, Fc])
    M=Softmax()(G)
    M_T=Permute((2,1),input_shape=(512,512))(M)
    Fc1=Dot(axes=(2,1))([Fc, M_T])
    Fc2=Reshape((7,7,512),input_shape=(49,512))(Fc1)
    Fc3=Lambda(lambda x: x * a)(Fc2)
    Ff=Add()([Fc3, F])
    
    #识别结果输出模块
    output_C1=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(7,7,512), data_format="channels_last")(Ff)
    output_C2=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(7,7,512), data_format="channels_last")(output_C1)
    output_C3=Conv2D(filters=512, kernel_size=3, strides=1, 
                     padding="same", activation = tf.nn.relu,
                     input_shape=(7,7,512), data_format="channels_last")(output_C2)
    Fp=Flatten()(output_C3)
    output_D1=Dense(512, input_shape=(25088,))(Fp)
    output_D2=Dense(64, input_shape=(512,))(output_D1)
    output_D3=Dense(8, input_shape=(64,))(output_D2)
    outputs=Softmax()(output_D3)

    model = Model(inputs, outputs, name="gesture_cnn")
    model.compile(optimizer = Adam(learning_rate=learning_rate),
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['sparse_categorical_accuracy'])
    return model

def decay_schedule(epoch, lr):
    if (epoch % 30 == 0) and (epoch != 0):
        lr = lr * 0.5
    return lr

def model_train(model, train_file, valid_file, epochs, batch_size):
    lr_scheduler = LearningRateScheduler(decay_schedule)

    train_data, train_labels = load_data(train_file)
    valid_data, valid_labels = load_data(valid_file)
    print("训练集和验证集加载成功，开始训练")

    history = model.fit(train_data,
             train_labels,
             epochs = epochs,
             batch_size = batch_size,
             shuffle=True,
             validation_data = (valid_data, valid_labels),
             callbacks=[lr_scheduler])
    return history