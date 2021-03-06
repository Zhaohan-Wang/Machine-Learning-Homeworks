import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""数据加载"""
path = '/Users/wangzhaohan/Desktop/ML-Program/mnist.npz'
file = np.load(path)
train_images, train_labels = file['x_train'], file['y_train']
test_images, test_labels = file['x_test'], file['y_test']

"""数据处理"""
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""模型搭建"""


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    # model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(train_images, train_labels, epochs=5, batch_size=64)
    return model


network = create_model()
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
