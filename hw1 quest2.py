import keras
from keras import layers
from keras.layers import Dropout
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn
import numpy


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


def create_model(number_of_layer, number_of_neurons, dropout_value, l1_l2_regularization):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(dropout_value))
    i = number_of_layer - 2
    while i > 0:
        if l1_l2_regularization == 1:
            model.add(
                layers.Dense(number_of_neurons, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)))
        if l1_l2_regularization == 2:
            model.add(
                layers.Dense(number_of_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
        if l1_l2_regularization == 3:
            model.add(
                layers.Dense(number_of_neurons, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.01)))
        if l1_l2_regularization == 0:
            model.add(
                layers.Dense(number_of_neurons, activation='relu'))
        i -= 1
    model.add(layers.Dense(10, activation='softmax'))
    # model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(train_images, train_labels, epochs=5, batch_size=64)
    return model


number_of_layers_range = [2, 3, 4]
number_of_neurons_range = [10, 25, 50, 100]
dropout_range = [0, 0.1, 0.25, 0.5]
l1_l2_range = [0, 1, 2, 3]
epochs_range = [5, 10, 15, 20, 25]
batch_size_range = [32, 64, 128, 256]

if __name__ == '__main__':
    result_array = []
    for a in number_of_layers_range:
        for b in number_of_neurons_range:
            for c in dropout_range:
                for d in l1_l2_range:
                    for e in epochs_range:
                        for f in batch_size_range:
                            network = create_model(a, b, c, d)
                            network.fit(train_images, train_labels, epochs=e, batch_size=f)
                            test_loss, test_acc = network.evaluate(test_images, test_labels)
                            result_array.append([a, b, c, d, e, f, test_loss, test_acc])
    print(result_array)
    result_array = numpy.array(result_array)
    ranked_result_array = result_array[result_array[:, 7].argsort()]
    print('Accuracy:', ranked_result_array[0][6], '\nLoss:', ranked_result_array[0][7], '\nNumber of layers:',
          ranked_result_array[0][0], '\nNumber of neurons:', ranked_result_array[0][1], '\nNumber of neurons:',
          ranked_result_array[0][2], '\nDropout parameter:', ranked_result_array[0][3], '\nL1/L2 regularization type:',
          ranked_result_array[0][4], '\nEpochs:', ranked_result_array[0][5], '\nBatch size:', ranked_result_array[0][6])

