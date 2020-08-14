import numpy as np
import matplotlib


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    if isinstance(x, list):
        new_output = []
        for i in x:
            new_output.append(abs(i))
        return new_output
    else:
        return abs(x)


class network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_n = input_nodes
        self.h_n = hidden_nodes
        self.o_n = output_nodes
        self.l_r = learning_rate
        self.w_i_h = np.random.normal(0.0, pow(self.h_n, -0.5), (self.h_n, self.i_n))
        self.w_h_o = np.random.normal(0.0, pow(self.o_n, -0.5), (self.o_n, self.h_n))
        self.act_func = lambda x: sigmoid(x)
        pass

    def train(self, input_list, target_list):
        input_array, target_array = np.array(input_list, ndmin=2).T, np.array(target_list, ndmin=2).T
        hidden_input = np.dot(self.w_i_h, input_array)
        hidden_output = self.act_func(hidden_input)
        final_input = np.dot(self.w_h_o, hidden_output)
        final_output = self.act_func(final_input)
        output_err = target_array - final_output
        hidden_err = np.dot(self.w_h_o.T, output_err * final_output * (1.0 - final_output))
        self.w_h_o += self.l_r * np.dot((output_err * final_output * (1.0 - final_output)), np.transpose(hidden_output))
        self.w_i_h += self.l_r * np.dot((hidden_err * hidden_output * (1.0 - hidden_output)), np.transpose(input_array))
        pass

    def query(self, input_list):
        input_array = np.array(input_list, ndmin=2).T
        hidden_input = np.dot(self.w_i_h, input_array)
        hidden_output = self.act_func(hidden_input)
        final_input = np.dot(self.w_h_o, hidden_output)
        final_output = self.act_func(final_input)
        return final_output


if __name__ == '__main__':
    input_nodes = 28*28
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    epochs = 5
    score_borad = []
    network_test = network(input_nodes, hidden_nodes, output_nodes, learning_rate)

    path = '/Users/wangzhaohan/Desktop/ML-Program/mnist.npz'
    file = np.load(path)
    train_images, train_labels = file['x_train'], file['y_train']
    test_images, test_labels = file['x_test'], file['y_test']
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    for i in range(epochs):
        for j in range(len(train_labels)):
            use_test_labels = np.zeros(output_nodes) + 0.01
            use_test_labels[int(train_labels[j])] = 0.99
            network_test.train(train_images[j], use_test_labels)
        print('Epoch: ', i+1, '/', epochs)

    for i in range(len(test_labels)):
        correct_lable = test_labels[i]
        predict_lable = np.argmax(network_test.query(test_images[i]))
        if predict_lable == correct_lable:
            score_borad.append(1)
        else:
            score_borad.append(0)
    score_borad = np.asarray(score_borad)
    print("Accuracy: ", score_borad.sum()/score_borad.size)