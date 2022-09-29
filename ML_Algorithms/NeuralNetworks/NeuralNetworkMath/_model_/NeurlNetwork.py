import numpy as np
from scipy.special import expit


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        # количество нейронов на каждом из слоёв
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # коэффициент обучения
        self.lr = learning_rate
        # матрицы весов
        self.w_ih = np.random.normal(loc=0.0,
                                     scale=pow(self.h_nodes, -0.5),
                                     size=(self.h_nodes, self.i_nodes)
                                     )
        self.w_ho = np.random.normal(loc=0.0,
                                     scale=pow(self.o_nodes, -0.5),
                                     size=(self.o_nodes, self.h_nodes)
                                     )
        # функция активации
        self.activation_func = lambda x: expit(x)

    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.w_ho.T, output_errors)

        self.w_ho += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                      np.transpose(hidden_outputs))

        self.w_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

    def predict(self, inputs_list):
        hidden_inputs = np.dot(self.w_ih, inputs_list)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs
