import numpy as np
from tqdm import tqdm
import dill as pickle
import os

from ML_Algorithms.NeuralNetworks.NeuralNetworkMath import NeuralNetwork
# noinspection PyProtectedMember
from ML_Algorithms.NeuralNetworks.NeuralNetworkMath._model_._methods_ import readData, prepareTrainData, prepareTestData


def main():
    # Количество входных слоев, фиксировано, так как размер изображение 784 пикселя
    input_nodes = 784
    # Количество выходных слоев, фиксировано, так как количество цифр 10
    output_nodes = 10
    hidden_nodes = 200
    learning_rate = 0.15
    epochs = 5
    filename = '../finalized_neural_network_math.sav'

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(os.getcwd())

    print('Train process')
    train_data_list = readData('resources/mnist_train_60000.csv')
    for epoch in tqdm(range(epochs)):
        for item in train_data_list:
            inputs, targets = prepareTrainData(item.split(','), output_nodes)
            network.train(inputs, targets)
    final_score = []

    print('Test process')
    test_data_list = readData('resources/mnist_test_10000.csv')
    for item in test_data_list:
        inputs, correct_value = prepareTestData(item.split(','))
        predict_value = np.argmax(network.predict(inputs))
        final_score.append(int(correct_value == predict_value))

    print('Accuracy: ', sum(final_score) * 100 / len(final_score), '%')

    with open(filename, 'wb') as f:
        pickle.dump(network, f)
    return 0


if __name__ == "__main__":
    main()