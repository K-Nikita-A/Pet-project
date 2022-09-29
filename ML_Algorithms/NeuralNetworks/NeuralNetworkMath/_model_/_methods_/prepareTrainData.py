import numpy as np

from ML_Algorithms.NeuralNetworks.NeuralNetworkMath._model_._methods_ import prepareInputsSignals


def prepareTrainData(all_values: list[str], output_nodes: int) -> tuple[list[float], list[float]]:
    inputs = prepareInputsSignals(all_values[1:])
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    return inputs, targets
