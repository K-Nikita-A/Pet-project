from ML_Algorithms.NeuralNetworks.NeuralNetworkMath._model_._methods_ import prepareInputsSignals


def prepareTestData(all_values: list[str]) -> tuple[list[float], int]:
    correct_value = int(all_values[0])
    inputs = prepareInputsSignals(all_values[1:])
    return inputs, correct_value
