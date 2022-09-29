import pickle
from PIL import Image
import numpy as np

from ML_Algorithms.NeuralNetworks.NeuralNetworkMath._model_._methods_ import prepareInputsSignals


def main():
    file_path_to_model = '../NeuralNetworkMath/finalized_neural_network_math.sav'
    file_path_to_image = '../NeuralNetworkMath/resources/7.png'

    loaded_model = pickle.load(open(file_path_to_model, 'rb'))

    img_array = np.asarray(Image.open(file_path_to_image).resize((28, 28)).convert('L'))
    # mnist и rgb используют противоположные кодировки цветов
    img_data = 255.0 - img_array.reshape(784)
    inputs = prepareInputsSignals(img_data)

    predict_value = np.argmax(loaded_model.predict(inputs))
    print(predict_value)
    return 0


if __name__ == "__main__":
    main()
