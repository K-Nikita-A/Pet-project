import numpy as np
import matplotlib.pyplot as mp


def showImageNumber(all_values):
    image_array = np.asfarray(all_values).reshape((28, 28))
    mp.imshow(image_array, cmap='Greys', interpolation='None')
    mp.show()