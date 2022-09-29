import numpy as np
from typing import Callable

# Преобразование значений пикселей от диапазона [0;255] к [0.01;1] для корректной работы функции активации
prepareInputsSignals: Callable[[list[str]], list[float]] = lambda values: (np.asfarray(values) * 0.99 / 255.0) + 0.01
