import numpy as np

class Normalize:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix / 255.0
        matrix = (matrix - self.mean) / self.std
        return matrix