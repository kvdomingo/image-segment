import numpy as np
import cv2 as cv
from numpy import ndarray


def bgr_to_ncc(image: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    i = image.sum(axis=2)
    b, g, r = cv.split(image) / i
    return i, r, g


def pixel_likelihood(r: ndarray, mu: float, sigma: float) -> ndarray:
    return (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
    )
