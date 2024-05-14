
from scipy import ndimage
import numpy as np
import cv2

def curvature_fea(img):
    edge_horizont = ndimage.sobel(img, 0)
    edge_vertical = ndimage.sobel(img, 1)
    curvature = np.hypot(edge_horizont, edge_vertical)
    curvature *= 255.0 / np.max(curvature)

    return np.mean(curvature)
