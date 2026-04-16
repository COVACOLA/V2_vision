# homography.py - Transformacion de perspectiva para rectificar el panel

import cv2
import numpy as np
from utils import order_points


def warp_panel(image, panel_points, width, height):
    """
    Aplica transformacion de perspectiva para que el panel quede
    como un rectangulo frontal del tamaño indicado.
    Devuelve la imagen rectificada y la matriz H.
    """
    # OpenCV necesita un orden estable de puntos para que la
    # homografia no invierta ni deforme el panel.
    src = order_points(panel_points)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, H, (width, height))

    return warped, H


def point_to_panel(point, H):
    """Transforma un punto de la imagen original a coordenadas del panel rectificado."""
    pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def point_from_panel(point, H):
    """Transforma un punto del panel rectificado de vuelta a la imagen original."""
    H_inv = np.linalg.inv(H)
    return point_to_panel(point, H_inv)
