import json
import os

import cv2
import numpy as np


def order_points(pts):
    """Ordena 4 puntos como: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")

    # El de menor suma x+y es el top-left, el de mayor es bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # La diferencia y-x minima es top-right, la maxima es bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def bbox_from_points(pts):
    """Calcula bounding box [x, y, w, h] a partir de un conjunto de puntos."""
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())
    x_max = int(pts[:, 0].max())
    y_max = int(pts[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def center_from_bbox(bbox):
    """Devuelve el centro [x, y] de una bbox [x, y, w, h]."""
    x, y, w, h = bbox
    return [int(x + w / 2), int(y + h / 2)]


def build_output_dict(panel_bbox, panel_points, detection_method, warped_size):
    """Construye la salida minima que consumiran otros modulos."""
    return {
        "panel": {
            "bbox": panel_bbox,
            "center": center_from_bbox(panel_bbox),
            "corners": panel_points.astype(int).tolist(),
            "detection_method": detection_method,
            "normalized_size": [int(warped_size[0]), int(warped_size[1])],
        }
    }


def save_json(data, path):
    """Guarda un dict como JSON con formato legible."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def draw_panel_bbox(image, panel_points, color=(0, 255, 0), thickness=2):
    """Dibuja el cuadrilatero del panel sobre una copia de la imagen."""
    out = image.copy()
    pts = panel_points.astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
