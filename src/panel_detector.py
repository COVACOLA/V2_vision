import cv2
import numpy as np

from config import (
    ANCHOR_MIN_AREA,
    ANCHOR_MIN_ASPECT,
    ANCHOR_MIN_FILL,
    ANCHOR_TAG_LAYOUT,
    ANCHOR_THRESHOLD,
    CANNY_HIGH,
    CANNY_LOW,
    MAX_ASPECT_ERROR,
    MAX_BORDER_TOUCHES,
    MAX_PANEL_AREA_RATIO,
    MIN_PANEL_AREA_RATIO,
    PANEL_ASPECT_RATIO,
    PANEL_HSV_LOWER,
    PANEL_HSV_UPPER,
)
from utils import order_points


def _build_gray_mask(image):
    # El panel suele ocupar una region gris y poco saturada.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(PANEL_HSV_LOWER, dtype=np.uint8),
        np.array(PANEL_HSV_UPPER, dtype=np.uint8),
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return mask


def _build_edge_mask(image):
    # Este camino refuerza contornos cuando el color no es suficiente.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, None, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges


def _build_detail_mask(image):
    # No se usa para recortar el panel, sino para puntuar si
    # una region tiene suficiente estructura interna.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def _detect_anchor_tags(image):
    # Las anclas negras son la pista mas fuerte cuando aparecen:
    # permiten ajustar el panel por geometria en lugar de por area.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(
        gray,
        ANCHOR_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < ANCHOR_MIN_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if width < 5 or height < 5:
            continue

        aspect = min(width, height) / max(width, height)
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / max(w * h, 1)
        if aspect < ANCHOR_MIN_ASPECT or fill < ANCHOR_MIN_FILL:
            continue

        box = order_points(cv2.boxPoints(rect).astype(np.float32))
        center = box.mean(axis=0)
        candidates.append({
            "center": center,
            "box": box,
        })

    if len(candidates) < 3:
        return None, mask

    # Se espera un patron concreto: dos anclas arriba y una abajo
    # a la izquierda. Con eso ya se puede estimar el plano completo.
    candidates.sort(key=lambda tag: float(tag["center"][1]))
    top_tags = sorted(candidates[:2], key=lambda tag: float(tag["center"][0]))
    lower_tags = sorted(candidates[2:], key=lambda tag: float(tag["center"][0]))

    if len(top_tags) != 2 or len(lower_tags) < 1:
        return None, mask

    anchors = {
        "top_left": top_tags[0],
        "top_right": top_tags[1],
        "bottom_left": lower_tags[0],
    }
    return anchors, mask


def _panel_from_anchor_tags(image, anchors):
    src_points = []
    dst_points = []
    for key in ("top_left", "top_right", "bottom_left"):
        x, y, w, h = ANCHOR_TAG_LAYOUT[key]
        src_points.extend([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ])
        dst_points.extend(anchors[key]["box"].tolist())

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src_points, dst_points, 0)
    if homography is None:
        return None

    # Proyectamos el rectangulo normalizado del panel completo a
    # la imagen real usando la homografia obtenida de las anclas.
    panel_template = np.array(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(panel_template, homography)[0]

    height, width = image.shape[:2]
    projected[:, 0] = np.clip(projected[:, 0], 0, width - 1)
    projected[:, 1] = np.clip(projected[:, 1], 0, height - 1)
    return order_points(projected.astype(np.float32))


def _candidate_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return rect, box


def _count_border_touches(box, image_shape):
    height, width = image_shape[:2]
    margin = int(min(width, height) * 0.02)
    x_min = box[:, 0].min()
    y_min = box[:, 1].min()
    x_max = box[:, 0].max()
    y_max = box[:, 1].max()

    return sum([
        x_min <= margin,
        y_min <= margin,
        x_max >= width - 1 - margin,
        y_max >= height - 1 - margin,
    ])


def _detail_fraction(box, detail_mask):
    polygon = np.zeros(detail_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(polygon, box.astype(np.int32), 255)
    inside = polygon > 0
    if not np.any(inside):
        return 0.0
    return float((detail_mask[inside] > 0).mean())


def _intensity_std(box, gray_image):
    polygon = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(polygon, box.astype(np.int32), 255)
    pixels = gray_image[polygon > 0]
    if pixels.size == 0:
        return 0.0
    return float(np.std(pixels))


def _score_candidate(box, contour_area, image_shape, detail_mask, gray_image):
    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    if width < 2 or height < 2:
        return -1.0

    if width > height:
        width, height = height, width

    aspect = width / height
    aspect_error = abs(aspect - PANEL_ASPECT_RATIO)
    if aspect_error > MAX_ASPECT_ERROR:
        return -1.0

    image_area = float(image_shape[0] * image_shape[1])
    rect_area = max(width * height, 1.0)
    area_ratio = rect_area / image_area
    if area_ratio < MIN_PANEL_AREA_RATIO or area_ratio > MAX_PANEL_AREA_RATIO:
        return -1.0

    fill_ratio = contour_area / rect_area
    border_touches = _count_border_touches(box, image_shape)
    if border_touches > MAX_BORDER_TOUCHES:
        return -1.0

    # Un buen candidato no solo tiene la geometria correcta; tambien
    # debe contener detalle y variacion tonal propios del panel.
    detail_score = min(_detail_fraction(box, detail_mask) / 0.10, 1.0)
    variation_score = min(_intensity_std(box, gray_image) / 60.0, 1.0)
    center = box.mean(axis=0)
    cx = center[0] / image_shape[1]
    cy = center[1] / image_shape[0]
    center_score = 1.0 - min(abs(cx - 0.5) + abs(cy - 0.5), 1.0)

    aspect_score = 1.0 - (aspect_error / MAX_ASPECT_ERROR)
    area_score = min(area_ratio / 0.25, 1.0)
    fill_score = min(fill_ratio, 1.0)
    border_penalty = border_touches * 0.35

    return (
        4.0 * aspect_score
        + 2.5 * area_score
        + 2.0 * fill_score
        + 2.5 * detail_score
        + 2.5 * variation_score
        + 1.5 * center_score
        - border_penalty
    )


def _find_best_candidate(mask, image_shape, detail_mask, gray_image):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1.0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 1000:
            continue

        _, box = _candidate_box(contour)
        score = _score_candidate(
            box,
            contour_area,
            image_shape,
            detail_mask,
            gray_image,
        )
        if score > best_score:
            best_score = score
            best_box = box

    return best_box, best_score


def detect_panel(image):
    """
    Detecta el panel completo y devuelve sus cuatro esquinas.

    Estrategia:
    1. Mascara HSV para superficies grises/poco saturadas.
    2. Mascara por bordes para escenas donde el contorno es mas estable que el color.
    3. Seleccion del mejor cuadrilatero segun area, aspecto y posicion.
    """
    masks = {
        "gray_mask": _build_gray_mask(image),
        "edge_mask": _build_edge_mask(image),
    }
    detail_mask = _build_detail_mask(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    anchor_tags, anchor_mask = _detect_anchor_tags(image)

    # Si las anclas aparecen, confiamos antes en esa geometria porque
    # suele ser mas precisa que el rectangulo global por contornos.
    if anchor_tags is not None:
        anchor_box = _panel_from_anchor_tags(image, anchor_tags)
        if anchor_box is not None:
            debug_data = {
                "gray_mask": masks["gray_mask"],
                "edge_mask": masks["edge_mask"],
                "detail_mask": detail_mask,
                "anchor_mask": anchor_mask,
                "anchors": {
                    key: value["box"].astype(int).tolist()
                    for key, value in anchor_tags.items()
                },
                "score": None,
            }
            return anchor_box, "anchor_tags", debug_data

    best_method = None
    best_box = None
    best_score = -1.0

    # Fallback: probamos ambas mascaras y nos quedamos con el
    # candidato de mejor puntuacion.
    for method, mask in masks.items():
        box, score = _find_best_candidate(
            mask,
            image.shape,
            detail_mask,
            gray_image,
        )
        if box is not None and score > best_score:
            best_method = method
            best_box = box
            best_score = score

    if best_box is None:
        raise RuntimeError("No se pudo detectar el panel con el flujo basico.")

    debug_data = {
        "gray_mask": masks["gray_mask"],
        "edge_mask": masks["edge_mask"],
        "detail_mask": detail_mask,
        "anchor_mask": anchor_mask,
        "anchors": None,
        "score": best_score,
    }
    return best_box, best_method, debug_data
