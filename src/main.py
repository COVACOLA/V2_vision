import os
import sys
import json

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WARP_WIDTH, WARP_HEIGHT, OUTPUT_DIR, DEBUG
from panel_detector import detect_panel
from homography import warp_panel
from utils import bbox_from_points, build_output_dict, save_json, draw_panel_bbox


def run_pipeline(image_path):
    """Ejecuta el flujo minimo: detectar panel, obtener bbox y normalizar."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    print(f"[1/4] Imagen cargada: {image_path} ({image.shape[1]}x{image.shape[0]})")

    panel_points, method, debug_data = detect_panel(image)
    print(f"[2/4] Panel detectado con '{method}'")

    # La bbox se calcula sobre la imagen original; la homografia
    # genera aparte una vista rectificada para la siguiente fase.
    panel_bbox = bbox_from_points(panel_points)

    warped, _ = warp_panel(image, panel_points, WARP_WIDTH, WARP_HEIGHT)
    print(f"[3/4] Panel normalizado a {WARP_WIDTH}x{WARP_HEIGHT} px")

    output = build_output_dict(
        panel_bbox=panel_bbox,
        panel_points=panel_points,
        detection_method=method,
        warped_size=(WARP_WIDTH, WARP_HEIGHT),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    json_path = os.path.join(OUTPUT_DIR, f"{base}_result.json")
    save_json(output, json_path)
    print(f"[4/4] Resultado guardado en {json_path}")

    if DEBUG:
        dbg_panel = draw_panel_bbox(image, panel_points, (0, 255, 0), 2)
        if debug_data.get("anchors"):
            for name, box in debug_data["anchors"].items():
                # Dibujamos tambien las anclas cuando el panel se ha
                # resuelto por esa via para poder revisar el ajuste.
                pts = cv2.convexHull(
                    np.array(box, dtype=np.int32).reshape(-1, 1, 2)
                )
                cv2.polylines(dbg_panel, [pts], True, (0, 165, 255), 2)
                label_pt = tuple(pts.reshape(-1, 2)[0])
                cv2.putText(
                    dbg_panel,
                    name,
                    label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_debug_panel.jpg"), dbg_panel)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_warped.jpg"), warped)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_debug_gray_mask.jpg"),
            debug_data["gray_mask"],
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_debug_edge_mask.jpg"),
            debug_data["edge_mask"],
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_debug_detail_mask.jpg"),
            debug_data["detail_mask"],
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_debug_anchor_mask.jpg"),
            debug_data["anchor_mask"],
        )

        print(f"Imagenes debug guardadas en {OUTPUT_DIR}/")

    return output


def main():
    if len(sys.argv) < 2:
        print("Uso: python src/main.py <ruta_imagen>")
        print("     python src/main.py data/input/panel-1.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: no se encontro el archivo {image_path}")
        sys.exit(1)

    try:
        result = run_pipeline(image_path)
        print("\n" + json.dumps(result, indent=2))
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
