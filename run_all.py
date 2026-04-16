import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import INPUT_DIR, OUTPUT_DIR  # noqa: E402
from main import run_pipeline  # noqa: E402
from utils import save_json  # noqa: E402


def _resolve_images(cli_paths):
    if cli_paths:
        images = []
        for raw_path in cli_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            images.append(path)
        return images

    return sorted(INPUT_DIR.glob("*.png"))


def _relative_to_project(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _build_item(image_path, result):
    panel = result["panel"]
    base = image_path.stem
    return {
        "image": image_path.name,
        "status": "ok",
        "detection_method": panel["detection_method"],
        "bbox": panel["bbox"],
        "center": panel["center"],
        "corners": panel["corners"],
        "result_json": _relative_to_project(OUTPUT_DIR / f"{base}_result.json"),
        "debug_panel": _relative_to_project(OUTPUT_DIR / f"{base}_debug_panel.jpg"),
        "warped": _relative_to_project(OUTPUT_DIR / f"{base}_warped.jpg"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el detector sobre todas las imagenes y resume resultados.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Rutas opcionales a imagenes concretas. Si se omiten, usa data/input/*.png",
    )
    args = parser.parse_args()

    images = _resolve_images(args.images)
    if not images:
        print("No se encontraron imagenes para procesar.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_images": len(images),
        "num_ok": 0,
        "num_failed": 0,
        "results": [],
    }

    print(f"Procesando {len(images)} imagen(es)...")
    for image_path in images:
        print(f"\n=== {image_path.name} ===")
        try:
            result = run_pipeline(str(image_path))
            item = _build_item(image_path, result)
            summary["results"].append(item)
            summary["num_ok"] += 1

            print(
                "OK  "
                f"method={item['detection_method']}  "
                f"bbox={item['bbox']}  "
                f"center={item['center']}"
            )
        except Exception as exc:
            summary["results"].append({
                "image": image_path.name,
                "status": "error",
                "error": str(exc),
            })
            summary["num_failed"] += 1
            print(f"ERROR  {exc}")

    summary_path = OUTPUT_DIR / "run_all_summary.json"
    save_json(summary, str(summary_path))

    print("\nResumen:")
    print(f"- OK: {summary['num_ok']}")
    print(f"- Errores: {summary['num_failed']}")
    print(f"- Archivo resumen: {_relative_to_project(summary_path)}")

    if summary["num_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
