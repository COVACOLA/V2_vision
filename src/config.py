from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Geometria real del panel ERC. La relacion de aspecto se usa
# como una de las restricciones geometricas del detector.
PANEL_WIDTH_MM = 330
PANEL_HEIGHT_MM = 450
PANEL_ASPECT_RATIO = PANEL_WIDTH_MM / PANEL_HEIGHT_MM

# Resolucion del panel normalizado tras la homografia.
PX_PER_MM = 2
WARP_WIDTH = PANEL_WIDTH_MM * PX_PER_MM
WARP_HEIGHT = PANEL_HEIGHT_MM * PX_PER_MM

# Rango HSV para superficies poco saturadas y grises, que
# suelen corresponder al cuerpo del panel.
PANEL_HSV_LOWER = (0, 0, 35)
PANEL_HSV_UPPER = (180, 85, 235)

# Umbrales del detector por bordes.
CANNY_LOW = 60
CANNY_HIGH = 180

# Filtros geometricos globales para descartar candidatos
# imposibles o demasiado alejados de la forma esperada.
MIN_PANEL_AREA_RATIO = 0.05
MAX_PANEL_AREA_RATIO = 0.80
MAX_ASPECT_ERROR = 0.35
MAX_BORDER_TOUCHES = 2

# Anclas negras visibles en varias imagenes del panel.
# Coordenadas normalizadas aproximadas dentro del panel rectificado:
# (x, y, w, h)
ANCHOR_TAG_LAYOUT = {
    "top_left": (0.03, 0.025, 0.155, 0.10),
    "top_right": (0.815, 0.025, 0.155, 0.10),
    "bottom_left": (0.03, 0.755, 0.155, 0.10),
}

ANCHOR_THRESHOLD = 45
ANCHOR_MIN_AREA = 200
ANCHOR_MIN_FILL = 0.60
ANCHOR_MIN_ASPECT = 0.70

# Cuando esta activado, el pipeline guarda mascaras y resultados
# intermedios para inspeccion visual.
DEBUG = True
