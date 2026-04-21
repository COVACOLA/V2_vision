# ERC Panel Vision

## Que hace este proyecto

- Detecta el panel completo en una imagen
- Devuelve bounding box y esquinas
- Rectifica el panel a un tamano fijo con homografia
- Exporta resultados en JSON e imagenes de depuracion

## Requisitos

- Python 3.10+
- Dependencias de [requirements.txt](requirements.txt)

## Instalacion rapida

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Procesar una imagen:

```bash
python3 src/main.py data/input/panel-2.png
```

Procesar todas las imagenes de ejemplo:

```bash
python3 run_all.py
```

## Entrada y salida

- Entradas de ejemplo en [data/input](data/input)
- Salidas en [data/output](data/output)

Para una imagen panel-2.png se generan, entre otros:

- panel-2_result.json
- panel-2_warped.jpg
- panel-2_debug_panel.jpg

Si ejecutas run_all.py, tambien se genera:

- run_all_summary.json

## Estructura 

- [src/main.py](src/main.py): pipeline para una imagen
- [src/panel_detector.py](src/panel_detector.py): deteccion del panel
- [src/homography.py](src/homography.py): rectificacion
- [run_all.py](run_all.py): ejecucion por lote

## Estado actual

Esta version se centra en detectar el panel completo. No incluye aun deteccion de componentes internos.
