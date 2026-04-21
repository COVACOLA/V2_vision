# ERC Panel Vision

Proyecto base de visión por computador para la tarea de mantenimiento del ERC.

En el estado actual el objetivo está acotado a lo más importante para empezar bien:

- detectar el panel completo dentro de una imagen
- devolver su `bounding box`
- obtener sus cuatro esquinas
- rectificar el panel a un tamaño fijo con homografía
- guardar salidas reutilizables por módulos posteriores

Todavía no se detectan componentes individuales del panel. Esa parte se ha dejado fuera a propósito para estabilizar antes la percepción global.

#Archivos principales que quedan activos:



## Qué hace el pipeline actual

El flujo actual tiene cuatro pasos:

1. Carga una imagen de entrada.
2. Busca candidatos a panel usando máscaras y contornos.
3. Selecciona el mejor cuadrilátero según geometría y contenido visual.
4. Genera una salida JSON y una imagen rectificada del panel.

El script principal es:

```bash
python3 src/main.py data/input/panel-2.png
```

## Qué usa de OpenCV y por qué

En esta fase se usa visión clásica, no modelos entrenados.

Técnicas usadas:

- `HSV mask`
- `Canny`
- operaciones morfológicas
- `findContours`
- `minAreaRect`
- `warpPerspective`

Qué aporta cada una:

- `HSV mask`: ayuda a aislar superficies grises o poco saturadas, útiles para encontrar el cuerpo del panel.
- `Canny`: resalta bordes y estructura cuando el color por sí solo no basta.
- morfología (`close`, `open`, `dilate`): limpia ruido y conecta regiones útiles.
- `findContours`: extrae regiones candidatas.
- `minAreaRect`: aproxima cada candidato a un rectángulo rotado.
- `warpPerspective`: rectifica el panel para que en la siguiente fase todos los componentes se analicen en un mismo sistema de referencia.

Este enfoque se eligió porque:

- no necesita dataset anotado
- es rápido de probar
- es fácil de depurar
- encaja con un objeto de geometría conocida

## Cómo decide el detector qué región es el panel

El detector no se queda con cualquier rectángulo grande. Puntúa cada candidato con varias señales:

- relación de aspecto parecida al panel real `330 / 450`
- tamaño razonable respecto a la imagen completa
- relleno suficiente dentro del rectángulo candidato
- posición razonable dentro de la imagen
- cantidad de detalle interno
- penalización si toca demasiado los bordes de la imagen

Esto está implementado en [src/panel_detector.py](/home/cova/Documents/V2_vision/src/panel_detector.py).

## Instalación

Si quieres usar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencias:

- `opencv-contrib-python>=4.6`
- `numpy>=1.26,<2.0`

En este entorno concreto ya estaban disponibles `cv2 4.6.0` y `numpy 1.26.4`, así que no fue necesario instalar nada extra para validar el proyecto.

## Cómo probarlo

Desde la raíz del proyecto:

```bash
python3 src/main.py data/input/panel-2.png
```

También puedes lanzar las tres imágenes de ejemplo una por una:

```bash
python3 src/main.py data/input/panel-1.png
python3 src/main.py data/input/panel-2.png
python3 src/main.py data/input/panel-3.png
```

Si quieres correr todas las imágenes de prueba de una vez y comparar resultados automáticamente:

```bash
python3 run_all.py
```

Ese script:

- recorre `data/input/*.png`
- ejecuta el pipeline sobre cada imagen
- deja todas las salidas normales en `data/output/`
- guarda un resumen comparativo en `data/output/run_all_summary.json`

También puedes pasar imágenes concretas:

```bash
python3 run_all.py data/input/panel-2.png data/input/panel-3.png
```

Para verificar que el código compila:

```bash
python3 -m py_compile src/*.py
```

## Qué hace cada imagen de prueba

Las imágenes de `data/input/` no son equivalentes entre sí. Esto es importante para interpretar los resultados.

### `panel-1.png`

Es una vista técnica frontal del panel integrada dentro de un dibujo de referencia con cotas y estructura exterior.

Sirve para:

- comprobar si el detector confunde el panel con otras zonas rectangulares
- ver cómo responde ante figuras técnicas limpias
- detectar fallos cuando hay zonas vacías grandes por encima del panel

Riesgo de esta imagen:

- contiene mucho fondo blanco y geometría artificial
- no representa una captura real de cámara
- el detector puede verse atraído por regiones técnicas grandes y limpias

### `panel-2.png`

Es la mejor imagen del repositorio para la fase actual. Muestra el panel en una escena más parecida a una vista operativa con perspectiva.

Sirve para:

- validar la detección principal del panel
- verificar que la homografía corrige la inclinación
- comprobar que la `bbox` y las esquinas tienen sentido físico

Esta es la imagen más útil ahora para iterar el detector base.

### `panel-3.png`

Es una figura explicativa inclinada con etiquetas y anotaciones del layout.

Sirve para:

- forzar casos ambiguos
- comprobar robustez frente a texto, flechas y líneas de documentación
- medir si el detector se desvía hacia otras superficies con forma parecida

Riesgo de esta imagen:

- mezcla panel real dibujado con carcasa, etiquetas y regiones laterales
- tampoco es una captura de rover real

## Qué archivos genera el programa

Cada ejecución genera archivos dentro de `data/output/`.

Para una imagen `panel-2.png`, se crean estos ficheros:

- `panel-2_result.json`
- `panel-2_debug_panel.jpg`
- `panel-2_warped.jpg`
- `panel-2_debug_anchor_mask.jpg`
- `panel-2_debug_gray_mask.jpg`
- `panel-2_debug_edge_mask.jpg`
- `panel-2_debug_detail_mask.jpg`

Si usas `python3 run_all.py`, además se genera:

- `run_all_summary.json`

## Qué significa cada salida

### `*_result.json`

Es la salida consumible por otros módulos.

Campos:

- `bbox`: caja del panel en la imagen original con formato `[x, y, w, h]`
- `center`: centro de la caja
- `corners`: cuatro esquinas detectadas del panel
- `detection_method`: máscara que produjo el mejor candidato
- `normalized_size`: tamaño de la imagen rectificada

Ejemplo:

```json
{
  "panel": {
    "bbox": [106, 47, 324, 369],
    "center": [268, 231],
    "corners": [[106, 58], [333, 47], [430, 378], [181, 416]],
    "detection_method": "anchor_tags",
    "normalized_size": [660, 900]
  }
}
```

### `*_debug_panel.jpg`

Muestra la imagen original con el contorno verde del panel detectado.

Úsalo para responder esta pregunta:

- ¿el detector está encuadrando el panel correcto o se está yendo a otra región?

Si esta imagen falla, todo lo demás hereda ese error.

### `*_warped.jpg`

Es el panel ya rectificado a un sistema de referencia fijo.

Úsalo para responder esta pregunta:

- ¿la homografía realmente ha puesto el panel “de frente” y listo para segmentar componentes después?

Esta imagen será la base de la siguiente fase.

### `*_debug_gray_mask.jpg`

Es la máscara HSV que intenta quedarse con superficies grises o poco saturadas.

Úsala para ver:

- qué partes de la imagen se están considerando candidatas por color
- si la carcasa o el fondo están entrando donde no deberían

Si aquí aparece demasiado fondo, hay que ajustar umbrales HSV.

### `*_debug_anchor_mask.jpg`

Es la máscara binaria usada para detectar los cuadrados negros de referencia del panel.

Úsala para ver:

- si OpenCV está encontrando correctamente las tres anclas negras
- si el panel puede resolverse por geometría fija en vez de por contorno global

Cuando esta máscara funciona bien, el método más fiable pasa a ser `anchor_tags`, porque ajusta mejor el plano real del panel que un `minAreaRect` sobre todo el conjunto.

### `*_debug_edge_mask.jpg`

Es la máscara de bordes y morfología.

Úsala para ver:

- si el contorno del panel destaca correctamente
- si hay demasiado ruido estructural
- si líneas del dibujo o anotaciones están compitiendo con el panel

Cuando el color no ayuda, esta imagen suele ser la más importante.

### `*_debug_detail_mask.jpg`

Es una máscara de detalle interno basada también en bordes.

No se usa para recortar directamente el panel, sino para puntuar candidatos.

Su función es evitar que el detector prefiera regiones grandes pero vacías. El panel real suele tener más estructura interna que una superficie blanca o una cara lateral lisa.

Úsala para responder:

- ¿la región candidata contiene suficiente información interna para parecer un panel real?

## Cómo interpretar el campo `detection_method`

El JSON devuelve qué familia de máscara ganó:

- `anchor_tags`: ganó el ajuste basado en los tres cuadrados negros de referencia
- `gray_mask`: ganó el candidato basado en color o saturación baja
- `edge_mask`: ganó el candidato basado en bordes

Ahora mismo, `panel-2.png` suele resolverse mejor con `anchor_tags`.

## Qué está validado ahora mismo

Validado localmente:

- el código compila con `python3 -m py_compile src/*.py`
- el pipeline corre sobre `panel-1.png`
- el pipeline corre sobre `panel-2.png`
- el pipeline corre sobre `panel-3.png`

Resultado práctico:

- `panel-2.png` es el caso más representativo y el detector encaja bien el panel
- `panel-1.png` y `panel-3.png` sirven más como pruebas de estrés o ambigüedad que como validación final

## Limitaciones actuales

- solo detecta el panel completo
- no detecta todavía `switches`, `sockets` ni `indicators`
- las imágenes de prueba no son suficientes para validar un sistema real de competición
- faltan imágenes reales con diferentes iluminaciones, fondos y ángulos

## Siguiente paso recomendado

Una vez estable la detección del panel completo, la siguiente fase debería hacerse siempre sobre `*_warped.jpg`.

Estrategia recomendada:

1. aplicar `k-means` en `Lab` o `HSV`
2. separar fondo, panel y componentes
3. extraer regiones por contornos
4. filtrar por geometría
5. etiquetar como `switch`, `socket` o `indicator`
6. devolver solo coordenadas y centros

Eso encaja con el alcance que definiste:

- localizar panel
- recortar y normalizar
- segmentar componentes
- devolver coordenadas reutilizables

## Estructura del proyecto

```text
data/
├── input/
│   ├── panel-1.png
│   ├── panel-2.png
│   └── panel-3.png
└── output/

src/
├── config.py
├── homography.py
├── main.py
├── panel_detector.py
└── utils.py
```
