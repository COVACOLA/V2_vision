# README Implementacion

Este documento explica la implementacion actual del proyecto desde el punto de vista de codigo, algoritmo y estado real para la reunion.

El objetivo de esta fase no es detectar componentes individuales, sino dejar resuelto y estable el problema base:

- localizar el panel completo
- obtener sus esquinas
- calcular su `bounding box`
- rectificarlo a una vista normalizada
- guardar una salida reutilizable por otros modulos

## Nombre del algoritmo

No se esta usando un algoritmo unico cerrado del tipo YOLO o una red neuronal entrenada. Lo implementado es un:

`pipeline de vision clasica con OpenCV para deteccion geometrica del panel`

Si quieres nombrarlo de forma mas tecnica en la reunion, puedes decir:

`deteccion de panel basada en anclas visuales, contornos y homografia`

## Que hace el algoritmo

El algoritmo hace seis cosas:

1. busca referencias negras del panel cuando estan visibles
2. si las encuentra, estima la geometria del panel a partir de esas anclas
3. si no aparecen, genera candidatos por color y por bordes
4. puntua esos candidatos por forma, area, detalle interno y posicion
5. selecciona el cuadrilatero mas probable como panel
6. rectifica el panel con homografia para dejarlo en una vista frontal normalizada

En terminos simples:

- localiza el panel
- estima su contorno
- lo pone de frente para poder analizarlo despues

## Tecnicas concretas que usa

Las tecnicas de OpenCV que realmente se estan usando son:

- `thresholding` para detectar anclas negras
- segmentacion en `HSV`
- deteccion de bordes con `Canny`
- operaciones morfologicas
- `findContours`
- `minAreaRect`
- `findHomography`
- `warpPerspective`

Forma corta de explicarlo en la reunion:

“Estamos usando un pipeline de vision clasica con OpenCV que detecta el panel por anclas negras y por geometria de contornos, y despues aplica una homografia para normalizar la vista.”

## Objetivo de lo implementado

Se implemento una primera fase de percepcion centrada en la deteccion global del panel.

La idea tecnica es:

1. encontrar el panel en la imagen original
2. representarlo como un cuadrilatero
3. convertir ese cuadrilatero a una imagen frontal normalizada
4. devolver coordenadas limpias para fases posteriores

Esto deja preparada la base para una segunda fase de segmentacion de componentes como `switches`, `sockets` e `indicators`.

## Estructura del codigo

Los archivos activos del proyecto son:

- [src/config.py](/home/cova/Documents/V2_vision/src/config.py)
- [src/panel_detector.py](/home/cova/Documents/V2_vision/src/panel_detector.py)
- [src/homography.py](/home/cova/Documents/V2_vision/src/homography.py)
- [src/utils.py](/home/cova/Documents/V2_vision/src/utils.py)
- [src/main.py](/home/cova/Documents/V2_vision/src/main.py)
- [run_all.py](/home/cova/Documents/V2_vision/run_all.py)

## Que hace cada archivo

### `src/config.py`

Centraliza todos los parametros del sistema.

Contiene:

- rutas del proyecto (`PROJECT_ROOT`, `INPUT_DIR`, `OUTPUT_DIR`)
- geometria real del panel ERC (`330 x 450 mm`)
- resolucion del panel normalizado (`660 x 900 px`)
- umbrales de color para la mascara HSV
- umbrales del detector por bordes (`Canny`)
- reglas de filtrado geometricas del panel
- configuracion de las anclas negras del panel
- bandera `DEBUG`

Su funcion es evitar numeros magicos repartidos por el codigo y permitir ajustar el detector sin tocar la logica principal.

### `src/panel_detector.py`

Es el nucleo del sistema. Aqui vive el algoritmo de deteccion del panel.

Tiene dos estrategias principales:

1. deteccion por anclas negras
2. deteccion por candidatos geometricos usando mascaras

Funciones principales:

- `_build_gray_mask(image)`
- `_build_edge_mask(image)`
- `_build_detail_mask(image)`
- `_detect_anchor_tags(image)`
- `_panel_from_anchor_tags(image, anchors)`
- `_score_candidate(...)`
- `_find_best_candidate(...)`
- `detect_panel(image)`

Es el archivo mas importante para explicar en la reunion.

### `src/homography.py`

Se encarga de la rectificacion del panel.

Recibe:

- imagen original
- esquinas detectadas del panel
- tamaño deseado de salida

Y devuelve:

- panel rectificado
- matriz de homografia

Esto permite trabajar despues en un espacio fijo y comparable entre imagenes.

### `src/utils.py`

Agrupa funciones auxiliares:

- ordenar esquinas en orden consistente
- construir la `bbox`
- obtener el centro de la `bbox`
- construir el JSON final
- guardar JSON
- dibujar el panel detectado

No contiene logica de deteccion, sino soporte geometrico y de salida.

### `src/main.py`

Es el pipeline ejecutable para una sola imagen.

Secuencia:

1. carga la imagen
2. llama a `detect_panel`
3. obtiene la `bbox`
4. rectifica con homografia
5. genera el JSON final
6. guarda imagenes debug

Es el punto de entrada para ejecutar:

```bash
python3 src/main.py data/input/panel-2.png
```

### `run_all.py`

Es el script de evaluacion en lote.

Sirve para:

- recorrer todas las imagenes de `data/input/`
- ejecutar el pipeline completo en cada una
- guardar resultados individuales
- construir un resumen comparativo en `data/output/run_all_summary.json`

Para reunion es util porque permite enseñar rapidamente:

- que imagenes funcionan
- que metodo se uso en cada una
- que `bbox` se obtuvo

## Algoritmo implementado

El algoritmo actual es vision clasica con OpenCV. No usa aprendizaje profundo ni modelos entrenados.

### Estrategia 1: deteccion por anclas negras

Es la estrategia preferente cuando en la imagen aparecen tres cuadrados negros visibles del panel.

Flujo:

1. convertir imagen a escala de grises
2. umbralizar oscuro con `cv2.threshold(..., THRESH_BINARY_INV)`
3. limpiar ruido con apertura morfologica
4. buscar contornos
5. filtrar candidatos por:
   - area minima
   - forma casi cuadrada
   - ocupacion suficiente del bounding box
6. ordenar los tres candidatos detectados como:
   - `top_left`
   - `top_right`
   - `bottom_left`
7. usar esas anclas para estimar una homografia del panel completo
8. proyectar el rectangulo normalizado del panel sobre la imagen real

Por que se usa:

- da una geometria mas fiel que un rectangulo rotado global
- aprovecha elementos fisicos muy distintivos del panel
- mejora especialmente en `panel-2.png`

Metodo devuelto en el JSON:

- `anchor_tags`

### Estrategia 2: deteccion por mascaras y candidatos geometricos

Si las anclas no aparecen o no son utilizables, el detector usa un fallback basado en color, bordes y puntuacion de candidatos.

#### 2.1 Mascara gris HSV

`_build_gray_mask(image)` intenta aislar superficies grises o poco saturadas.

Pasos:

- convertir a HSV
- aplicar `cv2.inRange`
- cerrar huecos con `MORPH_CLOSE`
- eliminar ruido con `MORPH_OPEN`

Sirve para encontrar regiones compatibles con la superficie del panel.

Metodo posible:

- `gray_mask`

#### 2.2 Mascara de bordes

`_build_edge_mask(image)` resalta la estructura del panel cuando el color no basta.

Pasos:

- convertir a gris
- mejorar contraste con CLAHE
- suavizar con `GaussianBlur`
- detectar bordes con `Canny`
- dilatar
- cerrar regiones con morfologia

Sirve para encontrar contornos fuertes del panel o de la estructura interna.

Metodo posible:

- `edge_mask`

#### 2.3 Mascara de detalle interno

`_build_detail_mask(image)` no se usa para detectar el panel directamente, sino para puntuar candidatos.

Idea:

- una region que realmente sea el panel suele tener mas estructura interna que una cara vacia

Esto ayuda a penalizar superficies lisas y grandes que por geometria sola podrian parecer validas.

### Puntuacion de candidatos

Cuando se usa el fallback, cada contorno candidato se convierte a un rectangulo con `minAreaRect` y se puntua.

Se valoran:

- aspecto parecido al panel real
- area razonable respecto a la imagen
- relleno del contorno dentro del rectangulo
- detalle interno
- variacion de intensidad
- cercania al centro de la imagen

Se penaliza:

- tocar demasiados bordes de la imagen

La mejor region segun esa puntuacion se toma como panel.

## Flujo completo del sistema

El flujo real del programa es este:

```text
Imagen original
-> deteccion del panel
-> esquinas del panel
-> bounding box
-> homografia
-> panel rectificado
-> JSON de salida + imagenes debug
```

## Salidas que genera

Para cada imagen se generan:

- `*_result.json`
- `*_debug_panel.jpg`
- `*_warped.jpg`
- `*_debug_gray_mask.jpg`
- `*_debug_edge_mask.jpg`
- `*_debug_detail_mask.jpg`
- `*_debug_anchor_mask.jpg`

### `*_result.json`

Contiene:

- `bbox`
- `center`
- `corners`
- `detection_method`
- `normalized_size`

Es la salida consumible por otros modulos.

### `*_debug_panel.jpg`

Muestra el cuadrilatero detectado sobre la imagen original.

### `*_warped.jpg`

Muestra el panel ya rectificado.

### `*_debug_gray_mask.jpg`

Muestra la segmentacion basada en color.

### `*_debug_edge_mask.jpg`

Muestra la segmentacion basada en bordes.

### `*_debug_detail_mask.jpg`

Muestra el nivel de estructura interna usado para puntuar candidatos.

### `*_debug_anchor_mask.jpg`

Muestra la segmentacion que busca las anclas negras.

## Estado actual de lo implementado

Implementado:

- deteccion del panel completo
- `bounding box`
- esquinas del panel
- homografia
- salida JSON reutilizable
- ejecucion por imagen
- ejecucion por lote con `run_all.py`
- salida visual de depuracion

No implementado todavia:

- deteccion de `switches`
- deteccion de `sockets`
- deteccion de `indicators`
- clasificacion de componentes
- seguimiento temporal
- integracion con brazo o planner

## Por que se eligio este enfoque

Se eligio una solucion clasica con OpenCV porque para esta reunion interesaba tener:

- algo funcional ya
- explicabilidad tecnica
- rapidez de iteracion
- facilidad para depurar errores
- independencia de datasets etiquetados

Para una fase inicial del proyecto tiene sentido. La complejidad se mantiene controlada y deja base para crecer.

## Resultados actuales

En las pruebas del repositorio:

- `panel-2.png` suele resolverse mejor por `anchor_tags`
- `panel-1.png` tambien puede resolverse por `anchor_tags`
- `panel-3.png` suele caer en `edge_mask` porque no presenta las anclas de forma clara

Esto indica que el sistema ya tiene:

- una estrategia principal mas fuerte cuando hay referencias negras
- una estrategia de respaldo cuando no las hay

## Limitaciones tecnicas actuales

- las coordenadas de `ANCHOR_TAG_LAYOUT` son aproximadas
- las imagenes del repositorio no equivalen a una base real de camara de rover
- el detector aun depende bastante de la presentacion visual del panel
- no hay validacion con video ni con escenas reales complejas

## Que se puede contar en la reunion

Si necesitas resumirlo de forma clara, estos son los puntos utiles:

1. Se ha implementado la fase base de percepcion del panel completo.
2. El sistema detecta el panel, calcula esquinas, `bbox` y una vista rectificada.
3. Se usa vision clasica con OpenCV, sin redes neuronales por ahora.
4. Hay dos estrategias:
   - anclas negras del panel
   - fallback por color, bordes y puntuacion geometrica
5. La salida ya esta preparada para ser consumida por modulos posteriores.
6. El siguiente paso natural es segmentar componentes sobre la imagen rectificada.

## Mensaje tecnico corto para presentar

Puedes decirlo asi:

“Para esta iteracion hemos cerrado la primera capa del sistema de vision: detectar el panel completo en imagen, estimar su geometria y normalizarlo con homografia. La implementacion actual usa OpenCV clasico con una estrategia preferente basada en anclas negras del panel y un fallback por mascaras y scoring geometrico. Con esto ya podemos entregar coordenadas estables del panel y preparar la siguiente fase, que sera la segmentacion de componentes sobre la vista rectificada.”

## Comandos utiles para la reunion

Ejecutar una sola imagen:

```bash
python3 src/main.py data/input/panel-2.png
```

Ejecutar todas:

```bash
python3 run_all.py
```

Compilar para validar sintaxis:

```bash
python3 -m py_compile src/*.py run_all.py
```
