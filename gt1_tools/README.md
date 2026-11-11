# Herramientas (`gt1_tools/`)

Programas de trabajo para preparar el banco: no validan un concepto, dejan listo lo que las pruebas necesitan. Se ejecutan como las pruebas:

```
uv run python gt1_tools/<herramienta>.py      # usa config/tools/<herramienta>.toml
```

| Herramienta | Para qué |
|---|---|
| [camera_viewer](#camera_viewer) | Ver la cámara en vivo y guardar imágenes |
| [record_video](#record_video) | Grabar video con la cámara |
| [video_frames](#video_frames) | Reproducir un video grabado y guardar imágenes de él |
| [face_preview](#face_preview) | Ver en vivo la detección de piezas y su cara (piel/hueso) |
| [calibrate](#calibrate) | Imprimir el tablero, calibrar la lente y la relación cámara-robot |
| [freedrive](#freedrive) | Mover el robot a mano viendo la cámara, con la punta del TCP marcada |
| [capture_empty_scene](#capture_empty_scene) | Guardar la imagen de la escena vacía para el detector por referencia (cámara mono scA1390) |
| [build_face_references](#build_face_references) | Juntar muestras de piel y hueso y generar las referencias del clasificador |
| teach_cell | Enseñar poses y zonas de `cell.toml` con el robot (próxima iteración) |
| summarize_run | Resumir una corrida contra los criterios (próxima iteración) |

---

## camera_viewer

Imagen en vivo, imágenes por segundo y resolución. `espacio` guarda la imagen en `captures/<camera_id>/`; `v` vuelve a la imagen en vivo después de detenerla con `x`.

Las imágenes guardadas sirven para probar la percepción sin cámara: `config/cameras/folder_basler.toml` lee `captures/basler_scA1390/`.

---

## record_video

Imagen en vivo; `r` empieza a grabar y `r` otra vez termina el video (también `x`). Queda en `videos/<camera_id>/<fecha_hora>.avi` y se anota en `runs/record_video/` con su duración y las imágenes por segundo reales de la cámara.

El video va en tiempo real aunque la cámara entregue menos imágenes por segundo que `video_fps` (cada imagen se repite hasta la siguiente). Cámara, códec y carpeta en `config/tools/record_video.toml`; para otra cámara, `--config` con un archivo `record_video_<variante>.toml`.

---

## video_frames

Reproduce un video a su velocidad real (no usa la cámara). `s` guarda la imagen que se está viendo en `frames/` junto al video, como `<video>_<número de imagen>.png`, y la anota en `runs/video_frames/`; `espacio` pausa y sigue; `v` lo reproduce de nuevo desde el inicio.

El video y el nombre de la carpeta van en `config/tools/video_frames.toml`. Las imágenes guardadas sirven para probar la percepción con una cámara de carpeta (`kind = "folder"`) que apunte a esa carpeta. Para probar con el video completo, una cámara de video (`kind = "video"`) en [face_preview](#face_preview).

---

## face_preview

Detección en vivo con cualquier cámara: cada pieza con su cara (verde piel, naranjo hueso, amarillo incierto; contorno gris = no tomable: las del borde muestran igual su cara, los grupos solo su estado) y en la barra de estado las imágenes por segundo, el tiempo de percepción y el conteo. `m` muestra la máscara del detector (para ajustar `config/perception/piece_*.toml`); `espacio` guarda la imagen en `captures/<camera_id>/`.

Con el clasificador por histograma, si la cámara no tiene referencias de cara solo detecta las piezas; el clasificador por textura (`face_texture`, el de `face_preview.toml`) no las necesita. Con `[calibration] method = "nominal_scale"` funciona sin calibrar la cámara con el robot (basta medir `mm_per_px`).

Para revisar sin cámara las imágenes de la C270 que guardó `video_frames` (`videos/webcam_logitech/frames/`):

```
uv run python gt1_tools/face_preview.py --config config/tools/face_preview_frames.toml
```

Con la Basler acA1600 (color) calibrada: `--config config/tools/face_preview_basler.toml` (detector por color `piece_color_background`, el de la webcam, sin escena vacía; clasificador `face_texture_basler`). Los dos están ajustados con otras cámaras: validarlos con esta. Con la scA1390 (mono) se usaba el detector por referencia `piece_reference` con la escena vacía de [capture_empty_scene](#capture_empty_scene); las imágenes del ajuste del 2026-11-04 están en `captures/basler_scA1390/`: `ref_hueso.png` y `ref_piel.png` (4 pechugas con el hueso y con la piel arriba, 8 de 8 bien) y `escena_4_pollos_*.png` (la escena anterior, con cartón).

**Con un video** (cámara `kind = "video"`, p. ej. `config/cameras/video_c270.toml`), procesa todas sus imágenes sin saltarse ninguna y graba lo que se ve en pantalla en la carpeta de la corrida: `runs/face_preview/<fecha_hora>/resultado_01.mp4`, con la duración del video original (si la percepción es más lenta que el video, la vista avanza más lento, pero el resultado no). El códec y la extensión van en `video_fourcc` y `video_file_extension` (mp4v + mp4 pesa bastante menos que MJPG + avi). `x` termina antes y guarda lo procesado; `v` lo procesa de nuevo desde el inicio en `resultado_02`, y así.

```
uv run python gt1_tools/face_preview.py --config config/tools/face_preview_video.toml
```

---

## calibrate

Al iniciar explica los dos métodos y pregunta qué paso hacer:

El tablero se elige en `board` de `config/tools/calibrate.toml`:

- `cognex_28x28_20mm` (el que se usa ahora): placa Cognex de 28 x 28 cuadros de 20 mm, sin marcadores, con una T al centro. Se detecta aunque se vea en parte o la T tape esquinas, pero no se sabe qué esquina es cuál: cada vista se numera desde lo que se ve (para calibrar da igual). Si la imagen es oscura y faltan esquinas, bajar `min_corner_contrast`; medir el espesor de la placa para `thickness_mm`.
- `charuco_a3`: tablero ChArUco para imprimir.

**0. Tablero.** Genera `calibration/tablero_<board>.png` (no hace falta con la placa Cognex). Imprimir en A3 al 100 % (sin "ajustar a la página"), pegar sobre una placa rígida y medir un cuadro con regla: si no mide `square_mm`, corregir `config/boards/<board>.toml`.

**1. Lente (intrínsecos).** Una vez por cámara, y de nuevo si cambia el lente, el foco o la resolución.

1. Mostrar el tablero a la cámara en distintas posiciones, distancias e inclinaciones (incluidas las esquinas de la imagen).
   Con la placa Cognex (560 mm, casi toda la vista): inclinarla 20-30° hacia cada lado, girarla en su plano y acercarla a la cámara para que cubra las esquinas de la imagen; sirven las vistas en que queda en parte fuera. La detección tarda ~0,6 s por imagen.
2. `espacio` agrega la vista actual (necesita al menos 6 esquinas visibles). Juntar `min_lens_views` vistas o más.
3. `c` calcula y guarda `calibration/<camera_id>/intrinsics.json`. Un error menor a ~0,5 px es bueno.

**2. Cámara-robot.** Requiere la lente calibrada y el pincho montado (`gripper = "pincho"`; su TCP se carga en el robot al iniciar). Pregunta qué método calcular: `homography`, `camera_model` o ambos.

1. Tablero plano sobre la mesa, dentro de la vista, y el robot fuera de la vista. `b` fija el tablero y elige las cinco esquinas a tocar entre las que se ven: las cuatro extremas y la más central.
2. `f` activa el freedrive. Llevar la punta del pincho a la esquina marcada en pantalla y apretar `t`. Repetir con las cinco (la marcada en amarillo es la siguiente; `u` deshace el último toque).
3. `c` calcula y guarda `homography.json` y/o `camera_model.json`, y desactiva el freedrive.
4. La barra de estado y el registro informan el error de los toques y del ajuste, y la **altura de la mesa** (superficie del tablero menos su espesor): copiarla a `table_z_mm` en `config/cell.toml`.

Qué método usa cada cámara lo decide `[calibration] method` en `config/cameras/<cámara>.toml`. `camera_model` es el recomendado: corrige la altura de la pieza; la homografía solo es exacta a la altura del tablero.

---

## freedrive

Imagen en vivo con la punta del TCP marcada según la calibración de la cámara (círculo amarillo) y el punto de la mesa justo debajo (círculo gris); la barra de estado muestra la posición del TCP en la base. `f` activa y desactiva el freedrive; al detener la imagen en vivo (`x`, `q`, cerrar la ventana) se apaga solo. Sirve para sacar el robot de la vista, llevarlo a una pose y leer su posición, y revisar a ojo la calibración: la marca amarilla debe caer sobre la punta.

**Poses con nombre** (`config/robot_poses.toml`): cada nombre de `poses_to_teach` en `config/tools/freedrive.toml` tiene una tecla (`1` el primero, `2` el segundo…) que guarda las articulaciones actuales con ese nombre, la posición del TCP (para quien lee el archivo) y la fecha. Si ya existía, la reemplaza y deja la anterior en el registro de la corrida; las demás poses no se tocan. Los programas eligen la suya por nombre:

- **`home`** (tecla `1`): el robot fuera de la imagen, con la herramienta hacia abajo y la punta más alta que `safe_height_mm`. Desde aquí capturan `pick_place_test` y `tray_loading_test` (`pose` de `[settings.clear_view]`) y parten a cada toma; sin ella enseñada, no arrancan. Después de enseñarla, probarla con `c` de `pick_place_test` a velocidad baja, con la mano en el paro.

`t` registra la pose actual en `runs/freedrive/<fecha_hora>/attempts.jsonl`: la posición y la orientación del TCP en la base, las articulaciones, y la imagen. Así se leen los valores de `config/cell.toml`:

- **Bandeja fija** (`tray_placement`): la punta del gripper en el centro del fondo de la bandeja. Se copian X e Y del TCP. El ángulo del lado largo sale de registrar dos puntos sobre ese lado.

Requiere el robot en modo remoto (el freedrive va por RTDE) y la calibración cámara-robot de la cámara. El gripper de `config/tools/freedrive.toml` es el montado (hoy `soft_x_asimetrico`): su TCP es el que se carga y se marca. Sirve también para confirmar el giro de 45° de su TCP: el eje Y que muestra la tablet tiene que ir de la parte ancha a la angosta.

---

## capture_empty_scene

Promedia `frames_to_average` imágenes de la escena sin piezas y las guarda en `references/<camera_id>/empty_scene.png`, la referencia del detector de piezas `kind = "reference"` (`config/perception/piece_reference.toml`). Se usa con la scA1390 (mono, `config/tools/capture_empty_scene.toml`); con la acA1600 los programas detectan las piezas por color y no la necesitan. Ese detector toma por pieza lo que cambió más de `min_difference_pct` respecto de la escena vacía, en relación con su brillo: la luz despareja, las manchas y los reflejos fijos del film se cancelan, y el lado del hueso se ve aunque tenga el mismo brillo medio que la mesa. Un cambio parejo de luz o de exposición se compensa solo.

```
uv run python gt1_tools/capture_empty_scene.py
```

1. Dejar la mesa como en las pruebas (film, bandeja), sin piezas y con el robot fuera de la vista (con [freedrive](#freedrive)).
2. `espacio` captura y guarda; `v` vuelve a la imagen en vivo.
3. Capturar de nuevo si se mueve el film o la cámara, o cambia la luz o la exposición: lo que cambie se toma por pieza. Sin la imagen, el detector por referencia se detiene al arrancar avisándolo.

---

## build_face_references

Junta muestras etiquetadas y genera las referencias del clasificador por histograma (`references/<camera_id>/face_histogram.npz`). Requiere la calibración de la cámara. El clasificador por textura no usa referencias: si `face_classifier` no es de tipo `histogram`, el programa se detiene avisándolo.

1. Dejar varias pechugas en la zona de toma, todas con la misma cara arriba.
2. `1` guarda cada pieza tomable como PIEL arriba; `2` como HUESO arriba. Mover las piezas y repetir: decenas de muestras de cada cara, con la variedad de la línea (tamaños, posiciones, iluminación).
3. `g` genera las referencias con **todas** las muestras guardadas (también las de sesiones anteriores) e informa cuántas muestras quedan bien clasificadas.

Las muestras se guardan como recortes con su máscara (`references/<camera_id>/samples/`): si se cambian `color_space`, `bins_per_channel` o `mask_erosion_px` en `config/perception/face_histogram.toml`, basta con volver a apretar `g`. `c` muestra las piezas detectadas sin guardar.
