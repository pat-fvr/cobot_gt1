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

**Con un video** (cámara `kind = "video"`, p. ej. `config/cameras/video_c270.toml`), procesa todas sus imágenes sin saltarse ninguna y graba lo que se ve en pantalla en la carpeta de la corrida: `runs/face_preview/<fecha_hora>/resultado_01.mp4`, con la duración del video original (si la percepción es más lenta que el video, la vista avanza más lento, pero el resultado no). El códec y la extensión van en `video_fourcc` y `video_file_extension` (mp4v + mp4 pesa bastante menos que MJPG + avi). `x` termina antes y guarda lo procesado; `v` lo procesa de nuevo desde el inicio en `resultado_02`, y así.

```
uv run python gt1_tools/face_preview.py --config config/tools/face_preview_video.toml
```

---

## calibrate

Al iniciar explica los dos métodos y pregunta qué paso hacer:

**0. Tablero.** Genera `calibration/tablero_charuco_a3.png`. Imprimir en A3 al 100 % (sin "ajustar a la página"), pegar sobre una placa rígida y medir un cuadro con regla: si no mide `square_mm`, corregir `config/boards/charuco_a3.toml`.

**1. Lente (intrínsecos).** Una vez por cámara, y de nuevo si cambia el lente, el foco o la resolución.

1. Mostrar el tablero a la cámara en distintas posiciones, distancias e inclinaciones (incluidas las esquinas de la imagen).
2. `espacio` agrega la vista actual (necesita al menos 6 esquinas visibles). Juntar `min_lens_views` vistas o más.
3. `c` calcula y guarda `calibration/<camera_id>/intrinsics.json`. Un error menor a ~0,5 px es bueno.

**2. Cámara-robot.** Requiere la lente calibrada y el pincho montado (`gripper = "pincho"`; su TCP se carga en el robot al iniciar). Pregunta qué método calcular: `homography`, `camera_model` o ambos.

1. Tablero plano sobre la mesa, dentro de la vista, y el robot fuera de la vista. `b` fija el tablero (deben verse las cinco esquinas a tocar).
2. `f` activa el freedrive. Llevar la punta del pincho a la esquina marcada en pantalla y apretar `t`. Repetir con las cinco (la marcada en amarillo es la siguiente; `u` deshace el último toque).
3. `c` calcula y guarda `homography.json` y/o `camera_model.json`, y desactiva el freedrive.
4. La barra de estado y el registro informan el error de los toques y del ajuste, y la **altura de la mesa** (superficie del tablero menos su espesor): copiarla a `table_z_mm` en `config/cell.toml`.

Qué método usa cada cámara lo decide `[calibration] method` en `config/cameras/<cámara>.toml`. `camera_model` es el recomendado: corrige la altura de la pieza; la homografía solo es exacta a la altura del tablero.

---

## build_face_references

Junta muestras etiquetadas y genera las referencias del clasificador por histograma (`references/<camera_id>/face_histogram.npz`). Requiere la calibración de la cámara. El clasificador por textura no usa referencias: si `face_classifier` no es de tipo `histogram`, el programa se detiene avisándolo.

1. Dejar varias pechugas en la zona de toma, todas con la misma cara arriba.
2. `1` guarda cada pieza tomable como PIEL arriba; `2` como HUESO arriba. Mover las piezas y repetir: decenas de muestras de cada cara, con la variedad de la línea (tamaños, posiciones, iluminación).
3. `g` genera las referencias con **todas** las muestras guardadas (también las de sesiones anteriores) e informa cuántas muestras quedan bien clasificadas.

Las muestras se guardan como recortes con su máscara (`references/<camera_id>/samples/`): si se cambian `color_space`, `bins_per_channel` o `mask_erosion_px` en `config/perception/face_histogram.toml`, basta con volver a apretar `g`. `c` muestra las piezas detectadas sin guardar.
