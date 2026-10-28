# Herramientas (`gt1_tools/`)

Programas de trabajo para preparar el banco: no validan un concepto, dejan listo lo que las pruebas necesitan. Se ejecutan como las pruebas:

```
uv run python gt1_tools/<herramienta>.py      # usa config/tools/<herramienta>.toml
```

| Herramienta | Para qué |
|---|---|
| [camera_viewer](#camera_viewer) | Ver la cámara en vivo y guardar imágenes |
| [calibrate](#calibrate) | Imprimir el tablero, calibrar la lente y la relación cámara-robot |
| [build_face_references](#build_face_references) | Juntar muestras de piel y hueso y generar las referencias del clasificador |
| teach_cell | Enseñar poses y zonas de `cell.toml` con el robot (próxima iteración) |
| summarize_run | Resumir una corrida contra los criterios (próxima iteración) |

---

## camera_viewer

Imagen en vivo, imágenes por segundo y resolución. `espacio` guarda la imagen en `captures/<camera_id>/`; `v` vuelve a la imagen en vivo después de detenerla con `x`.

Las imágenes guardadas sirven para probar la percepción sin cámara: `config/cameras/folder_basler.toml` lee `captures/basler_scA1390/`.

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

Qué método usa cada cámara lo decide `calibration_method` en `config/cameras/<cámara>.toml`. `camera_model` es el recomendado: corrige la altura de la pieza; la homografía solo es exacta a la altura del tablero.

---

## build_face_references

Junta muestras etiquetadas y genera las referencias del clasificador por histograma (`references/<camera_id>/face_histogram.npz`). Requiere la calibración de la cámara.

1. Dejar varias pechugas en la zona de toma, todas con la misma cara arriba.
2. `1` guarda cada pieza tomable como PIEL arriba; `2` como HUESO arriba. Mover las piezas y repetir: decenas de muestras de cada cara, con la variedad de la línea (tamaños, posiciones, iluminación).
3. `g` genera las referencias con **todas** las muestras guardadas (también las de sesiones anteriores) e informa cuántas muestras quedan bien clasificadas.

Las muestras se guardan como recortes con su máscara (`references/<camera_id>/samples/`): si se cambian `color_space`, `bins_per_channel` o `mask_erosion_px` en `config/perception/face_histogram.toml`, basta con volver a apretar `g`. `c` muestra las piezas detectadas sin guardar.
