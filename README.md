# GT1: carga robotizada de pechuga entera en bandeja (piloto)

Piloto del proyecto 171 (Agrosuper) para automatizar el puesto donde hoy dos operadoras toman pechugas enteras con hueso desde una canaleta, las dan vuelta si vienen con el hueso arriba y las acomodan en una bandeja PET con la piel arriba.

El piloto valida, con un **UR10e**, un **soft gripper** y una **cámara fija** (Basler ace acA1600 color por GigE; antes la Scout scA1390 monocromática) que ve la zona de toma y la bandeja:

1. que la cámara distinga si la pieza viene con la piel o con el hueso arriba;
2. que el robot tome la pieza y la deje en la bandeja con la piel arriba;
3. que el robot la voltee cuando viene con el hueso arriba;
4. una rutina automática que haga todo lo anterior y verifique el resultado con la cámara.

El código está hecho para crecer: cámara, robot, gripper, detectores y estrategias de toma, volteo y colocación son piezas intercambiables, que se eligen en archivos de configuración.

El vocabulario del proyecto (pieza, cara, zona, estación, corrida…) está en [GLOSSARY.md](GLOSSARY.md). Las reglas del código, para personas y agentes (Codex, Claude Code), están en [AGENTS.md](AGENTS.md).

## Instalación

Requiere Windows, el **SDK MVS** (para la cámara GigE) y [uv](https://docs.astral.sh/uv/).

```
uv sync          # crea .venv con Python 3.12 y las dependencias
uv run pytest    # verifica la instalación (sin hardware)
```

En este equipo uv quedó instalado para el usuario en
`%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts`:
agregar esa carpeta al `PATH` para usar `uv` directamente.

Para desarrollar con VS Code, elegir el intérprete `.venv\Scripts\python.exe`.

## Flujo de trabajo en el banco

| Paso | Programa | Resultado |
|---|---|---|
| 1. Ver la cámara, ajustar foco y exposición | `gt1_tools/camera_viewer.py` | Imagen nítida; completar `expected_model` / `expected_serial` en `config/cameras/basler_acA1600.toml` con lo que informa la cámara |
| 2. Preparar el tablero | `gt1_tools/calibrate.py` (opción 0) o la placa Cognex | Tablero ChArUco impreso en A3, o la placa Cognex 28x28 de 20 mm (`board` en `config/tools/calibrate.toml`) |
| 3. Calibrar la lente | `gt1_tools/calibrate.py` (opción 1) | `calibration/basler_acA1600/intrinsics.json` |
| 4. Calibrar cámara-robot (con el pincho) | `gt1_tools/calibrate.py` (opción 2) | `camera_model.json` y la altura de la mesa para `cell.toml` |
| 5. Definir las zonas | editar `config/cell.toml` | Zonas de toma (mitad superior de la imagen), bandejas (mitad inferior) y volteo |
| 6. Juntar muestras de piel y hueso (solo para el clasificador por histograma) | `gt1_tools/build_face_references.py` | `references/basler_acA1600/face_histogram.npz` |
| 7. Medir el acierto de cara | `gt1_test/face_test.py` | Registro en `runs/face_test/` |
| 8. Probar el gripper | `gt1_test/gripper_test.py` | TCP y carga cargados en el robot; acciones verificadas |
| 9. Enseñar la pose fuera de la vista y la bandeja fija | `gt1_tools/freedrive.py` (`1` guarda `home`, `t` registra la pose) | `home` en `config/robot_poses.toml` y `tray_placement` en `config/cell.toml` |
| 10. Tomar y dejar en la bandeja | `gt1_test/pick_place_test.py` | Registro en `runs/pick_place_test/` |
| 11. Cargar las bandejas detectadas | `gt1_test/tray_loading_test.py` | Registro en `runs/tray_loading_test/` |
| 12. Cargar la bandeja fija con tope, una pechuga tras otra | `gt1_tools/freedrive.py` (`2` guarda `tray_drop`) y `gt1_test/demo_static_tray.py` | Registro en `runs/demo_static_tray/` |

Detalle de cada prueba en [gt1_test/README.md](gt1_test/README.md) y de cada herramienta en [gt1_tools/README.md](gt1_tools/README.md).

### Valores por medir

Los archivos de configuración traen valores iniciales marcados **POR MEDIR** o **POR COMPLETAR**:

- `config/grippers/soft_2dedos.toml`: TCP (punta de los dedos cerrados), masa y centro de gravedad. Se cargan en el robot y afectan la detección de colisiones.
- `config/grippers/pincho.toml`: largo del pincho de calibración.
- `config/grippers/gripper_gorilla-4_ang10.toml`: masa, centro de gravedad y tiempos de cierre y apertura.
- `config/grippers/soft_x_asimetrico.toml` (el de `tray_loading_test` y `pick_place_test`): largo del TCP (se supuso 136 mm, como el Gorilla), masa, centro de gravedad y tiempos. **Confirmar con freedrive** que el giro de 45° del TCP deja su eje Y de la parte ancha (-Y) a la angosta (+Y); si la X está girada al otro lado, cambiar el signo de Rz.
- `config/robot_poses.toml`: la pose `home` (fuera de la vista), **POR ENSEÑAR** con freedrive; sin ella, `pick_place_test` y `tray_loading_test` no arrancan. La pose `tray_drop` (soltar sobre la bandeja fija con tope, fuera de la vista), **POR ENSEÑAR**; sin ella, `demo_static_tray` no arranca.
- `config/cell.toml`: bandeja fija (`tray_placement`), **POR ENSEÑAR** antes de correr `pick_place_test`.
- `config/strategies/`: alturas y velocidades de la toma y la colocación; hacia dónde quedan los lóbulos en la bandeja (`lobes_toward_deg`, **POR DEFINIR** con Agrosuper).
- `config/cameras/basler_acA1600.toml`: exposición y ganancia (`genicam_nodes`, **POR AJUSTAR**).
- `config/cell.toml`: altura de la mesa (la da la calibración) y zonas.
- `config/trays/bandeja_pet.toml`: el borde (217 x 167 mm) y el alto (80 mm) son medidos; el fondo (175 x 122 mm) se midió en la imagen, ±5 mm: **medirlo con regla**. El borde de sellado, pedir la ficha a Agrosuper. Antes de cargar, comparar el centro que da `tray_loading_test` (`c`) con la punta del gripper en el centro de una bandeja (freedrive).

## Configuración (`config/`)

Un archivo por componente; cada programa elige los suyos por nombre.

| Carpeta / archivo | Contenido |
|---|---|
| `station.toml` | Hardware montado según la última configuración confirmada. Lo actualizan los programas |
| `cell.toml` | Disposición del banco: altura de la mesa, zonas (toma, bandeja, volteo) en mm de la base y bandeja fija |
| `robot_poses.toml` | Poses del robot con nombre, en articulaciones (`home`: fuera de la vista, desde donde se captura; `tray_drop`: soltar sobre la bandeja fija con tope). Las escribe freedrive al enseñarlas; los programas las eligen por nombre |
| `robots/` | `UR10e.toml` (RTDE), `sim.toml` (simulado) |
| `cameras/` | `basler_scA1390.toml` y `basler_acA1600.toml` (MVS), `logitech_c270.toml` (webcam USB), `folder_basler.toml` y `folder_c270.toml` (imágenes guardadas), `video_c270.toml` (video grabado) |
| `grippers/` | TCP, carga, acciones → salidas digitales y lado ancho (`wide_side`): `soft_x_asimetrico.toml` (soft gripper en X, TCP girado 45°, parte ancha hacia -Y), `gripper_gorilla-4_ang10.toml`, `soft_2dedos.toml`, `pincho.toml` |
| `trays/` | Medidas de cada bandeja |
| `perception/` | Detectores de piezas: `piece_background.toml` (piezas claras sobre fondo oscuro), `piece_color_background.toml` (fondo de color, ignora sombras; el de la webcam y la acA1600), `piece_reference.toml` (diferencia con la escena vacía; se usó con la Basler mono, que no ve color), `piece_skin_outline.toml` (solo piel arriba, por su contorno: sin escena vacía, con cualquier fondo). Detector de bandejas: `tray_rim_and_floor.toml` (el pliegue del fondo en los lados hacia la sombra de las paredes y el labio del borde en los otros; también en la sombra y bajo los reflejos del film). Clasificadores de cara: `face_texture.toml` (textura: piel lisa, hueso con costillas; sin referencias, mono o color), `face_histogram.toml` / `face_histogram_hsv.toml` (histograma contra referencias de piel y hueso) |
| `boards/` | Tablero de calibración |
| `strategies/` | Estrategias de toma y colocación: `pick_top_down.toml`, `place_top_down.toml` (alturas y velocidades) |
| `tests/`, `tools/` | Un archivo por programa: componentes elegidos y ajustes propios |

La validación es estricta: un campo que falta, que sobra o con el tipo equivocado detiene el programa con el archivo y el campo en el mensaje. La descripción de cada campo está en `src/gt1/config/schemas.py`.

Para probar algo puntual sin tocar la configuración de siempre, se copia el archivo del programa y se usa `--config`:

```
uv run python gt1_test/face_test.py --config config/tests/face_test_webcam.toml
```

Cuando un programa pide hardware distinto al de `station.toml`, muestra las diferencias, pide confirmar que lo montado es correcto y actualiza `station.toml`. Lo simulado (robot `sim`, cámara de carpeta) no se compara.

## Registros de cada corrida

Cada ejecución crea `runs/<programa>/<fecha_hora>/`:

| Archivo | Contenido |
|---|---|
| `attempts.jsonl` | Una línea JSON por intento: los datos para medir los criterios |
| `images/` | Imágenes de cada intento (captura y superposición) |
| `config/` | Copia de los archivos de configuración usados |
| `run.log` | Lo que pasó: conexiones, advertencias, errores |

`runs/`, `captures/`, `videos/` y las muestras de `references/` no se versionan (`.gitignore`): se respaldan aparte.

## Mapa del código

```
src/gt1/                       biblioteca (las capas y sus reglas están en AGENTS.md)
  geometry.py                  rotaciones, transformaciones rígidas, ajuste de Kabsch
  paths.py                     carpetas del proyecto
  domain/                      tipos del problema, sin hardware
    pose.py                    Pose: posición [mm] + rotación
    frame.py                   Frame: una imagen y su hora de captura
    pieces.py                  Piece, PieceStatus, Face (piel/hueso/incierto), FaceResult
    io.py                      entradas y salidas digitales del controlador
    tool.py                    ToolSettings (TCP + carga)
    tray.py, zone.py           bandeja detectada, zona de la celda
    plan.py                    Plan y sus pasos (MoveLinear, MoveJoints, GripperCommand, Wait, SetOutput,
                               WaitForInput)
  config/
    schemas.py                 modelo pydantic de cada archivo de config/
    loading.py                 ConfigRepository: lectura y validación estricta
    station.py                 diferencias con station.toml
    robot_poses.py             guardar las poses enseñadas en robot_poses.toml
  hardware/
    camera/                    Camera; mvs.py (Basler GigE), webcam.py, folder.py, video.py (video grabado)
    robot/                     Robot; ur_rtde.py (UR10e), sim.py
    gripper/                   Gripper; digital_outputs.py, fixed.py (pincho), tool_settings.py
  calibration/
    boards/                    CalibrationBoard; charuco.py, chessboard.py (sin marcadores, p. ej. Cognex),
                               corner_grid.py (grilla de ajedrez que crece desde una semilla); esquinas a tocar
    intrinsics.py              modelo de la lente
    extrinsics.py              tablero en la base a partir de toques; ajuste de los dos métodos
    projection.py              PixelProjector: HomographyProjector, CameraModelProjector
    store.py                   archivos de calibration/<camera_id>/
  perception/
    image_ops.py               máscaras, zona en la imagen, escala de grises
    pieces/                    PieceDetector; background.py (fondo de alto contraste), reference.py (diferencia
                               con la escena vacía), empty_scene.py (su archivo), skin_outline.py (piel arriba
                               por su contorno), midline.py (línea central y lóbulos), contours.py (medir en la base)
    face/                      FaceClassifier; texture.py, histogram.py, references.py (muestras y referencias)
    tray/                      TrayDetector; rim_and_floor.py (labio del borde y pliegue del fondo en vistas
                               cenitales a cada altura, de lo grueso a lo fino)
    verify.py                  PlacementVerifier (interfaz)
    scene.py                   ScenePerception: de una imagen a la escena (zona, piezas y su cara)
  planning/                    PickStrategy, FlipStrategy, PlaceStrategy; top_down_pick.py, top_down_place.py
                               (con los lóbulos hacia donde se pide), assignment.py (qué pieza va a qué bandeja),
                               clear_view.py (a la pose fuera de la vista, subiendo antes si está bajo; sin
                               moverse si ya está ahí), tray_station.py (soltar en la bandeja fija con tope)
  runtime/
    program.py                 run_program: argumentos, logging, config, registro, errores
    station.py                 confirmación del hardware al iniciar
    factories.py               construye cada componente según su kind (y la bandeja fija)
    executor.py                PlanExecutor: lleva a cabo un Plan paso por paso; se corta limpio
    captures.py                captura con el robot quieto, escena vacía
    clear_view.py              ClearViewCamera: lleva el robot fuera de la vista y captura (sin repetir
                               la espera de vibración si el robot ya estaba quieto)
    tray_stopper.py            TrayStopper: tope de la bandeja fija, cerrado en segundo plano
    records.py                 lo que se registra de piezas, bandejas y planes
    console.py                 OperatorConsole: ventana, teclas, ayuda, barra de estado
    runner.py                  hilo de trabajo con señal de parada
    overlay.py                 dibujo de zonas, piezas, bandejas, línea central y paneles de texto
    run_log.py                 registro de la corrida
    video_recording.py         grabación de video en tiempo real (record_video, face_preview)
    terminal.py                preguntas por la terminal
gt1_test/                      pruebas de validación (ver su README)
gt1_tools/                     herramientas (ver su README)
tests/                         pytest sin hardware; test_architecture.py verifica las capas; scenes.py arma
                               escenas sintéticas y test_real_scenes.py repasa fotos de la Basler (tests/data/)
```

## Estado

**Hecho:** biblioteca completa para percepción y arranque; `face_test`, `gripper_test`, `camera_viewer`, `calibrate`, `build_face_references`; 315 tests sin hardware; prueba de humo de todos los programas con cámara de carpeta y robot simulado, incluida la calibración completa con un tablero sintético.

**Antes de mover el robot:** el UR10e real aún no ha corrido `pick_place_test`, `tray_loading_test` ni `demo_static_tray` de punta a punta. Un intento de `pick_place_test` chocó al ir a una pose que nadie había enseñado; por eso las poses viven en `config/robot_poses.toml` y los programas no arrancan si falta la que usan. Enseñar primero las poses y completar los valores de [Valores por medir](#valores-por-medir), y la primera vez correr cada prueba a velocidad baja con la mano en la parada de emergencia. `config/station.toml` describe el banco tal como estaba montado al escribirse: el primer programa que se corra muestra las diferencias y pide confirmarlas.

**Riesgos abiertos** (detalle en [gt1_test/README.md](gt1_test/README.md)): el detector de bandejas da los centros hasta ~12 mm corridos hacia +Y de la base (sin explicar aún); no detecta bandejas giradas ~40° en la imagen ni las cruzadas por la franja de luz de abajo a la izquierda (test `xfail` con `escena_3_bandejas_giradas`); una bandeja cortada por el borde de la imagen puede reemplazarse por una falsa girada 90°; en pechugas casi simétricas los lóbulos pueden salir invertidos (la verificación después de cargar lo detecta).

**En curso:** `pick_place_test` con la bandeja fija: estrategias de toma y colocación desde arriba, ejecutor y verificación con la cámara, probados sin hardware; falta enseñar las poses y probarlo en el banco.

**Carga de bandejas detectadas (2026-11-05):** `tray_loading_test` detecta las pechugas con la piel arriba (`piece_skin_outline`, sin escena vacía) y las bandejas (`tray_rim_and_floor`), empareja cada pieza con la bandeja libre más cercana y la deja con la línea central a lo largo de la bandeja y los lóbulos hacia `lobes_toward_deg`; cada carga se verifica con la cámara. Validado con fotos (ver `tests/test_real_scenes.py`) y con robot simulado; falta enseñar las poses y probarlo en el banco.

**Cuatro bandejas y gripper en X (2026-11-10):** con la acA1600, `escena_4_bandejas_4_pollos` (4 bandejas a 8°, 22°, 38° y -36°; 4 pechugas con distinto giro): el detector de bandejas las encuentra a ≤ 13 mm y ≤ 7° de lo medido a mano y el de pechugas, las 4 (también con ruido de sensor). La toma pone la parte ancha del gripper asimétrico hacia los lóbulos (`planning/grasp.py`) y la colocación la mantiene así hasta dejar los lóbulos hacia `lobes_toward_deg`. Cargado de punta a punta con robot simulado sobre esa foto; falta enseñar las poses y probarlo en el banco.

**Bandeja fija con tope (2026-11-11):** `demo_static_tray` carga una tras otra las pechugas de la zona de toma (de izquierda a derecha, o al revés con `pick_from`) en la bandeja que detiene un tope (DO2) sobre la cinta: espera el tope cerrado y el sensor (DI1), suelta, abre el tope y va por la pieza siguiente mientras la bandeja cargada sale; `TrayStopper` cierra el tope en segundo plano 2,5 s después de que el sensor deja de verla (`stopper_close_delay_s`). Al iniciar lo cierra y verifica la salida. La foto siguiente se toma desde la pose de soltar, sin moverse. Los planes tienen pasos de salida digital y de espera a una entrada o salida (`SetOutput`, `WaitForInput`, `WaitForOutput`). Probado con robot simulado y una cinta simulada; falta enseñar `tray_drop` y probarlo en el banco.

**Próximas iteraciones:** límites de la celda → verificador de colocación (borde de sellado) → `flip_test` (estación de volteo) → `demo` → `teach_cell`, `summarize_run`. Después: un clasificador YOLO como alternativa, URSim y un gemelo digital como implementaciones de `Robot`, y la cinta en movimiento.
