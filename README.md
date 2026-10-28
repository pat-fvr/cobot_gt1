# GT1: carga robotizada de pechuga entera en bandeja (piloto)

Piloto del proyecto 171 (Agrosuper) para automatizar el puesto donde hoy dos operadoras toman pechugas enteras con hueso desde una canaleta, las dan vuelta si vienen con el hueso arriba y las acomodan en una bandeja PET con la piel arriba.

El piloto valida, con un **UR10e**, un **soft gripper** y una **cámara fija** (Basler Scout scA1390 por GigE) que ve la zona de toma y la bandeja:

1. que la cámara distinga si la pieza viene con la piel o con el hueso arriba;
2. que el robot tome la pieza y la deje en la bandeja con la piel arriba;
3. que el robot la voltee cuando viene con el hueso arriba;
4. una rutina automática que haga todo lo anterior y verifique el resultado con la cámara.

El código está hecho para crecer: cámara, robot, gripper, detectores y estrategias de toma, volteo y colocación son piezas intercambiables, que se eligen en archivos de configuración.

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
| 1. Ver la cámara, ajustar foco y exposición | `gt1_tools/camera_viewer.py` | Imagen nítida; completar `expected_model` / `expected_serial` en `config/cameras/basler_scA1390.toml` con lo que informa la cámara |
| 2. Imprimir el tablero | `gt1_tools/calibrate.py` (opción 0) | Tablero ChArUco en A3 |
| 3. Calibrar la lente | `gt1_tools/calibrate.py` (opción 1) | `calibration/basler_scA1390/intrinsics.json` |
| 4. Calibrar cámara-robot (con el pincho) | `gt1_tools/calibrate.py` (opción 2) | `camera_model.json` y la altura de la mesa para `cell.toml` |
| 5. Definir las zonas | editar `config/cell.toml` | Zonas de toma, bandeja y volteo dentro de la vista |
| 6. Juntar muestras de piel y hueso | `gt1_tools/build_face_references.py` | `references/basler_scA1390/face_histogram.npz` |
| 7. Medir el acierto de cara | `gt1_test/face_test.py` | Registro en `runs/face_test/` |
| 8. Probar el gripper | `gt1_test/gripper_test.py` | TCP y carga cargados en el robot; acciones verificadas |

Detalle de cada prueba en [gt1_test/README.md](gt1_test/README.md) y de cada herramienta en [gt1_tools/README.md](gt1_tools/README.md).

### Valores por medir

Los archivos de configuración traen valores iniciales marcados **POR MEDIR** o **POR COMPLETAR**:

- `config/grippers/soft_2dedos.toml`: TCP (punta de los dedos cerrados), masa y centro de gravedad. Se cargan en el robot y afectan la detección de colisiones.
- `config/grippers/pincho.toml`: largo del pincho de calibración.
- `config/cameras/basler_scA1390.toml`: modelo y número de serie (el programa muestra los reales si no coinciden).
- `config/cell.toml`: altura de la mesa (la da la calibración) y zonas.
- `config/trays/bandeja_pet.toml`: medidas de la bandeja (pedir la ficha a Agrosuper).

## Configuración (`config/`)

Un archivo por componente; cada programa elige los suyos por nombre.

| Carpeta / archivo | Contenido |
|---|---|
| `station.toml` | Hardware montado según la última configuración confirmada. Lo actualizan los programas |
| `cell.toml` | Disposición del banco: altura de la mesa y zonas (toma, bandeja, volteo) en mm de la base |
| `robots/` | `UR10e.toml` (RTDE), `sim.toml` (simulado) |
| `cameras/` | `basler_scA1390.toml` (MVS), `webcam_color.toml`, `folder_basler.toml` (imágenes guardadas) |
| `grippers/` | TCP, carga y acciones → salidas digitales: `soft_2dedos.toml`, `pincho.toml` |
| `trays/` | Medidas de cada bandeja |
| `perception/` | `piece_background.toml` (detector de piezas), `face_histogram.toml` (clasificador de cara) |
| `boards/` | Tablero de calibración |
| `strategies/` | Estrategias de toma, volteo y colocación (próxima iteración) |
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

`runs/`, `captures/` y las muestras de `references/` no se versionan (`.gitignore`): se respaldan aparte.

## Mapa del código

```
src/gt1/                       biblioteca (las capas y sus reglas están en CLAUDE.md)
  geometry.py                  rotaciones, transformaciones rígidas, ajuste de Kabsch
  paths.py                     carpetas del proyecto
  domain/                      tipos del problema, sin hardware
    pose.py                    Pose: posición [mm] + rotación
    frame.py                   Frame: una imagen y su hora de captura
    pieces.py                  Piece, PieceStatus, Face (piel/hueso/incierto), FaceResult
    tool.py                    salidas digitales y ToolSettings (TCP + carga)
    tray.py, zone.py           bandeja detectada, zona de la celda
    plan.py                    Plan y sus pasos (MoveLinear, MoveJoints, GripperCommand, Wait)
  config/
    schemas.py                 modelo pydantic de cada archivo de config/
    loading.py                 ConfigRepository: lectura y validación estricta
    station.py                 diferencias con station.toml
  hardware/
    camera/                    Camera; mvs.py (Basler GigE), webcam.py, folder.py
    robot/                     Robot; ur_rtde.py (UR10e), sim.py
    gripper/                   Gripper; digital_outputs.py, fixed.py (pincho), tool_settings.py
  calibration/
    board.py                   tablero ChArUco: detección, esquinas a tocar, imagen para imprimir
    intrinsics.py              modelo de la lente
    extrinsics.py              tablero en la base a partir de toques; ajuste de los dos métodos
    projection.py              PixelProjector: HomographyProjector, CameraModelProjector
    store.py                   archivos de calibration/<camera_id>/
  perception/
    image_ops.py               máscaras, zona en la imagen, escala de grises
    pieces/                    PieceDetector; background.py (fondo de alto contraste)
    face/                      FaceClassifier; histogram.py, references.py (muestras y referencias)
    tray/                      TrayDetector (interfaz)
    verify.py                  PlacementVerifier (interfaz)
  planning/                    PickStrategy, FlipStrategy, PlaceStrategy (interfaces)
  runtime/
    program.py                 run_program: argumentos, logging, config, registro, errores
    station.py                 confirmación del hardware al iniciar
    factories.py               construye cada componente según su kind
    console.py                 OperatorConsole: ventana, teclas, ayuda, barra de estado
    runner.py                  hilo de trabajo con señal de parada
    overlay.py                 dibujo de zonas, piezas y paneles de texto
    run_log.py                 registro de la corrida
    terminal.py                preguntas por la terminal
gt1_test/                      pruebas de validación (ver su README)
gt1_tools/                     herramientas (ver su README)
tests/                         pytest sin hardware; test_architecture.py verifica las capas
```

## Estado

**Hecho:** biblioteca completa para percepción y arranque; `face_test`, `gripper_test`, `camera_viewer`, `calibrate`, `build_face_references`; 115 tests sin hardware; prueba de humo de todos los programas con cámara de carpeta y robot simulado, incluida la calibración completa con un tablero sintético.

**Próximas iteraciones:** ejecutor de planes y límites de la celda → `tray_test` (detector de bandeja) → `pick_place_test` (estrategias de toma y colocación, verificador) → `flip_test` (estación de volteo) → `demo` → `teach_cell`, `summarize_run`. Después: un clasificador YOLO como alternativa, URSim y un gemelo digital como implementaciones de `Robot`, y la cinta en movimiento.
