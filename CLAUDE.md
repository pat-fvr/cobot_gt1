# cobot_gt1

Piloto de carga robotizada de pechuga entera en bandeja (GT1, proyecto 171, Agrosuper): UR10e + soft gripper + cámara fija. Valida que el robot detecte la pieza, reconozca la cara (piel/hueso), la voltee si viene con el hueso arriba y la deje en la bandeja con la piel arriba.

- Mapa de carpetas y módulos, instalación, flujo de trabajo en el banco: [README.md](README.md)
- Qué valida cada prueba y su protocolo: [gt1_test/README.md](gt1_test/README.md)
- Herramientas (visor, calibración, referencias): [gt1_tools/README.md](gt1_tools/README.md)
- Vocabulario del dominio (pieza, cara, zona, estación…): [GLOSSARY.md](GLOSSARY.md)

El código es la referencia de calidad del equipo: **muy limpio y comprensible** para alguien que lo lee por primera vez. Cada cambio lo deja igual o mejor.

## 1. Capas

Biblioteca en `src/gt1/`, programas en `gt1_test/` (pruebas de validación con hardware) y `gt1_tools/` (herramientas de trabajo). Cada capa importa solo de las de abajo:

```
gt1_test/, gt1_tools/        programas: cada uno importa solo de gt1
gt1.runtime                  arma componentes, consola, registro, arranque
gt1.hardware  gt1.calibration  gt1.perception  gt1.planning
gt1.config                   esquemas y carga de config/
gt1.domain                   tipos del problema
gt1.geometry, gt1.paths      matemática pura, carpetas
```

- Un programa es autónomo: importa solo de `gt1`, nunca de otro programa. Lo que dos programas comparten se mueve a la biblioteca.
- `gt1.runtime.factories` es el único lugar que elige implementaciones (`match` sobre `kind`).
- `tests/test_architecture.py` verifica estas reglas; al agregar una capa o un paquete de programas, se actualiza ahí.

## 2. Interfaces e implementaciones

Todo lo intercambiable es una interfaz (ABC) con implementaciones en archivos separados: `Camera`, `Robot`, `Gripper`, `PixelProjector`, `PieceDetector`, `FaceClassifier`, `TrayDetector`, `PlacementVerifier`, `PickStrategy`, `FlipStrategy`, `PlaceStrategy`.

Agregar una implementación (p. ej. una cámara nueva):

1. Clase en su propio archivo junto a la interfaz (`hardware/camera/pylon.py`), que recibe su modelo de configuración en el constructor.
2. Modelo de configuración en `config/schemas.py` con `kind: Literal["..."]`, agregado a la unión discriminada (`CameraConfig`). Una interfaz con una sola implementación usa el modelo directo; con la segunda pasa a ser unión.
3. Un caso en `runtime/factories.py`.
4. Un archivo de ejemplo en `config/<carpeta>/`.
5. Tests sin hardware de su comportamiento.

Listo cuando: pyright no informa casos faltantes en el `match`, el archivo de ejemplo carga en `tests/test_config.py` y el README de la carpeta la menciona.

**Estrategias puras.** Las estrategias de `gt1.planning` reciben lo que vio la cámara y devuelven un `Plan` (`domain/plan.py`): sin cámara, robot, archivos ni estado mutable, deterministas, probadas con pytest. El ejecutor es quien toca el robot.

## 3. Agregar un programa

1. `gt1_test/<nombre>.py` o `gt1_tools/<nombre>.py`, con el mismo esqueleto que los existentes: docstring con propósito, protocolo y teclas; `Components` y `Settings` (pydantic, `StrictModel`); una función cuerpo que recibe `Program[...]`; `main()` que llama a `run_program`.
2. `config/tests/<nombre>.toml` o `config/tools/<nombre>.toml` con la selección explícita de componentes y los ajustes propios.
3. El cuerpo construye los componentes con `factories`, abre el hardware con `with`, llama a `program.confirm_station(...)` antes de usarlo y entrega la interacción a `OperatorConsole`.
4. Cada intento se registra con `program.run_log.record(datos, imágenes)`: los datos son lo que hace falta para medir el criterio de la prueba.
5. Entrada en `PROGRAM_SCHEMAS` de `tests/test_config.py` y en el README de su carpeta.

## 4. Configuración

- Todo valor que ajusta el proceso (IPs, velocidades, umbrales, tiempos, zonas, rutas) vive en `config/` y en ningún otro lugar. El código no tiene valores por defecto para ellos: ni defaults de argparse, ni de pydantic, ni constantes de ajuste.
- Constantes con nombre en el módulo son válidas solo para detalles técnicos o de presentación que nadie ajusta en el banco (tamaño de la barra de estado, margen de recorte), con nombre en MAYÚSCULAS y comentario.
- Un archivo por componente, nombrado por su archivo: `camera = "basler_scA1390"` → `config/cameras/basler_scA1390.toml`. Los programas eligen componentes en `config/tests/` o `config/tools/`; la disposición del banco va en `config/cell.toml`.
- Cada campo nuevo: en el modelo de `config/schemas.py` con `Field(description=...)`, y con valor en cada archivo que use ese modelo. La validación es estricta (campo de más, de menos o tipo equivocado detienen el programa).
- `station.toml` describe el hardware real montado y lo escriben los programas al confirmar un cambio; lo simulado no se anota ahí.
- La calibración y las referencias de cara son **por cámara** (`calibration/<camera_id>/`, `references/<camera_id>/`).

## 5. Robot y seguridad

- Solo `hardware/robot/ur_rtde.py` habla con ur_rtde y solo `hardware/camera/mvs.py` con el SDK MVS (son los únicos archivos con `# pyright: basic`).
- **Un hilo a la vez**: el hardware se usa desde el hilo de trabajo de `OperatorConsole` (o desde el principal mientras no hay trabajo en curso). Para detener, otro hilo activa la señal `stop`; los movimientos la revisan y frenan solos. La tecla `x` responde siempre.
- Lo lento (capturar, mover, calcular) va en un trabajo (`bind_job`); `bind_action` es solo para lo instantáneo.
- Antes de usar el robot: `confirm_station` carga el TCP y la carga del gripper desde su archivo y el robot los lee de vuelta. El archivo del gripper es la única fuente del TCP y del payload.
- Agarrar y soltar con el robot quieto. Ante un error del robot: detener, dejar el ciclo en un estado seguro, mostrar el motivo y registrar; el paso siguiente arranca solo si el anterior terminó.

## 6. Percepción

- Se decide con imágenes capturadas con el robot quieto y fuera de la vista, tomadas después de que llegó (`Frame.captured_at_s`).
- El resultado se verifica con la cámara en lugar de suponerlo; los reintentos tienen un máximo de la configuración.
- Una pieza dudosa no se toma: `Face.UNCERTAIN` y `PieceStatus` distinto de `OK` se registran y se saltan.
- Umbrales con histéresis entre lo que dispara una acción y su objetivo, para que el error de medición no la repita.
- Los detectores miden en mm de la base (proyectan el contorno antes de medir): los umbrales no dependen de la cámara.
- Todo programa con cámara funciona con `FolderCamera` (imágenes guardadas), para probar sin hardware.

## 7. Unidades y sistemas de referencia

- mm, grados y segundos en todo el código; metros y radianes solo dentro de `ur_rtde.py`.
- El nombre dice la unidad: `_mm`, `_deg`, `_s`, `_px`, `_mm2`, `_kg`.
- Las transformaciones se nombran `destino_from_origen` (`base_from_camera` lleva puntos de la cámara a la base). Sin sufijo de sistema, una posición está en la base del robot.
- La geometría reutilizable vive en `gt1/geometry.py`.

## 8. Código limpio

- Funciones cortas con un solo propósito; si una función necesita un comentario para separar partes, son dos funciones.
- Nombres completos que explican sin leer el cuerpo (`piece_histogram`, `touch_rms_mm`).
- Tipos completos: pyright estricto pasa sin errores. Los arreglos usan `FloatArray`, `Image`, `NDArray[np.int32]`.
- Datos inmutables: `@dataclass(frozen=True)` en el dominio, modelos pydantic `frozen`. Estado mutable solo en objetos con ciclo de vida (drivers, sesiones de un programa).
- Dependencias por constructor; sin estado global ni singletons.
- Los errores llevan un mensaje que dice qué pasó y qué hacer (archivo, campo, valor encontrado y esperado). Cada `except` nombra la excepción, actúa sobre ella (registra, informa al operador, deja un estado seguro) o la relanza con contexto (`raise ... from error`).
- Los comentarios explican el *por qué* (una decisión, un gotcha del hardware); el *qué* lo dice el código. Docstring de módulo en cada archivo con su propósito.
- Lo que aparece dos veces va a una función de la biblioteca. El código que no se usa se borra (git lo recuerda).
- `ruff format` define el formato; líneas de 110 caracteres.

## 9. Tests

- Todo lo que no es driver tiene pytest sin hardware: geometría, configuración, calibración (con cámara sintética), percepción (con imágenes sintéticas), estrategias, arranque.
- Para el hardware se usan las implementaciones simuladas (`SimRobot`, `FolderCamera`) o las de `tests/fakes.py`, en lugar de mocks ad hoc.
- Un error encontrado en el banco o en una prueba de humo entra como test antes de corregirlo.

## 10. Idioma

Identificadores en inglés. Comentarios, docstrings, mensajes, textos de la consola y documentos en español. La consola convierte el texto a ASCII al dibujarlo (OpenCV no dibuja tildes).

## 11. Antes de terminar un cambio

```
uv run ruff check .
uv run ruff format .
uv run pyright
uv run pytest
```

Listo cuando los cuatro pasan sin errores, y el README de cada carpeta tocada (raíz, `gt1_test/`, `gt1_tools/`) describe lo que cambió.
