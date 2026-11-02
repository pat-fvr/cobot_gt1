# Pruebas de validación (`gt1_test/`)

Cada prueba valida un concepto del piloto con el hardware del banco y deja un registro para medirlo. Se ejecuta desde la carpeta del proyecto:

```
uv run python gt1_test/<prueba>.py                      # usa config/tests/<prueba>.toml
uv run python gt1_test/<prueba>.py --config otra.toml   # otra configuración (p. ej. con la webcam)
```

Al iniciar, cada prueba compara su hardware con `config/station.toml`. Si hay diferencias, las muestra y pide confirmación por la terminal antes de seguir (y actualiza `station.toml`). Las teclas comunes en todas las ventanas son `x` (detener), `h` (ayuda) y `q` (salir). El registro de cada corrida queda en `runs/<prueba>/<fecha_hora>/` (ver el [README principal](../README.md#registros-de-cada-corrida)).

| Prueba | Valida | Estado |
|---|---|---|
| [face_test](#face_test) | La cámara distingue piel de hueso | Lista |
| [gripper_test](#gripper_test) | El gripper responde a cada acción de su archivo | Lista |
| tray_test | La cámara encuentra la bandeja | Próxima iteración |
| pick_place_test | El robot toma una pieza con la piel arriba y la deja en la bandeja | Próxima iteración |
| flip_test | El robot voltea una pieza con el hueso arriba (estación de volteo) | Próxima iteración |
| demo | Rutina completa: detectar → clasificar → tomar → voltear si hace falta → dejar → verificar | Próxima iteración |

---

## face_test

**Valida:** que la cámara reconozca la cara de cada pieza (criterio propuesto R2: ≥ 99,5 % de acierto).

**Requiere:** la calibración de la cámara (`gt1_tools/calibrate.py`) y, con el clasificador por histograma, las referencias de cara (`gt1_tools/build_face_references.py`). El clasificador por textura (`face_texture`) no usa referencias.

**Protocolo:**

1. Dejar varias pechugas en la zona de toma, separadas y todas con la **misma** cara hacia arriba.
2. Apretar `1` si están con la piel arriba o `2` si están con el hueso arriba: se registra la captura con esa etiqueta.
3. Mover las piezas (posición, giro), repetir. Darlas vuelta y repetir con la otra tecla.

| Tecla | Acción |
|---|---|
| `c` | Ver la clasificación sin registrar |
| `1` | Registrar: todas con la PIEL arriba |
| `2` | Registrar: todas con el HUESO arriba |
| `espacio` | Registrar sin etiqueta |

En pantalla, cada pieza muestra su cara y su margen (verde = piel, naranjo = hueso, amarillo = incierto). El contorno gris es una pieza no tomable: las que tocan el borde de la zona muestran igual su cara (y quedan en el registro), pero no cuentan para el acierto; los grupos de piezas no se clasifican. La barra de estado acumula el acierto sobre las piezas tomables.

**Registro:** por captura, la etiqueta, el tiempo de percepción y, por pieza: estado, centro, ángulo, largo, ancho, área, cara detectada, margen, lo que midió el clasificador (`measurements`: distancias a las referencias o textura en %/mm) y si acertó.

**Configuración:** `config/tests/face_test.toml` (Basler, clasificador por histograma) y `config/tests/face_test_c270.toml` (webcam C270 sobre fondo turquesa, clasificador por textura).

---

## gripper_test

**Valida:** el cableado y el archivo de un gripper: cada acción de `config/grippers/<gripper>.toml` mueve los dedos como se espera.

**Protocolo:** con el robot quieto, apretar el número de cada acción y observar el gripper. La pantalla lista las acciones y el estado de las salidas digitales que usa.

| Tecla | Acción |
|---|---|
| `1`..`9` | Ejecutar la acción de ese número (en el orden del archivo) |

Al iniciar se cargan en el robot el TCP y la carga del gripper y se verifican leyéndolos de vuelta: es también la forma rápida de cargar un gripper recién montado.

**Registro:** por acción, su nombre, la duración y el estado de las salidas al terminar.

**Configuración:** `config/tests/gripper_test.toml` (robot y gripper). Con `robot = "sim"` funciona sin robot.
