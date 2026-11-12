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
| [pick_place_test](#pick_place_test) | El robot toma una pieza con la piel arriba y la deja en la bandeja fija | Lista sin hardware; por probar en el banco |
| [tray_loading_test](#tray_loading_test) | El robot carga las pechugas con la piel arriba en las bandejas que detecta la cámara, con los lóbulos hacia donde se pide | Lista sin hardware; por probar en el banco |
| [demo_static_tray](#demo_static_tray) | El robot carga una tras otra las pechugas que ve la cámara en la bandeja fija con tope, hasta que no queda ninguna | Lista sin hardware; por probar en el banco |
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

---

## pick_place_test

**Valida:** que el robot tome una pechuga con la piel arriba de la mitad superior de la escena (zona `pick`) y la deje en la bandeja fija de la mitad inferior, y que la cámara la vea después en la bandeja con la piel arriba.

**Requiere:**
- La calibración cámara-robot de la cámara.
- El gripper montado y probado con [gripper_test](#gripper_test).
- La pose `home` en `config/robot_poses.toml` (fuera de la vista), enseñada con `gt1_tools/freedrive.py` (tecla `1`). Si no está, la prueba no arranca y dice cómo enseñarla.
- En `config/cell.toml`, el centro y ángulo de la bandeja (`tray_placement`), leídos con freedrive (`t` registra la pose del TCP). Con los valores de partida, **no correr esta prueba**: el robot dejaría la pieza en un lugar que nadie revisó.
- El robot en modo remoto.

**Cómo se mueve:**
- Antes de cada captura, el robot va a la pose fuera de la vista (`pose = "home"` en `[settings.clear_view]`), por articulaciones, para que ni él ni su sombra tapen las piezas. Si la punta está más baja que `safe_height_mm` (`[settings.clear_view]`), primero sube en vertical.
- Toma (`config/strategies/pick_top_down.toml`): abre el gripper, va en vertical, desde donde esté, a la altura de llegada (`approach_height_mm`, 90 mm sobre la mesa), viaja a esa altura hasta quedar sobre el centro de la pieza con el eje Y del TCP a lo largo de su línea central, baja hasta tocar la mesa, cierra, baja `after_grip_descent_mm` más (los dedos del soft gripper se recogen al cerrar; así quedan bajo la pieza) y sube. Se detiene sobre la pieza antes de bajar y para cerrar; solo pasa en curva por el punto de arriba de la subida (`blend_radius_mm`, hoy 30 mm): empieza a avanzar 30 mm antes de terminar de subir. Velocidades (2026-11-10): 1500 mm/s de viaje, 500 mm/s al bajar y subir, con aceleración de 8000 mm/s² (`config/robots/UR10e.toml`); si al bajar cerrado vuelve la parada C153A1, bajar primero `descend_speed_mm_s`. Con un gripper asimétrico (`wide_side` en `config/grippers/`), la parte ancha va hacia los lóbulos y la angosta hacia la punta; sin los lóbulos medidos, solo se alinea el eje.
- Colocación (`config/strategies/place_top_down.toml`): llega en línea recta sobre el centro de la bandeja, a `release_height_mm` sobre el fondo (hoy 90 mm, 10 mm sobre el borde), con el eje Y del TCP a lo largo del lado largo de la bandeja. La pieza va fija en el gripper: la parte ancha sigue hacia los lóbulos. Si el detector midió hacia dónde están los lóbulos (`piece_skin_outline`), gira la pieza para que queden hacia `lobes_toward_deg`; si no (`piece_color_background`), elige el sentido que menos gira la muñeca. Llega a 1500 mm/s y abre ahí detenido, sin bajar ni subir; espera el `settle_s` de la acción `release` del gripper (0,2 s en `soft_x_asimetrico`) y sigue. Abrir antes de la toma siguiente no espera de nuevo: las salidas ya están así. Como llega desde el costado, `release_height_mm` no puede ser menor que la profundidad de la bandeja.
- Tomar y dejar son un solo plan: la subida pasa en curva al viaje a la bandeja. La salida de la vista va a 400 mm/s en vertical y a 120°/s por articulaciones (el máximo de la base y el hombro del UR10e), con aceleración de 480°/s².
- Las alturas se miden desde la mesa (`table_z_mm`) y desde el fondo de la bandeja.

**Protocolo:**

1. Bandeja vacía en su lugar y pechugas con la piel arriba en la zona de toma.
2. `c` muestra la pieza elegida (la tomable con la piel arriba de cara más segura), con el eje Y del gripper en amarillo (una barra marca la parte ancha) y el borde de la bandeja en celeste. El robot solo va a la pose fuera de la vista.
3. `p` hace un intento: toma, deja en la bandeja, sale de la vista y verifica con la cámara.
4. Sacar la pieza de la bandeja, devolverla a la zona de toma en otra posición o ángulo, y repetir.

| Tecla | Acción |
|---|---|
| `c` | Ver la pieza elegida (el robot va a la pose fuera de la vista) |
| `p` | Intento: tomar, dejar en la bandeja y verificar |
| `a` | Abrir el gripper donde esté el robot (si un intento se detuvo con la pieza tomada) |

Las piezas con el hueso arriba, inciertas, en el borde de la zona o en grupo no se toman; se cuentan en el registro. Si el robot informa un error o se detiene con `x`, el intento se corta ahí con el robot detenido y el gripper como estaba (si tenía la pieza, la sigue teniendo). La barra de estado dice en qué paso se cortó.

**Registro:** por intento:
- El resultado:
  - `colocada`: una pieza en la zona de la bandeja, entera y con la piel arriba.
  - `mal_colocada`: después de soltar, la cámara no ve eso en la bandeja.
  - `sin_pieza`: no había pieza tomable con la piel arriba.
  - `sin_verificar`: la pieza se dejó, pero no se pudo capturar después.
  - `detenido` / `fallo`: el intento se cortó; queda en qué paso.
- La pieza elegida: centro, ángulo, medidas, cara y margen.
- Las piezas que no se tomaron y por qué.
- El plan (tomar y dejar, `execution`) con sus pasos y hasta dónde llegó.
- Las piezas que vio la cámara en la bandeja.
- Imágenes: la captura, su superposición, y la captura y superposición de después.

**Configuración:** `config/tests/pick_place_test.toml`.

---

## tray_loading_test

**Valida:** que la cámara encuentre las pechugas con la piel arriba en la mitad superior de la escena (zona `pick`) y las bandejas en la mitad inferior (zona `tray`), sin escena vacía, y que el robot cargue cada bandeja con una pechuga, con su línea central a lo largo del lado largo de la bandeja y los lóbulos hacia `lobes_toward_deg` de `config/strategies/place_top_down.toml`.

**Requiere:** lo mismo que [pick_place_test](#pick_place_test), salvo `tray_placement`. Además:
- Medir con regla el fondo de la bandeja (`config/trays/bandeja_pet.toml`: hoy es medida de la imagen) y comparar el centro de una bandeja que da `c` con la punta del gripper en su centro (freedrive): en las fotos el detector da los centros hasta ~12 mm corridos hacia +Y de la base.
- Confirmar con freedrive la orientación del TCP de `soft_x_asimetrico` (parte ancha hacia -Y).
- Definir con Agrosuper hacia dónde van los lóbulos en la bandeja (`lobes_toward_deg`, hoy -160°: hacia arriba en la imagen, como llegan las piezas en las fotos).

**Detección:**
- Pechugas (`config/perception/piece_skin_outline.toml`): por el contorno, como un salto de brillo relativo (en % por mm), así que vale sobre la mesa oscura, sobre cartón blanco y en la sombra. Una candidata se acepta si mide como una pechuga (área, largo, ancho, convexa) y su superficie es lisa como la piel: el lado del hueso no se detecta. Antes de medir se le quita la sombra (lo más oscuro que la mesa alrededor, si la pieza es más clara): con la acA1600 agrandaba la punta hasta invertir los lóbulos. La línea central es el eje de simetría del contorno y los lóbulos están en la mitad con más superficie (flecha magenta): en una pieza casi ovalada (mitades con 6-8 % de diferencia) el sentido puede salir al revés, y la verificación después de cargar lo detecta (`mal_orientada`). En las fotos: 4 de 4 en `ref_piel`, 2 de 2 con la piel en `escena_4_pollos`, 0 en `ref_hueso`; con la acA1600, 4 de 4 en `escena_4_bandejas_4_pollos` y 3 de 3 sobre film azul (`escena_film_azul`). Tarda ~0,1 s con la acA1600: los contornos se buscan en la imagen reducida a la mitad (`search_scale`); la textura y las medidas, en la completa.
- Bandejas (`config/perception/tray_rim_and_floor.toml`): la bandeja transparente se ve por dos líneas oscuras, el labio del borde (a `depth_mm` sobre la mesa) y el pliegue donde las paredes tocan el fondo. Cuál se ve depende de la luz: el pliegue, en los lados cuya pared hace sombra sobre él (`shadow_toward_deg`; con la acA1600, abajo y a la derecha en la imagen), y el labio en los otros. Cada lado se busca solo con la suya: si cada lado eligiera la mejor de las dos, una misma línea calzaría en dos posiciones y el centro se corría ~3 cm. Se buscan los dos rectángulos de `config/trays/bandeja_pet.toml`, concéntricos, en vistas cenitales a cada altura. Una pose es bandeja si se ven sus cuatro lados, el pliegue más fuerte pasa `min_fold_contrast_pct` y el mejor lado largo y el mejor corto pasan `min_line_contrast_pct` (dos lados paralelos no fijan dónde está); y si lo que encierra parece PET vacío: paredes ni tan oscuras como una caja o una sombra, interior ni tan claro como una pieza. Con la acA1600 (2026-11-10): las 4 bandejas de `escena_4_bandejas_4_pollos` y las 2 de `escena_bandejas_vacias`, a ≤ 13 mm y ≤ 7° de lo medido a mano (la de más abajo, contra su pliegue medido en la imagen: a mano se había anotado 7° de menos), y las 3 sobre film azul (`escena_film_azul`); ninguna en las fotos de la scA1390. Tarda ~0,15-0,2 s (antes ~9 s): la búsqueda va de lo grueso a lo fino, todos los ángulos cada `coarse_angle_step_deg` en una grilla de `coarse_grid_mm` y cada pose que resulta afinada a `grid_mm` y `angle_step_deg`. Una bandeja que sale de la imagen no se detecta, y puede aparecer una girada 90° y corrida en su lugar: dejar las bandejas enteras dentro de la imagen. Una bandeja con una pieza adentro puede no detectarse (la pieza tapa el fondo); igual no se ofrece como libre. **No se detectan** (`escena_3_bandejas_giradas`, 2026-11-10): una bandeja girada unos 40° en la imagen (lado largo a ~-20° o ~70° en la base), porque dos de sus lados quedan paralelos a la sombra y no muestran ni pliegue ni labio; y una cruzada por la franja de luz y sombra de abajo a la izquierda, cuyas paredes se ven tan oscuras como lo que no es bandeja. Dejar las bandejas derechas (lado largo vertical en la imagen, ±25°) y fuera de esa franja, o emparejar la luz.

**Cómo se mueve:** toma y colocación como en [pick_place_test](#pick_place_test). `c` decide todo con una sola captura desde la pose fuera de la vista y `p` ejecuta lo que decidió, de inmediato y sin otra captura. Las pechugas tomables y las bandejas libres se ordenan de izquierda a derecha en la imagen (`load_order_toward_deg`: 110,5° en la base con la calibración de la acA1600) y se cargan en ese orden: la primera pechuga en la primera bandeja, la segunda en la segunda. Si hay más pechugas que bandejas libres, las de la derecha se quedan. Una bandeja está libre si no tiene una pieza adentro. Con todas las cargas planificadas, el robot las hace seguidas: después de soltar una pieza va directo a la siguiente, sin pasar por la pose fuera de la vista, y vuelve a ella al terminar. Alturas de la punta sobre la mesa: va en vertical a 90 mm, viaja a 90 mm hasta quedar sobre la pieza, baja a 0, cierra, baja a -40 (los dedos se recogen al cerrar), sube a 90, va a la bandeja a 90 y suelta. Tomar y dejar son un solo plan por carga: la subida pasa en curva al viaje a la bandeja, así que la pieza empieza a avanzar a unos 60 mm sobre la mesa, todavía sobre la zona de toma. Como no vuelve a mirar entre cargas, las pechugas tienen que estar separadas: si al tomar una se mueve otra, el robot la busca donde estaba.

**Protocolo:**

1. Bandejas vacías en la mitad inferior de la mesa; pechugas con la piel arriba, separadas, en la superior.
2. `c` va a la pose fuera de la vista, captura y calcula las cargas: muestra las piezas con su línea central, las bandejas con su contraste y el sentido pedido para los lóbulos, y qué pieza va a cuál (número y línea blanca, en el orden de carga). No mover nada en la mesa hasta apretar `p`.
3. `p` ejecuta las cargas que calculó `c`, seguidas, y vuelve fuera de la vista. Lo calculado se usa una sola vez: sin una `c` antes, `p` no hace nada. Con `verify_loads = true` (`[settings]`; hoy `false`), captura desde ahí y verifica cada carga. Si el robot no completa una carga, el ciclo se corta ahí con el robot detenido donde paró.
4. Vaciar las bandejas, devolver las piezas en otras posiciones y ángulos, y repetir.

| Tecla | Acción |
|---|---|
| `c` | Capturar y calcular las cargas (el robot va a la pose fuera de la vista) |
| `p` | Ejecutar las cargas calculadas con `c` (y la verificación, si está activada) |
| `a` | Abrir el gripper donde esté el robot (si una carga se detuvo con la pieza tomada) |

**Registro:** por ciclo, el tiempo de percepción, cuántas piezas tomables y bandejas (libres y detectadas) vio, si se verificó, y por carga (`loads`):
- El resultado:
  - `dejada`: la pieza se soltó en la bandeja, sin verificar (`verify_loads = false`).
  - `cargada`: una pechuga con la piel arriba dentro de la bandeja, con los lóbulos a menos de `max_lobes_error_deg` de lo pedido.
  - `mal_orientada`: la pieza quedó en la bandeja con los lóbulos hacia otro lado.
  - `sin_pieza_en_bandeja`: después de soltar, la cámara no ve una pechuga con la piel arriba en la bandeja (o ve más de una).
  - `sin_verificar`: la pieza se dejó, pero no se pudo capturar después (o una carga siguiente se cortó).
  - `detenido` / `fallo`: la carga se cortó; queda en qué paso. Las siguientes no se hacen.
- La pieza (centro, eje, sentido de los lóbulos, medidas), la bandeja (centro, ángulo, contraste) y el sentido pedido para los lóbulos.
- El plan (tomar y dejar, `execution`) con sus pasos y hasta dónde llegó, y lo que vio la cámara dentro de la bandeja.
- Imágenes: la captura y su superposición con las asignaciones; con verificación, también la captura y la superposición de después.

**Configuración:** `config/tests/tray_loading_test.toml`.

---

## demo_static_tray

**Valida:** el ciclo continuo con la bandeja fija: el robot toma las pechugas de la zona de toma (`pick_zone = "pick"`: la plataforma azul; deja fuera la cinta, por donde sale la bandeja cargada mientras el robot captura) una tras otra, de izquierda a derecha en la imagen, y suelta cada una en una bandeja distinta que llega por la cinta y detiene un tope, hasta que no queda ninguna. No usa el detector de bandejas: la bandeja está siempre en el mismo lugar.

**Requiere:**
- La calibración cámara-robot de la cámara y el gripper probado con [gripper_test](#gripper_test).
- La pose `tray_drop` en `config/robot_poses.toml`: la punta sobre la bandeja detenida por el tope, a la altura de soltar, **fuera de la vista de la cámara** (desde ahí se captura). Se enseña con `gt1_tools/freedrive.py` (tecla `2`). Si no está, la prueba no arranca y dice cómo enseñarla.
- El tope en DO2 (1 cerrado, 0 abierto) y el sensor de bandeja en DI1 (1 con una bandeja en el tope), en `[settings.tray_station]`. Al iniciar, el programa cierra el tope y verifica que la salida quedó en 1; si no, no arranca.
- El robot en modo remoto.

**Cómo se mueve:** cada carga es:
1. Captura desde `tray_drop`, sin mover el robot (la primera vez lo lleva ahí: si la punta está más baja que `safe_height_mm`, sube antes en vertical). Elige la pechuga tomable de más a la izquierda en la imagen (`pick_from = "right"`: la de más a la derecha). Izquierda y derecha se miden a lo largo de `left_to_right_deg` en la base (110,5° con la calibración de la acA1600, como `load_order_toward_deg` de [tray_loading_test](#tray_loading_test)).
2. La toma como en [pick_place_test](#pick_place_test), configurada en `[settings.pick]` del propio `config/tests/demo_static_tray.toml` (alturas, velocidades y cuánto baja después de cerrar, `after_grip_descent_mm`): sube con la pieza solo hasta 20 mm sobre la plataforma (`lift_height_mm`), y de ahí va por articulaciones a `tray_drop` (`safe_height_mm` también es 20 mm: no vuelve a subir).
3. Espera que el tope esté cerrado (que salió la bandeja anterior: si no, el sensor podría estar viendo todavía la bandeja cargada), espera a que el sensor vea la bandeja y suelta la pieza ahí, sin bajar.
4. Abre el tope para que salga la bandeja cargada y vuelve a 1 **sin esperarla**: el tope se cierra en segundo plano (`gt1/runtime/tray_stopper.py`) `stopper_close_delay_s` (2,5 s) después de que el sensor (DI1) deja de verla, mientras el robot captura y va por la pieza siguiente. La foto se toma apenas suelta: el robot ya estaba quieto en `tray_drop`, así que solo espera lo que falte de `capture_settle_s`. `x` no corta el cierre del tope (cerrarlo es el estado seguro); al terminar el programa con la bandeja todavía en el sensor, el tope queda abierto y se cierra al volver a iniciar.

Y vuelve a 1 hasta que la foto no muestra pechugas tomables. Detecta las pechugas con `piece_skin_outline`: solo las que tienen la piel arriba; las que están en el borde de la zona o en grupo no se toman. Como cada carga empieza con una foto nueva, una pieza que se movió al tomar otra se busca donde quedó.

**Protocolo:**

1. Pechugas con la piel arriba, separadas, sobre la plataforma (la zona de toma); la cinta con bandejas vacías.
2. `c` va a `tray_drop`, captura y muestra la pieza que se tomaría primero, con el eje Y del gripper, sin tomarla.
3. `p` carga hasta que no quedan pechugas. `x` detiene en cualquier momento, también mientras espera la bandeja: el robot queda donde paró y, si tenía la pieza, la sigue teniendo (`a` la suelta).

| Tecla | Acción |
|---|---|
| `c` | Ver la pieza que se toma primero (el robot va a la pose de soltar) |
| `p` | Cargar las pechugas una tras otra hasta que no quede ninguna |
| `a` | Abrir el gripper donde esté el robot (si una carga se detuvo con la pieza tomada) |

La espera de la bandeja no tiene tiempo máximo: la barra de estado dice qué entrada se espera. Con `robot = "sim"` el sensor nunca se enciende: el robot simulado queda esperando la bandeja con la pieza tomada.

**Registro:** por carga:
- El resultado: `cargada` (la pieza se soltó y la bandeja salió), `detenido` o `fallo` (la carga se cortó; queda en qué paso). La foto final sin piezas no se registra: la barra de estado dice cuántas se cargaron.
- El tiempo de percepción, cuántas piezas tomables había, `pick_from` y la pieza elegida (centro, eje, sentido de los lóbulos, medidas).
- Los planes de tomar (`pick`) y de llevar y soltar (`drop`: ir a `tray_drop`, tope, espera de la bandeja, soltar, abrir y cerrar el tope) con sus pasos y hasta dónde llegaron.
- `cycle_s`: desde la foto hasta cerrar el tope detrás de la bandeja cargada.
- Imágenes: la captura y su superposición.

**Configuración:** `config/tests/demo_static_tray.toml`.
