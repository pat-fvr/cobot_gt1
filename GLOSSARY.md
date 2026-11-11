# GT1: carga robotizada de pechuga en bandeja

Vocabulario del piloto GT1: un robot toma pechugas enteras de la zona de toma, las voltea si vienen con el hueso arriba y las deja en la bandeja con la piel arriba. Entre paréntesis, la palabra que usa el código (los identificadores están en inglés).

## El producto y la bandeja

**Pieza** (*piece*):
Una pechuga entera con hueso, tal como la ve la cámara o la toma el robot.
_Evitar_: producto, pollo, objeto, pechuga (en el código)

**Cara** (*face*):
El lado de la pieza que queda hacia arriba: piel, hueso o incierta.
_Evitar_: lado, orientación

**Piel** (*skin*):
La cara exterior, lisa y clara. Es la que debe quedar hacia arriba en la bandeja.

**Hueso** (*bone*):
La cara interior, con costillas, cartílago y pliegues. Una pieza con el hueso arriba hay que voltearla.
_Evitar_: costilla, reverso

**Cara incierta** (*uncertain*):
El resultado cuando no se distingue con seguridad la piel del hueso. Una pieza con la cara incierta no se toma.
_Evitar_: desconocida, dudosa

**Bandeja** (*tray*):
El envase PET transparente donde termina la pieza con la piel arriba.
_Evitar_: contenedor, caja, envase

**Línea central** (*midline*):
La línea que separa los dos lóbulos de la pechuga entera, vista desde arriba: su eje de simetría. En la bandeja queda a lo largo del lado largo.
_Evitar_: eje (salvo el eje mayor, que en una pieza con línea central medida es la misma línea)

**Lóbulos** (*lobes*):
Las dos mitades de la pechuga a cada lado de la línea central; juntas forman el extremo más ancho. El otro extremo es la punta. En la bandeja los lóbulos van hacia un sentido fijo (*lobes_toward_deg*).

**Fondo de la bandeja** (*tray floor*):
La base de la bandeja, apoyada en la mesa. El pliegue oscuro donde las paredes se doblan hacia el fondo es una de las dos líneas que busca el detector de bandejas.

**Borde de la bandeja** (*tray rim*):
El contorno superior de la bandeja, a su alto sobre la mesa; sus medidas son de labio a labio. El labio es la otra línea que busca el detector de bandejas.

**Borde de sellado** (*seal flange*):
El borde superior de la bandeja donde va el film. Tiene que quedar libre de producto.

**Canaleta**:
Por donde llegan las piezas en la línea real. En el banco la reemplaza la zona de toma.

## El banco y la celda

**Banco**:
El puesto de pruebas físico del piloto: robot, gripper, cámara, mesa y bandeja.
_Evitar_: línea, planta

**Celda** (*cell*):
La disposición del banco: la altura de la mesa y las zonas, en la base del robot.
_Evitar_: layout, puesto

**Mesa** (*table*):
La superficie de trabajo donde se apoyan las piezas, el tablero y la bandeja.

**Zona** (*zone*):
Un polígono sobre la mesa donde ocurre una parte del ciclo.
_Evitar_: área, región

**Zona de toma** (*pick zone*):
Donde se dejan las piezas para que el robot las tome.

**Zona de bandeja** (*tray zone*):
Donde están las bandejas.

**Zona de caída del volteo** (*flip landing zone*):
Donde caen las piezas después de la estación de volteo.

**Vista completa** (*full view*):
La zona que cubre toda la imagen de la cámara.

**Estación de volteo** (*flip station*):
El dispositivo fijo de la celda (rampa, caída) que da vuelta la pieza.

**Estación de bandeja** (*tray station*):
El tramo de la cinta de bandejas donde un **tope** (*stopper*, una salida digital: 1 cerrado, 0 abierto) detiene cada bandeja bajo la pose de soltar y un **sensor de bandeja** (*tray sensor*, una entrada digital) dice que llegó. La bandeja está siempre en el mismo lugar: no se detecta con la cámara (`demo_static_tray`).

**Estación** (*station*):
El hardware que está montado en el banco ahora: qué robot, cámara, gripper y bandeja. El operador lo confirma al iniciar un programa.
_Evitar_: setup, montaje

## Robot y herramienta

**Base** (*base*):
El sistema de referencia del robot. Una posición sin otro sistema indicado está en la base.

**Herramienta** (*tool*):
Lo que va montado en la brida del robot: un gripper o el pincho.

**Gripper** (*gripper*):
La herramienta que agarra la pieza; en el piloto, un soft gripper en X asimétrico: su parte ancha va hacia los lóbulos y la angosta hacia la punta (*wide_side*).
_Evitar_: pinza, garra, efector

**Acción del gripper** (*gripper action*):
Un movimiento de los dedos con nombre (agarrar, soltar…) que el gripper ejecuta con salidas digitales del robot.

**Pincho** (*pointer*):
La herramienta con punta fina para tocar las esquinas del tablero al calibrar.
_Evitar_: puntero, aguja

**TCP**:
La punta de la herramienta, respecto de la brida. En el gripper es la punta de los dedos cerrados.

**Carga** (*payload*):
La masa de la herramienta y su centro de gravedad, que el robot necesita para detectar colisiones.

**Fuera de la vista**:
Robot ubicado de modo que no tapa la zona que la cámara está mirando.

**Pose enseñada** (*taught pose*):
Articulaciones del robot guardadas con un nombre en `robot_poses.toml` después de llevarlo a mano con freedrive. **Home** es la pose enseñada fuera de la vista desde la que se captura y se parte a cada toma; **tray_drop**, la de soltar sobre la estación de bandeja, también fuera de la vista.
_Evitar_: posición de reposo, pose de captura

**Freedrive**:
Modo en que el operador mueve el robot con la mano.

## El ciclo

**Ciclo**:
Lo que el robot hace con una pieza: detectar, clasificar la cara, tomar, voltear si hace falta, dejar en la bandeja y verificar.

**Toma** (*pick*):
Desde fuera de la vista hasta la pieza agarrada y levantada.
_Evitar_: picking, agarre

**Volteo** (*flip*):
Desde la pieza tomada con el hueso arriba hasta dejarla con la piel arriba, lista para tomarla otra vez.
_Evitar_: giro, rotación

**Colocación** (*place*):
Desde la pieza tomada hasta dejarla en la bandeja, con el robot de vuelta fuera de la vista.
_Evitar_: depósito, descarga

**Estrategia** (*strategy*):
Una forma de tomar, voltear o colocar una pieza, que a partir de lo que vio la cámara produce un plan.

**Plan** (*plan*):
Los pasos en orden (movimientos, acciones del gripper, salidas digitales, esperas de tiempo o de una entrada) que lleva a cabo una estrategia.

**Ejecutor** (*executor*):
El que lleva a cabo un plan con el robot y lo detiene si algo falla.

**Carga de bandeja** (*load*):
Una pieza tomada de la zona de toma y dejada en una bandeja libre, verificada con la cámara. Una bandeja está libre si no tiene una pieza adentro ni se cargó en el ciclo.
_Evitar_: carga a secas (es la del gripper, *payload*), llenado

**Verificación de colocación** (*placement verification*):
Revisar con la cámara cómo quedó la pieza. La colocación es exitosa si la pieza está en la bandeja, con la piel arriba y con el borde de sellado libre.

## Percepción

**Cámara física**:
La cámara montada, identificada por su *camera_id*. Su calibración y sus referencias le pertenecen a ella.

**Cámara simulada**:
Una fuente de imágenes guardadas (carpeta o video) que se hace pasar por una cámara física y usa sus datos.
_Evitar_: cámara falsa, mock

**Imagen** (*frame*):
Una captura de la cámara con la hora en que se tomó.
_Evitar_: foto

**Escena** (*scene*):
Lo que vio la cámara en una imagen: la zona revisada y las piezas que hay en ella, cada una con su cara si se clasificó.
_Evitar_: vista, inspección

**Detección** (*detection*):
Una pieza de la escena junto con su cara, si se clasificó.

**Pieza tomable** (*OK*):
Una pieza que el robot puede tomar tal como se ve.

**Grupo** (*cluster*):
Varias piezas juntas o encimadas que se ven como una sola. No se toma ni se clasifica su cara.
_Evitar_: montón, pila

**Pieza en el borde** (*cut by zone edge*):
Una pieza que toca el borde de la zona y puede estar en parte fuera de ella. Se clasifica su cara pero no se toma.

**Escena vacía** (*empty scene*):
Imagen de la zona sin piezas y con el robot fuera de la vista, promedio de varias capturas. El detector por referencia toma por pieza lo que cambió respecto de ella; se captura de nuevo si cambia la luz, el film, la exposición o la cámara.
_Evitar_: fondo, imagen de fondo

**Clasificador de cara** (*face classifier*):
Lo que decide la cara de una pieza a partir de la imagen.

**Margen** (*margin*):
Qué tan clara fue la decisión de la cara: positivo indica piel, negativo hueso, y cerca de cero es incierta.

**Muestra** (*sample*):
El recorte de una pieza tomable guardado con la cara que el operador indicó.

**Etiqueta** (*label*):
La cara que el operador dice que tienen las piezas, contra la que se compara lo detectado.

**Referencias de cara** (*face references*):
Lo que la cámara aprendió de las muestras de piel y de hueso, contra lo que compara el clasificador por histograma.
_Evitar_: modelo, entrenamiento

**Textura**:
La variación del brillo dentro de la pieza, en % por mm. Es baja en la piel y alta en el hueso.

## Calibración

**Calibración** (*calibration*):
La relación de cada cámara física entre un píxel de la imagen y un punto en la base del robot.

**Lente** (*intrinsics*):
El modelo óptico de la cámara: foco, centro y distorsión.
_Evitar_: intrínsecos (salvo en el código)

**Cámara-robot** (*extrinsics*):
La posición de la cámara respecto de la base, obtenida con el tablero y toques del pincho.

**Tablero** (*board*):
El patrón que se usa para calibrar: un ChArUco impreso o una placa de ajedrez sin marcadores (Cognex).
_Evitar_: patrón, damero

**Toque** (*touch*):
Una esquina del tablero tocada con la punta del pincho, con su posición en la base registrada.

**Homografía** / **Modelo de cámara** / **Escala nominal**:
Los tres métodos para pasar de píxeles a mm: un plano calibrado con el robot, el modelo 3D de la cámara calibrado con el robot, o una escala aproximada para probar la percepción sin calibrar (no sirve para mover el robot).

## Programas y registros

**Prueba** (*test*):
Un programa de `gt1_test/` que valida un concepto del piloto con el hardware del banco y deja un registro para medirlo.
_Evitar_: ensayo, test (reservado para pytest)

**Herramienta de trabajo** (*tool*):
Un programa de `gt1_tools/` que prepara el banco (visor, calibración, referencias, video).

**Criterio**:
Lo que mide una prueba y el valor que debe alcanzar (p. ej. R2: ≥ 99,5 % de acierto de cara).

**Acierto**:
La fracción de piezas tomables y etiquetadas cuya cara se detectó bien, sin contar las inciertas.

**Corrida** (*run*):
Una ejecución de un programa, con su carpeta de registro.
_Evitar_: sesión, ejecución

**Intento** (*attempt*):
Una línea del registro de la corrida: lo que hace falta para medir el criterio, con sus imágenes.

**Operador**:
La persona que maneja el banco desde la consola de operación.

**Consola de operación** (*operator console*):
La ventana con la imagen, la barra de estado y las teclas de un programa.

## Ambigüedades

- **Estación** y **estación de volteo** son cosas distintas: la estación es el hardware montado; la estación de volteo es un dispositivo de la celda. Decir siempre «estación de volteo» completo.
- **Herramienta** quiere decir dos cosas: lo que va en la brida del robot y un programa de `gt1_tools/`. Para el programa se dice «herramienta de trabajo».
- **Prueba** es un programa de validación con hardware; el **test** de pytest está en `tests/` y no usa hardware.
