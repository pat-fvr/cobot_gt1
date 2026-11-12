"""Esquema de cada archivo de configuración.

Un modelo por tipo de archivo. La validación es estricta: un campo que falta,
uno que sobra o un tipo equivocado detiene el programa con el archivo y el
campo en el mensaje. Ningún campo tiene valor por defecto: el valor está en el
archivo o el programa no arranca.

Los componentes con varias implementaciones (robot, cámara, gripper,
detectores) llevan un campo ``kind`` que elige la implementación.
"""

from __future__ import annotations

import itertools
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gt1.domain import DigitalInput, DigitalOutput, InputPort, OutputPort

Vector3 = tuple[float, float, float]
Point2 = tuple[float, float]
Joints6 = tuple[float, float, float, float, float, float]
CalibrationMethod = Literal["homography", "camera_model"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ---------------------------------------------------------------------- robots/


class UrRtdeRobotConfig(StrictModel):
    """UR10e (u otro e-Series) controlado por RTDE con ur_rtde."""

    kind: Literal["ur_rtde"]
    ip: str = Field(description="IP del controlador del robot.")
    linear_accel_mm_s2: float = Field(gt=0, description="Aceleración de los movimientos rectos.")
    joint_accel_deg_s2: float = Field(gt=0, description="Aceleración de los movimientos articulares.")
    stop_decel_mm_s2: float = Field(gt=0, description="Desaceleración al detener un movimiento recto.")
    stop_decel_deg_s2: float = Field(gt=0, description="Desaceleración al detener un movimiento articular.")
    poll_period_s: float = Field(
        gt=0, description="Cada cuánto se revisa un movimiento en curso (y si hay que detenerlo)."
    )
    tool_position_tol_mm: float = Field(
        gt=0, description="Diferencia aceptada entre el TCP pedido y el que informa el robot."
    )
    tool_rotation_tol_deg: float = Field(gt=0, description="Ídem para la orientación del TCP.")
    payload_tol_kg: float = Field(
        gt=0, description="Diferencia aceptada entre la carga pedida y la que informa el robot."
    )


class SimRobotConfig(StrictModel):
    """Robot simulado: llega al instante a cada objetivo y guarda el estado de sus salidas."""

    kind: Literal["sim"]
    initial_tcp_mm: Vector3 = Field(description="Posición inicial de la punta en la base.")
    initial_joints_deg: Joints6 = Field(description="Articulaciones iniciales.")


RobotConfig = Annotated[UrRtdeRobotConfig | SimRobotConfig, Field(discriminator="kind")]


# ---------------------------------------------------------------------- cameras/


class HomographyCalibrationConfig(StrictModel):
    """Homografía plana calibrada con el robot (calibration/<camera_id>/homography.json)."""

    method: Literal["homography"]


class CameraModelCalibrationConfig(StrictModel):
    """Modelo 3D de la cámara calibrado con el robot (calibration/<camera_id>/camera_model.json)."""

    method: Literal["camera_model"]


class NominalScaleCalibrationConfig(StrictModel):
    """Escala aproximada, sin calibrar: solo para probar la percepción (no para mover el robot).

    Los píxeles se pasan a mm multiplicando por ``mm_per_px``, en el sistema de
    la imagen (origen arriba a la izquierda, Y hacia abajo).
    """

    method: Literal["nominal_scale"]
    mm_per_px: float = Field(gt=0, description="mm de la mesa que cubre un píxel (medir un objeto conocido).")


CameraCalibrationConfig = Annotated[
    HomographyCalibrationConfig | CameraModelCalibrationConfig | NominalScaleCalibrationConfig,
    Field(discriminator="method"),
]


class MvsCameraConfig(StrictModel):
    """Cámara GigE Vision a través del SDK MVS (probado con la Basler Scout scA1390)."""

    kind: Literal["mvs"]
    camera_id: str = Field(
        description="Cámara física: sus datos están en calibration/<camera_id>/ y references/<camera_id>/."
    )
    calibration: CameraCalibrationConfig = Field(description="Cómo se pasa de píxeles a mm.")
    ip: str = Field(description="IP de la cámara.")
    expected_model: str = Field(description="Modelo que debe informar la cámara al conectarse.")
    expected_serial: str = Field(description="Número de serie que debe informar la cámara al conectarse.")
    first_frame_timeout_ms: int = Field(
        gt=0, description="Espera de la primera imagen (la Scout tarda ~1,1 s)."
    )
    frame_timeout_ms: int = Field(gt=0, description="Espera de cada imagen siguiente.")
    max_incomplete_frames: int = Field(
        gt=0, description="Imágenes seguidas con paquetes perdidos antes de fallar."
    )
    genicam_nodes: dict[str, float | int | bool | str] = Field(
        description="Nodos GenICam que se fijan al conectar (p. ej. ExposureTimeAbs). Vacío = no se tocan."
    )


class WebcamCameraConfig(StrictModel):
    """Cámara USB a través de OpenCV."""

    kind: Literal["webcam"]
    camera_id: str = Field(
        description="Cámara física: sus datos están en calibration/<camera_id>/ y references/<camera_id>/."
    )
    calibration: CameraCalibrationConfig = Field(description="Cómo se pasa de píxeles a mm.")
    device_name: str = Field(description="Nombre de la cámara en Windows (Administrador de dispositivos).")
    width_px: int = Field(
        gt=0, description="Ancho pedido; si la cámara entrega otro, el programa se detiene."
    )
    height_px: int = Field(gt=0, description="Alto pedido.")
    fourcc: str = Field(
        min_length=4,
        max_length=4,
        description="Formato de video: MJPG (comprimido, más imágenes/s) o YUY2 (sin compresión, más lento).",
    )
    warmup_frames: int = Field(
        ge=1, description="Imágenes que se descartan al conectar mientras se ajusta la exposición."
    )
    first_frame_timeout_s: float = Field(gt=0, description="Plazo para recibir esas imágenes al conectar.")
    open_attempts: int = Field(ge=1, description="Veces que se reabre la cámara si no entrega imágenes.")
    capture_properties: dict[str, float] = Field(
        description="Propiedades de OpenCV que se fijan al conectar, sin el prefijo CAP_PROP_ "
        "(p. ej. EXPOSURE, AUTO_WB). Vacío = las de la cámara."
    )


class FolderCameraConfig(StrictModel):
    """Imágenes guardadas, en orden alfabético: para probar la percepción sin cámara."""

    kind: Literal["folder"]
    camera_id: str = Field(
        description="La cámara que tomó las imágenes: se usan su calibración y sus referencias."
    )
    calibration: CameraCalibrationConfig = Field(description="Cómo se pasa de píxeles a mm.")
    directory: str = Field(description="Carpeta con las imágenes (.png, .jpg, .bmp), relativa al proyecto.")
    loop: bool = Field(description="Al terminar, vuelve a la primera imagen.")


class VideoFileCameraConfig(StrictModel):
    """Un video grabado, imagen por imagen: para probar la percepción sin cámara."""

    kind: Literal["video"]
    camera_id: str = Field(
        description="La cámara que grabó el video: se usan su calibración y sus referencias."
    )
    calibration: CameraCalibrationConfig = Field(description="Cómo se pasa de píxeles a mm.")
    path: str = Field(description="Archivo de video (.mp4, .avi), relativo al proyecto.")


CameraConfig = Annotated[
    MvsCameraConfig | WebcamCameraConfig | FolderCameraConfig | VideoFileCameraConfig,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------- grippers/


class TcpConfig(StrictModel):
    position_mm: Vector3 = Field(description="Punta de la herramienta respecto de la brida.")
    rotation_vector_deg: Vector3 = Field(
        description="Orientación de la punta (vector de rotación, como en la consola UR)."
    )


class PayloadConfig(StrictModel):
    mass_kg: float = Field(ge=0, description="Masa de la herramienta (sin pieza).")
    center_of_gravity_mm: Vector3 = Field(description="Centro de gravedad respecto de la brida.")


class OutputChannel(StrictModel):
    """Una salida digital del controlador (``{ port = "standard", number = 3 }`` es DO3)."""

    port: OutputPort = Field(description="standard (DO), configurable (CO) o tool (TO).")
    number: int = Field(ge=0, le=7)

    def digital_output(self) -> DigitalOutput:
        return DigitalOutput(self.port, self.number)


class InputChannel(StrictModel):
    """Una entrada digital del controlador (``{ port = "standard", number = 1 }`` es DI1)."""

    port: InputPort = Field(description="standard (DI), configurable (CI) o tool (TI).")
    number: int = Field(ge=0, le=7)

    def digital_input(self) -> DigitalInput:
        return DigitalInput(self.port, self.number)


class OutputSetting(OutputChannel):
    value: bool


class GripperActionConfig(StrictModel):
    description: str
    outputs: tuple[OutputSetting, ...] = Field(min_length=1, description="Salidas que se fijan, en orden.")
    settle_s: float = Field(
        ge=0,
        description=(
            "Espera después de fijar las salidas (los dedos tardan en moverse). No se espera si las "
            "salidas ya estaban así."
        ),
    )


# Hacia qué lado del eje Y del TCP queda la parte ancha del gripper (va hacia los lóbulos de la pieza).
WideSide = Literal["-y", "+y", "symmetric"]


class DigitalOutputGripperConfig(StrictModel):
    """Gripper accionado por salidas digitales del robot (p. ej. soft gripper neumático)."""

    kind: Literal["digital_outputs"]
    description: str
    tcp: TcpConfig
    payload: PayloadConfig
    wide_side: WideSide = Field(
        description="Lado del eje Y del TCP hacia el que queda la parte ancha de un gripper asimétrico: al "
        "tomar, va hacia los lóbulos de la pieza y la parte angosta hacia la punta. 'symmetric' si no "
        "tiene un lado ancho."
    )
    actions: dict[str, GripperActionConfig] = Field(
        description="Acciones por nombre. 'grip' (tomar) y 'release' (soltar) son obligatorias."
    )

    @model_validator(mode="after")
    def _has_grip_and_release(self) -> DigitalOutputGripperConfig:
        missing = {"grip", "release"} - self.actions.keys()
        if missing:
            raise ValueError(f"Faltan las acciones obligatorias: {', '.join(sorted(missing))}.")
        return self


class FixedToolConfig(StrictModel):
    """Herramienta sin acciones (p. ej. el pincho de calibración)."""

    kind: Literal["fixed"]
    description: str
    tcp: TcpConfig
    payload: PayloadConfig


GripperConfig = Annotated[DigitalOutputGripperConfig | FixedToolConfig, Field(discriminator="kind")]


# ---------------------------------------------------------------------- trays/


class TrayConfig(StrictModel):
    description: str
    rim_length_mm: float = Field(gt=0, description="Largo exterior del borde (de labio a labio).")
    rim_width_mm: float = Field(gt=0, description="Ancho exterior del borde (de labio a labio).")
    floor_length_mm: float = Field(
        gt=0, description="Largo del fondo, entre los pliegues donde las paredes tocan la mesa."
    )
    floor_width_mm: float = Field(
        gt=0, description="Ancho del fondo, entre los pliegues donde las paredes tocan la mesa."
    )
    depth_mm: float = Field(gt=0, description="Alto desde el fondo hasta el borde de sellado.")
    seal_flange_mm: float = Field(gt=0, description="Ancho del borde de sellado.")
    min_clearance_to_flange_mm: float = Field(
        ge=0, description="Distancia mínima de la pieza al borde de sellado."
    )

    @model_validator(mode="after")
    def _floor_inside_rim(self) -> Self:
        if not (self.floor_length_mm < self.rim_length_mm and self.floor_width_mm < self.rim_width_mm):
            raise ValueError(
                f"El fondo ({self.floor_length_mm:g} x {self.floor_width_mm:g} mm) tiene que ser más chico "
                f"que el borde ({self.rim_length_mm:g} x {self.rim_width_mm:g} mm): las paredes se abren "
                "hacia arriba."
            )
        if self.rim_width_mm > self.rim_length_mm or self.floor_width_mm > self.floor_length_mm:
            raise ValueError(
                "El largo es el lado mayor: rim_length_mm y floor_length_mm van con el lado largo."
            )
        return self


# ---------------------------------------------------------------------- cell.toml


class ZoneConfig(StrictModel):
    polygon_mm: tuple[Point2, ...] = Field(min_length=3, description="Vértices (X, Y) en la base, en orden.")


class ZonesConfig(StrictModel):
    pick: ZoneConfig = Field(description="Donde se dejan las piezas para que el robot las tome.")
    tray: ZoneConfig = Field(description="Donde está la bandeja.")
    flip_landing: ZoneConfig = Field(description="Donde caen las piezas después de la estación de volteo.")

    def by_name(self, name: ZoneName) -> ZoneConfig:
        return {"pick": self.pick, "tray": self.tray, "flip_landing": self.flip_landing}[name]


ZoneName = Literal["pick", "tray", "flip_landing"]
# Una zona de cell.toml, o toda la imagen (para cámaras sin celda, p. ej. con escala nominal).
ViewZoneName = Literal["pick", "tray", "flip_landing", "full_view"]


class TrayPlacementConfig(StrictModel):
    """Dónde está la bandeja cuando es fija (mientras no hay detector de bandeja)."""

    center_mm: Point2 = Field(description="Centro del fondo de la bandeja (X, Y) en la base.")
    axis_angle_deg: float = Field(
        description="Ángulo del lado largo de la bandeja respecto del eje X de la base."
    )


class CellConfig(StrictModel):
    """Cómo está armado el banco de pruebas."""

    table_z_mm: float = Field(description="Altura de la superficie de la mesa en la base del robot.")
    zones: ZonesConfig
    tray_placement: TrayPlacementConfig = Field(description="Bandeja fija: dónde se dejan las piezas.")


# ---------------------------------------------------------------------- perception/


class OtsuThreshold(StrictModel):
    method: Literal["otsu"]


class FixedThreshold(StrictModel):
    method: Literal["fixed"]
    level: int = Field(ge=0, le=255)


class PieceContourConfig(StrictModel):
    """Limpieza de la máscara y medición de los contornos: lo común a los detectores de piezas."""

    blur_kernel_px: int = Field(ge=1, description="Suavizado antes de separar piezas y fondo (impar).")
    morphology_kernel_px: int = Field(
        ge=1, description="Limpieza de la máscara: elimina puntos y rellena huecos (impar)."
    )
    contour_height_mm: float = Field(
        ge=0, description="Altura sobre la mesa a la que se proyecta el contorno."
    )
    min_area_mm2: float = Field(gt=0, description="Manchas más chicas se ignoran.")
    max_single_piece_area_mm2: float = Field(
        gt=0, description="Sobre esto, el contorno es un grupo de piezas."
    )

    @model_validator(mode="after")
    def _odd_kernels(self) -> Self:
        for name in ("blur_kernel_px", "morphology_kernel_px"):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} debe ser impar.")
        return self


class BackgroundPieceDetectorConfig(PieceContourConfig):
    """Piezas sobre un fondo de alto contraste: umbral + contornos."""

    kind: Literal["background"]
    pieces_are: Literal["brighter", "darker", "different"] = Field(
        description="Las piezas son más claras, más oscuras o de otro color que el fondo."
    )
    threshold: Annotated[OtsuThreshold | FixedThreshold, Field(discriminator="method")]


class ReferencePieceDetectorConfig(PieceContourConfig):
    """Piezas sobre una escena conocida: diferencia con una imagen de la escena vacía."""

    kind: Literal["reference"]
    min_difference_pct: float = Field(
        gt=0,
        description="Diferencia mínima con la escena vacía, en % de su brillo, para que un píxel sea pieza.",
    )
    glint_min_level: int = Field(
        ge=1,
        le=255,
        description=(
            "Brillo (0-255) desde el que un punto más angosto que morphology_kernel_px, en la imagen o "
            "en la escena vacía, es un reflejo del film: no cuenta como cambio, porque los reflejos se "
            "mueven con cualquier roce del film."
        ),
    )
    speck_kernel_px: int = Field(
        ge=1,
        description=(
            "Antes de juntar los trozos de cada pieza, borra los puntos sueltos más angostos que esto "
            "(destellos tenues del film). Menor que morphology_kernel_px."
        ),
    )


class SkinOutlinePieceDetectorConfig(PieceContourConfig):
    """Pechugas con la piel arriba por su contorno: bordes de contraste relativo, sin escena vacía.

    ``morphology_kernel_px`` es la apertura que quita, después de rellenar lo encerrado, los bordes
    sueltos que quedaron pegados a la pieza.
    """

    kind: Literal["skin_outline"]
    search_scale: float = Field(
        gt=0,
        le=1,
        description="Escala de la imagen en que se buscan los contornos; los kernels en px (de la imagen "
        "completa) se reducen igual. La textura y las medidas se toman en la imagen completa.",
    )
    glint_kernel_px: int = Field(
        ge=1,
        description="Antes de buscar bordes se borran los reflejos del film y las rayas más angostos que "
        "esto (apertura y cierre en gris, impar).",
    )
    edge_levels_pct_per_mm: tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Contrastes relativos (variación del brillo en % por mm) desde los que un píxel es borde, de "
            "mayor a menor. Cada pieza se acepta con el primero con el que su contorno queda cerrado: los "
            "bajos cierran contornos de poco contraste (piel clara sobre fondo claro) pero juntan más "
            "bordes del fondo."
        ),
    )
    outline_gap_kernel_px: int = Field(
        ge=1, description="Cierre que une los trozos del contorno separados por menos que esto (impar)."
    )
    background_ring_px: int = Field(
        ge=1, description="Ancho de la franja alrededor de la candidata en que se mide el brillo del fondo."
    )
    shadow_level_fraction: float = Field(
        ge=0,
        description="Si la pieza es más clara que el fondo, se le quita lo que es más oscuro que el fondo "
        "en más que esta fracción de lo que la pieza es más clara: su sombra, que cierra contorno con ella.",
    )
    length_range_mm: Point2 = Field(
        description="Largo mínimo y máximo de una pechuga (a lo largo de su eje)."
    )
    width_range_mm: Point2 = Field(description="Ancho mínimo y máximo de una pechuga.")
    min_solidity: float = Field(
        gt=0,
        le=1,
        description="Área del contorno sobre la de su envolvente convexa: más bajo es una mancha irregular.",
    )
    skin_smoothing_mm: float = Field(
        gt=0, description="Textura de la superficie: suavizado antes de medirla (como el clasificador)."
    )
    skin_mask_erosion_mm: float = Field(
        ge=0, description="Textura de la superficie: borde de la pieza que no se mide."
    )
    max_skin_texture_pct_per_mm: float = Field(
        gt=0,
        description="Textura máxima de la piel; con más, es el lado del hueso, film o una bandeja.",
    )

    @model_validator(mode="after")
    def _odd_kernels_and_decreasing_levels(self) -> Self:
        for name in ("glint_kernel_px", "outline_gap_kernel_px"):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} debe ser impar.")
        levels = self.edge_levels_pct_per_mm
        if any(lower >= higher for higher, lower in itertools.pairwise(levels)):
            raise ValueError(f"edge_levels_pct_per_mm debe ir de mayor a menor: {list(levels)}.")
        for name in ("length_range_mm", "width_range_mm"):
            low, high = getattr(self, name)
            if not 0 < low < high:
                raise ValueError(f"{name} debe ser [mínimo, máximo] con 0 < mínimo < máximo: {[low, high]}.")
        return self


PieceDetectorConfig = Annotated[
    BackgroundPieceDetectorConfig | ReferencePieceDetectorConfig | SkinOutlinePieceDetectorConfig,
    Field(discriminator="kind"),
]


class RimAndFloorTrayDetectorConfig(StrictModel):
    """Bandejas por el pliegue del fondo en los lados en sombra y el labio del borde en los otros,
    confirmadas por lo que encierran (PET vacío, parecido a la mesa)."""

    kind: Literal["rim_and_floor"]
    grid_mm: float = Field(gt=0, description="Paso de la vista cenital de la zona: precisión de la posición.")
    angle_step_deg: float = Field(
        gt=0, le=10, description="Paso de los ángulos con que se afina cada pose de la búsqueda gruesa."
    )
    coarse_grid_mm: float = Field(
        gt=0,
        description="Paso de la grilla de la búsqueda gruesa, múltiplo de grid_mm: cada celda guarda la "
        "línea más marcada de las de grid_mm que junta.",
    )
    coarse_angle_step_deg: float = Field(
        gt=0, le=20, description="Paso con que la búsqueda gruesa prueba los ángulos de 0 a 180°."
    )
    glint_width_mm: float = Field(
        gt=0,
        description="Los reflejos más angostos que esto se borran antes de buscar líneas: entre los "
        "reflejos del film quedan huecos oscuros que parecen líneas.",
    )
    dark_line_width_mm: float = Field(
        gt=0,
        description="Las líneas oscuras más angostas que esto se realzan; una sombra o un cambio de mesa, "
        "más anchos, no.",
    )
    shadow_toward_deg: float = Field(
        description="Hacia dónde caen las sombras de las paredes, como ángulo en la base: los lados cuya "
        "normal hacia afuera apunta hacia aquí (a menos de 90°) muestran el pliegue del fondo; los otros, "
        "el labio del borde. Depende de la luz del banco.",
    )
    side_tolerance_mm: float = Field(
        ge=0, description="Cuánto puede estar corrida la línea de cada lado respecto del rectángulo ideal."
    )
    side_length_fraction: float = Field(
        gt=0,
        le=1,
        description="Parte central de cada lado que se mide (las esquinas son redondeadas).",
    )
    neighbor_offset_mm: float = Field(
        gt=0,
        description="Distancia a las paralelas, por dentro y por fuera, con que se compara cada lado.",
    )
    min_fold_contrast_pct: float = Field(
        gt=0,
        description="Contraste mínimo del pliegue en el mejor lado en sombra (cuánto más oscura es su línea "
        "que las paralelas, en %) para aceptar la bandeja.",
    )
    min_line_contrast_pct: float = Field(
        gt=0,
        description="Contraste mínimo, en %, del mejor lado largo y del mejor lado corto: cada uno fija "
        "dónde está la bandeja a lo largo del otro. Dos lados paralelos no la fijan, y el borde de una "
        "pechuga o de una caja es una sola línea.",
    )
    min_wall_vs_table_pct: float = Field(
        description="Brillo mínimo de las paredes respecto de la mesa alrededor, en % (negativo = más "
        "oscuras): una caja o la sombra de una pieza oscurecen más que el PET.",
    )
    max_floor_vs_table_pct: float = Field(
        description="Brillo máximo del interior respecto de la mesa alrededor, en %: una pechuga aclara "
        "más que el PET vacío.",
    )

    @model_validator(mode="after")
    def _coarse_grid_multiple_of_grid(self) -> Self:
        factor = self.coarse_grid_mm / self.grid_mm
        if round(factor) < 1 or abs(factor - round(factor)) > 1e-6:
            raise ValueError(
                f"coarse_grid_mm ({self.coarse_grid_mm}) debe ser un múltiplo de grid_mm ({self.grid_mm})."
            )
        return self


TrayDetectorConfig = RimAndFloorTrayDetectorConfig


class HistogramFaceClassifierConfig(StrictModel):
    """Cara por comparación del histograma de la pieza con referencias de piel y de hueso."""

    kind: Literal["histogram"]
    color_space: Literal["gray", "hsv"] = Field(
        description="gray para cámaras mono; hsv (tono y saturación) para color."
    )
    bins_per_channel: int = Field(ge=2, le=256)
    mask_erosion_px: int = Field(ge=0, description="Se ignora este borde de la pieza (mezcla pieza y fondo).")
    min_margin: float = Field(ge=0, description="Bajo este margen la cara queda 'incierto'.")


class TextureFaceClassifierConfig(StrictModel):
    """Cara por la textura de la superficie: la piel es lisa; el hueso, costillas y cartílago."""

    kind: Literal["texture"]
    smoothing_mm: float = Field(
        gt=0, description="Se suaviza a esta escala antes de medir: ignora el ruido y el poro de la piel."
    )
    mask_erosion_mm: float = Field(
        ge=0, description="Se ignora este borde de la pieza (el salto pieza-fondo no es textura)."
    )
    threshold_pct_per_mm: float = Field(
        gt=0, description="Frontera piel/hueso: variación media del brillo, en % por mm."
    )
    min_margin_pct_per_mm: float = Field(
        ge=0, description="A menos de esto de la frontera la cara queda 'incierto'."
    )


FaceClassifierConfig = Annotated[
    HistogramFaceClassifierConfig | TextureFaceClassifierConfig, Field(discriminator="kind")
]


# ---------------------------------------------------------------------- strategies/


class TopDownPickConfig(StrictModel):
    """Toma desde arriba: el gripper baja vertical al centro de la pieza, con su eje Y a lo largo de ella."""

    travel_speed_mm_s: float = Field(
        gt=0, description="Velocidad hasta quedar sobre la pieza, a la altura de llegada."
    )
    approach_height_mm: float = Field(
        gt=0,
        description=(
            "Altura de la punta (TCP) sobre la mesa a la que va primero en vertical, desde donde esté, y "
            "a la que viaja hasta quedar sobre la pieza, antes de bajar."
        ),
    )
    grasp_height_mm: float = Field(ge=0, description="Altura de la punta sobre la mesa al cerrar el gripper.")
    after_grip_descent_mm: float = Field(
        ge=0,
        description=(
            "Cuánto baja la punta después de cerrar, antes de subir (0: no baja). Los dedos de un soft "
            "gripper se recogen al cerrar y quedan sobre la mesa: bajar los vuelve a meter bajo la pieza. "
            "Puede dejar el TCP bajo la mesa, porque el TCP es la punta de los dedos abiertos."
        ),
    )
    lift_height_mm: float = Field(gt=0, description="Altura de la punta sobre la mesa con la pieza tomada.")
    descend_speed_mm_s: float = Field(
        gt=0,
        description=(
            "Velocidad de los movimientos verticales: a la altura de llegada, al bajar a la pieza y al subir "
            "con ella."
        ),
    )
    blend_radius_mm: float = Field(
        ge=0,
        description=(
            "Radio con que el robot pasa en curva, sin detenerse, por el punto de arriba de la subida (de "
            "subir con la pieza a ir a la bandeja): empieza a avanzar antes de terminar de subir. 0: se "
            "detiene ahí. Sobre la pieza se detiene siempre antes de bajar."
        ),
    )

    @model_validator(mode="after")
    def _grasp_is_lowest(self) -> Self:
        if self.grasp_height_mm >= min(self.approach_height_mm, self.lift_height_mm):
            raise ValueError("grasp_height_mm debe ser menor que approach_height_mm y que lift_height_mm.")
        return self

    @model_validator(mode="after")
    def _blend_fits_the_lift(self) -> Self:
        """El controlador rechaza una curva más larga que el tramo; con la mitad queda margen."""
        lowest_mm = self.grasp_height_mm - self.after_grip_descent_mm
        rise_mm = self.lift_height_mm - lowest_mm
        if 2 * self.blend_radius_mm > rise_mm:
            raise ValueError(
                f"blend_radius_mm = {self.blend_radius_mm} no cabe en la subida con la pieza "
                f"({rise_mm:.0f} mm): debe ser a lo más la mitad."
            )
        return self


class TopDownPlaceConfig(StrictModel):
    """Colocación sobre la bandeja: el gripper llega directo sobre su centro, alineado con ella, y suelta."""

    travel_speed_mm_s: float = Field(gt=0, description="Velocidad hasta quedar sobre la bandeja.")
    release_height_mm: float = Field(
        gt=0,
        description=(
            "Altura de la punta sobre el fondo de la bandeja al llegar y abrir el gripper. El gripper "
            "llega desde el costado a esta altura: no menos que la profundidad de la bandeja."
        ),
    )
    lobes_toward_deg: float = Field(
        description=(
            "Hacia dónde quedan los lóbulos en la bandeja (ángulo respecto del eje X de la base): de los "
            "dos sentidos del lado largo de la bandeja, el más cercano a este. Solo con piezas cuyo "
            "detector distingue la punta de los lóbulos; las otras se dejan girando la muñeca lo menos "
            "posible."
        )
    )


# ---------------------------------------------------------------------- boards/


class CharucoBoardConfig(StrictModel):
    """Tablero ChArUco de calibración."""

    kind: Literal["charuco"]
    squares_x: int = Field(ge=3)
    squares_y: int = Field(ge=3)
    square_mm: float = Field(gt=0)
    marker_mm: float = Field(gt=0)
    dictionary: str = Field(description="Diccionario ArUco de OpenCV, p. ej. DICT_5X5_100.")
    thickness_mm: float = Field(ge=0, description="Espesor del tablero (de la mesa a la superficie impresa).")


class ChessboardConfig(StrictModel):
    """Tablero de ajedrez sin marcadores (p. ej. una placa Cognex); se detecta aunque se vea en parte."""

    kind: Literal["chessboard"]
    squares_x: int = Field(ge=3, description="Cuadros a lo ancho (las esquinas internas son uno menos).")
    squares_y: int = Field(ge=3, description="Cuadros a lo alto.")
    square_mm: float = Field(gt=0)
    thickness_mm: float = Field(ge=0, description="Espesor del tablero (de la mesa a la superficie impresa).")
    min_corner_contrast: float = Field(
        gt=0,
        description="Diferencia mínima de brillo (0-255) entre los cuadros claros y oscuros que rodean "
        "una esquina para aceptarla. Bajarla si la imagen es oscura y faltan esquinas.",
    )


BoardConfig = Annotated[CharucoBoardConfig | ChessboardConfig, Field(discriminator="kind")]


# ---------------------------------------------------------------------- station.toml


class StationConfig(StrictModel):
    """Hardware montado en el banco según la última configuración confirmada."""

    robot: str
    camera: str
    gripper: str
    tray: str


# ---------------------------------------------------------------------- robot_poses.toml


class RobotPoseConfig(StrictModel):
    """Una pose enseñada del robot (``[<nombre>]`` en ``robot_poses.toml``)."""

    joints_deg: Joints6 = Field(description="Articulaciones, de la base a la muñeca 3: el robot va por aquí.")
    tcp_mm: Vector3 = Field(
        description="Posición del TCP en la base cuando se enseñó, para quien lee el archivo: el robot "
        "no la usa."
    )
    taught_at: str = Field(description="Fecha y hora en que se enseñó (AAAA-MM-DD HH:MM).")


RobotPoses = dict[str, RobotPoseConfig]


# ---------------------------------------------------------------------- ajustes compartidos de programas


class ClearViewSettings(StrictModel):
    """Cómo un programa saca el robot de la vista para capturar (``[settings.clear_view]``)."""

    pose: str = Field(
        description=(
            "Pose de robot_poses.toml fuera de la vista de la cámara, con la herramienta hacia abajo y "
            "sobre la mesa: desde aquí se captura (siempre desde la misma pose, para que su sombra sea la "
            "de la escena vacía) y se parte a cada toma."
        )
    )
    safe_height_mm: float = Field(
        gt=0,
        description="Altura de la punta sobre la mesa a la que sube en vertical antes de ir a la pose "
        "fuera de la vista.",
    )
    rise_speed_mm_s: float = Field(gt=0, description="Velocidad de esa subida.")
    clear_speed_deg_s: float = Field(gt=0, description="Velocidad articular hasta la pose fuera de la vista.")
    capture_settle_s: float = Field(
        ge=0, description="Espera desde que el robot llega a la pose fuera de la vista hasta capturar."
    )


class TrayStationSettings(StrictModel):
    """Bandeja fija detenida por un tope, con un sensor que dice que llegó (``[settings.tray_station]``).

    El tope se cierra con su salida en 1 y se abre con 0; el sensor da 1 con una bandeja contra el tope.
    """

    stopper: OutputChannel = Field(
        description="Salida del tope: 1 cerrado (detiene la bandeja bajo la pieza), 0 abierto (la deja "
        "pasar)."
    )
    tray_sensor: InputChannel = Field(
        description="Entrada del sensor de bandeja: 1 con una bandeja en el tope. Después de soltar, el tope "
        "se abre y se vuelve a cerrar cuando el sensor pasa a 0 (salió la bandeja cargada)."
    )
    stopper_close_delay_s: float = Field(
        ge=0,
        description="Cuánto después de que el sensor deja de ver la bandeja cargada se cierra el tope: el "
        "sensor la pierde antes de que termine de pasar sobre él.",
    )
