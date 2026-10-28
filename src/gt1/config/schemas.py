"""Esquema de cada archivo de configuración.

Un modelo por tipo de archivo. La validación es estricta: un campo que falta,
uno que sobra o un tipo equivocado detiene el programa con el archivo y el
campo en el mensaje. Ningún campo tiene valor por defecto: el valor está en el
archivo o el programa no arranca.

Los componentes con varias implementaciones (robot, cámara, gripper,
detectores) llevan un campo ``kind`` que elige la implementación.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gt1.domain import OutputPort

Vector3 = tuple[float, float, float]
Point2 = tuple[float, float]
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
    initial_joints_deg: tuple[float, float, float, float, float, float] = Field(
        description="Articulaciones iniciales."
    )


RobotConfig = Annotated[UrRtdeRobotConfig | SimRobotConfig, Field(discriminator="kind")]


# ---------------------------------------------------------------------- cameras/


class MvsCameraConfig(StrictModel):
    """Cámara GigE Vision a través del SDK MVS (probado con la Basler Scout scA1390)."""

    kind: Literal["mvs"]
    camera_id: str = Field(
        description="Cámara física: sus datos están en calibration/<camera_id>/ y references/<camera_id>/."
    )
    calibration_method: CalibrationMethod = Field(
        description="Calibración que se usa para pasar de píxeles a la base."
    )
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
    calibration_method: CalibrationMethod
    device_index: int = Field(ge=0, description="Índice de la cámara para OpenCV (0 = la primera).")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    warmup_frames: int = Field(
        ge=0, description="Imágenes que se descartan al conectar mientras se ajusta la exposición."
    )


class FolderCameraConfig(StrictModel):
    """Imágenes guardadas, en orden alfabético: para probar la percepción sin cámara."""

    kind: Literal["folder"]
    camera_id: str = Field(
        description="La cámara que tomó las imágenes: se usan su calibración y sus referencias."
    )
    calibration_method: CalibrationMethod
    directory: str = Field(description="Carpeta con las imágenes (.png, .jpg, .bmp), relativa al proyecto.")
    loop: bool = Field(description="Al terminar, vuelve a la primera imagen.")


CameraConfig = Annotated[
    MvsCameraConfig | WebcamCameraConfig | FolderCameraConfig, Field(discriminator="kind")
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


class OutputSetting(StrictModel):
    port: OutputPort = Field(description="standard (DO), configurable (CO) o tool (TO).")
    number: int = Field(ge=0, le=7)
    value: bool


class GripperActionConfig(StrictModel):
    description: str
    outputs: tuple[OutputSetting, ...] = Field(min_length=1, description="Salidas que se fijan, en orden.")
    settle_s: float = Field(
        ge=0, description="Espera después de fijar las salidas (los dedos tardan en moverse)."
    )


class DigitalOutputGripperConfig(StrictModel):
    """Gripper accionado por salidas digitales del robot (p. ej. soft gripper neumático)."""

    kind: Literal["digital_outputs"]
    description: str
    tcp: TcpConfig
    payload: PayloadConfig
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
    inner_length_mm: float = Field(gt=0, description="Largo interior del fondo.")
    inner_width_mm: float = Field(gt=0, description="Ancho interior del fondo.")
    depth_mm: float = Field(gt=0, description="Alto desde el fondo hasta el borde de sellado.")
    seal_flange_mm: float = Field(gt=0, description="Ancho del borde de sellado.")
    min_clearance_to_flange_mm: float = Field(
        ge=0, description="Distancia mínima de la pieza al borde de sellado."
    )


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


class CellConfig(StrictModel):
    """Cómo está armado el banco de pruebas."""

    table_z_mm: float = Field(description="Altura de la superficie de la mesa en la base del robot.")
    zones: ZonesConfig


# ---------------------------------------------------------------------- perception/


class OtsuThreshold(StrictModel):
    method: Literal["otsu"]


class FixedThreshold(StrictModel):
    method: Literal["fixed"]
    level: int = Field(ge=0, le=255)


class BackgroundPieceDetectorConfig(StrictModel):
    """Piezas sobre un fondo de alto contraste: umbral + contornos."""

    kind: Literal["background"]
    pieces_are: Literal["brighter", "darker"] = Field(
        description="Las piezas son más claras o más oscuras que el fondo."
    )
    threshold: Annotated[OtsuThreshold | FixedThreshold, Field(discriminator="method")]
    blur_kernel_px: int = Field(ge=1, description="Suavizado antes del umbral (impar).")
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
    def _odd_kernels(self) -> BackgroundPieceDetectorConfig:
        for name in ("blur_kernel_px", "morphology_kernel_px"):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} debe ser impar.")
        return self


# Con una segunda implementación, pasa a ser una unión discriminada por ``kind`` (como CameraConfig).
PieceDetectorConfig = BackgroundPieceDetectorConfig


class HistogramFaceClassifierConfig(StrictModel):
    """Cara por comparación del histograma de la pieza con referencias de piel y de hueso."""

    kind: Literal["histogram"]
    color_space: Literal["gray", "hsv"] = Field(
        description="gray para cámaras mono; hsv (tono y saturación) para color."
    )
    bins_per_channel: int = Field(ge=2, le=256)
    mask_erosion_px: int = Field(ge=0, description="Se ignora este borde de la pieza (mezcla pieza y fondo).")
    min_margin: float = Field(ge=0, description="Bajo este margen la cara queda 'incierto'.")


FaceClassifierConfig = HistogramFaceClassifierConfig


# ---------------------------------------------------------------------- boards/


class CharucoBoardConfig(StrictModel):
    """Tablero ChArUco de calibración."""

    squares_x: int = Field(ge=3)
    squares_y: int = Field(ge=3)
    square_mm: float = Field(gt=0)
    marker_mm: float = Field(gt=0)
    dictionary: str = Field(description="Diccionario ArUco de OpenCV, p. ej. DICT_5X5_100.")
    thickness_mm: float = Field(ge=0, description="Espesor del tablero (de la mesa a la superficie impresa).")


# ---------------------------------------------------------------------- station.toml


class StationConfig(StrictModel):
    """Hardware montado en el banco según la última configuración confirmada."""

    robot: str
    camera: str
    gripper: str
    tray: str
