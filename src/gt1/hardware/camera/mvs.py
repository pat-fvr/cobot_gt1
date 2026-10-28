"""Cámara GigE Vision a través del SDK MVS (Hikrobot).

Probado con la Basler Scout scA1390 (monocromática, 1392x1040). MVS habla
GigE Vision estándar, así que sirve para cualquier cámara GigE con los nodos
GenICam habituales. ``pypylon`` no funciona bien con esta cámara.

El SDK se carga al conectar, para que el resto del paquete se pueda importar
en equipos sin MVS instalado.
"""

# pyright: basic
# El wrapper de MVS (MvImport) es ctypes sin tipos: este archivo es el único que lo toca.

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from gt1.config.schemas import MvsCameraConfig
from gt1.domain import Frame
from gt1.domain.frame import Image

from .base import Camera

logger = logging.getLogger(__name__)

MVS_RUNTIME_DIR = r"C:\Program Files (x86)\Common Files\MVS\Runtime"


class MvsError(RuntimeError):
    """Una llamada al SDK MVS devolvió un error."""

    def __init__(self, operation: str, code: int) -> None:
        super().__init__(f"{operation} falló (código MVS 0x{code & 0xFFFFFFFF:08X}).")


@dataclass(frozen=True)
class _DeviceIdentity:
    model: str
    serial: str


class MvsCamera(Camera):
    def __init__(self, config: MvsCameraConfig) -> None:
        self._config = config
        self._sdk: ModuleType | None = None
        self._handle: Any = None
        self._waiting_first_frame = True

    @property
    def camera_id(self) -> str:
        return self._config.camera_id

    @property
    def is_simulated(self) -> bool:
        return False

    def connect(self) -> None:
        if self._handle is not None:
            return
        sdk = _load_sdk()
        _check(sdk.MvCamera.MV_CC_Initialize(), "MV_CC_Initialize")
        self._sdk = sdk
        try:
            device, identity = self._find_device(sdk)
            self._check_identity(identity)
            handle = sdk.MvCamera()
            _check(handle.MV_CC_CreateHandle(device), "MV_CC_CreateHandle")
            self._handle = handle
            _check(handle.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0), "MV_CC_OpenDevice")
            self._configure(handle, sdk)
            _check(handle.MV_CC_StartGrabbing(), "MV_CC_StartGrabbing")
        except Exception:
            self.disconnect()
            raise
        self._waiting_first_frame = True
        logger.info("Cámara conectada: %s S/N %s (%s).", identity.model, identity.serial, self._config.ip)

    def disconnect(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.MV_CC_StopGrabbing()
            handle.MV_CC_CloseDevice()
            handle.MV_CC_DestroyHandle()
            logger.info("Cámara desconectada.")
        if self._sdk is not None:
            self._sdk.MvCamera.MV_CC_Finalize()
            self._sdk = None

    def capture(self) -> Frame:
        """Primera imagen completa que llegue; las que pierden paquetes se descartan."""
        handle, sdk = self._require_connected()
        for _ in range(self._config.max_incomplete_frames):
            timeout_ms = (
                self._config.first_frame_timeout_ms
                if self._waiting_first_frame
                else self._config.frame_timeout_ms
            )
            buffer = sdk.MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(buffer), 0, ctypes.sizeof(buffer))
            _check(handle.MV_CC_GetImageBuffer(buffer, timeout_ms), f"Esperar imagen ({timeout_ms} ms)")
            self._waiting_first_frame = False
            captured_at_s = time.time()
            try:
                lost_packets = buffer.stFrameInfo.nLostPacket
                if lost_packets:
                    logger.warning("Imagen descartada: %d paquetes perdidos.", lost_packets)
                    continue
                image = self._to_array(handle, sdk, buffer)
            finally:
                handle.MV_CC_FreeImageBuffer(buffer)
            return Frame(image=image, captured_at_s=captured_at_s)
        raise RuntimeError(
            f"{self._config.max_incomplete_frames} imágenes seguidas con paquetes perdidos: "
            "revisar la red y el MTU."
        )

    def _find_device(self, sdk: ModuleType) -> tuple[Any, _DeviceIdentity]:
        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        _check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE, devices), "MV_CC_EnumDevices")
        found: list[str] = []
        for index in range(devices.nDeviceNum):
            device = ctypes.cast(devices.pDeviceInfo[index], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
            gige = device.SpecialInfo.stGigEInfo
            ip = ".".join(str((gige.nCurrentIp >> shift) & 0xFF) for shift in (24, 16, 8, 0))
            identity = _DeviceIdentity(model=_decode(gige.chModelName), serial=_decode(gige.chSerialNumber))
            if ip == self._config.ip:
                return device, identity
            found.append(f"{identity.model} S/N {identity.serial} ({ip})")
        raise RuntimeError(
            f"No se encontró la cámara en {self._config.ip}. Encontradas: {', '.join(found) or 'ninguna'}."
        )

    def _check_identity(self, identity: _DeviceIdentity) -> None:
        expected = (self._config.expected_model, self._config.expected_serial)
        if (identity.model, identity.serial) != expected:
            raise RuntimeError(
                f"En {self._config.ip} hay una cámara {identity.model} S/N {identity.serial}, y "
                f"cameras/{self._config.camera_id}.toml espera {expected[0]} S/N {expected[1]}. "
                "Si es la cámara correcta, corrija expected_model / expected_serial."
            )

    def _configure(self, handle: Any, sdk: ModuleType) -> None:
        # Si el paquete de la cámara supera el MTU de la tarjeta de red, las imágenes no llegan.
        packet_size = handle.MV_CC_GetOptimalPacketSize()
        if packet_size > 0:
            _warn_if_failed(handle.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size), "GevSCPSPacketSize")
        # Con la CPU cargada se pierden paquetes UDP; el reenvío GigE Vision los recupera.
        _warn_if_failed(handle.MV_GIGE_SetResend(1, 10, 50), "reenvío de paquetes")
        # Adquisición continua sin trigger (el nodo TriggerMode depende del selector).
        for selector in ("AcquisitionStart", "FrameStart"):
            if handle.MV_CC_SetEnumValueByString("TriggerSelector", selector) == 0:
                _warn_if_failed(handle.MV_CC_SetEnumValueByString("TriggerMode", "Off"), "TriggerMode")
        _warn_if_failed(handle.MV_CC_SetEnumValue("TriggerMode", sdk.MV_TRIGGER_MODE_OFF), "TriggerMode")
        _warn_if_failed(handle.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous"), "AcquisitionMode")
        for node, value in self._config.genicam_nodes.items():
            _check(_set_node(handle, node, value), f"Fijar {node} = {value}")

    def _to_array(self, handle: Any, sdk: ModuleType, buffer: Any) -> Image:
        info = buffer.stFrameInfo
        width, height = info.nWidth, info.nHeight
        if info.enPixelType == sdk.PixelType_Gvsp_Mono8:
            pixels = ctypes.cast(buffer.pBufAddr, ctypes.POINTER(ctypes.c_ubyte))
            return np.ctypeslib.as_array(pixels, shape=(height, width)).copy()
        # Cualquier otro formato (Bayer, YUV, Mono12…) lo convierte el SDK a BGR8.
        size = width * height * 3
        converted = (ctypes.c_ubyte * size)()
        parameters = sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
        ctypes.memset(ctypes.byref(parameters), 0, ctypes.sizeof(parameters))
        parameters.nWidth = width
        parameters.nHeight = height
        parameters.pSrcData = buffer.pBufAddr
        parameters.nSrcDataLen = info.nFrameLen
        parameters.enSrcPixelType = info.enPixelType
        parameters.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
        parameters.pDstBuffer = converted
        parameters.nDstBufferSize = size
        _check(handle.MV_CC_ConvertPixelTypeEx(parameters), "Convertir a BGR8")
        return np.frombuffer(converted, dtype=np.uint8, count=size).reshape(height, width, 3).copy()

    def _require_connected(self) -> tuple[Any, ModuleType]:
        if self._handle is None or self._sdk is None:
            raise RuntimeError("La cámara no está conectada.")
        return self._handle, self._sdk


def _load_sdk() -> ModuleType:
    """Importa el wrapper Python de MVS (``MvImport``) que instala el SDK."""
    development_dir = os.environ.get("MVCAM_COMMON_RUNENV")
    if not development_dir:
        raise RuntimeError("No está la variable MVCAM_COMMON_RUNENV: ¿está instalado el SDK MVS?")
    wrapper_dir = str(Path(development_dir) / "Samples" / "Python" / "MvImport")
    if wrapper_dir not in sys.path:
        sys.path.append(wrapper_dir)
    # Desde Python 3.8 las DLL no se buscan en PATH: se registra la carpeta del runtime.
    runtime_dir = Path(MVS_RUNTIME_DIR) / "Win64_x64"
    if runtime_dir.is_dir():
        os.add_dll_directory(str(runtime_dir))
    import MvCameraControl_class as sdk  # type: ignore[import-not-found]  # noqa: PLC0415

    return sdk


def _set_node(handle: Any, node: str, value: float | int | bool | str) -> int:
    if isinstance(value, bool):
        return handle.MV_CC_SetBoolValue(node, value)
    if isinstance(value, int):
        return handle.MV_CC_SetIntValue(node, value)
    if isinstance(value, float):
        return handle.MV_CC_SetFloatValue(node, value)
    return handle.MV_CC_SetEnumValueByString(node, value)


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise MvsError(operation, code)


def _warn_if_failed(code: int, node: str) -> None:
    if code != 0:
        logger.warning("No se pudo configurar %s (código MVS 0x%08X).", node, code & 0xFFFFFFFF)


def _decode(characters: Any) -> str:
    return bytes(characters).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
