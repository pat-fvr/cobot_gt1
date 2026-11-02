"""Cámaras: interfaz ``Camera`` e implementaciones (MVS, webcam, carpeta, video)."""

from .base import Camera, NoMoreFramesError

__all__ = ["Camera", "NoMoreFramesError"]
