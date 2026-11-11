"""Detectores de bandeja: interfaz ``TrayDetector`` e implementaciones; la bandeja apoyada en la mesa."""

from .base import TrayDetector, tray_contains, tray_on_table

__all__ = ["TrayDetector", "tray_contains", "tray_on_table"]
