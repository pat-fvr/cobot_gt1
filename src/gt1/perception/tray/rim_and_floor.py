"""Bandejas por el pliegue del fondo en los lados en sombra y el labio del borde en los otros.

La bandeja PET es transparente: casi no contrasta con la mesa y lo que se ve a
través de ella cambia con la mesa. Lo que se ve depende de hacia dónde mira
cada lado respecto de la luz:

- En los lados cuya normal hacia afuera apunta hacia ``shadow_toward_deg``, la
  pared hace sombra sobre el pliegue donde se dobla hacia el fondo: una línea
  oscura angosta, la evidencia más fuerte de una bandeja (40-50 % con la
  acA1600, en la foto de 4 bandejas del 2026-11-10).
- En los otros lados se ve el labio del borde, a ``depth_mm`` de la mesa: como
  una línea oscura o como un escalón de brillo entre la pared y la mesa. Bajo
  los reflejos del film puede no verse nada.

Cada lado se evalúa solo con la evidencia que le corresponde. Si cada lado
pudiera elegir entre el pliegue y el labio, una misma línea calzaría en dos
poses separadas por la distancia entre ellos (~25 mm) y el centro se correría.

1. Dos vistas cenitales de la zona sobre la misma grilla en mm: a la altura
   de la mesa y a la del borde. En cada una, su rectángulo se ve con sus
   medidas, sin la deformación de la perspectiva.
2. Logaritmo del brillo (una sombra resta una constante y no cambia el
   contraste), sin los reflejos angostos del film (apertura). Las líneas
   oscuras se realzan con un "black hat" del ancho de la línea (el borde de una
   sombra o un cambio de mesa, más anchos, no); el escalón es el gradiente. La
   evidencia se ensancha en ``side_tolerance_mm``: una línea corrida hasta ahí
   del rectángulo ideal cuenta igual.
3. En una vista girada con el lado largo a lo largo de las columnas, cada lado
   puntúa su evidencia media sobre él menos la de dos paralelas, por dentro y
   por fuera: una zona llena de líneas no puntúa.
4. Una pose es candidata si sus cuatro lados están dentro de la imagen (con
   tres, la caja de la botonera pasaba por bandeja), el pliegue de un lado en
   sombra se ve (``min_fold_contrast_pct``) y otro lado más la confirma
   (``min_line_contrast_pct``). Su puntaje es el promedio de los lados.
5. La búsqueda va de lo grueso a lo fino, para que tarde decenas de ms y no
   segundos: todos los ángulos (cada ``coarse_angle_step_deg``) en la grilla
   gruesa (``coarse_grid_mm``, cada celda con la línea más marcada de las que
   junta), y cada pose que resulta se afina a ``grid_mm`` y ``angle_step_deg``
   en una ventana de la bandeja girada con ella, todos los ángulos juntos
   (arreglos de vista, fila, columna). La pose afinada se juzga con los mismos
   criterios en la grilla fina.
6. Las candidatas, de la mejor a la peor, se confirman por lo que encierran:
   el PET vacío se ve casi como la mesa (paredes de -8 a +6 %, interior hasta
   +15 % con la acA1600). El borde de una caja o de una pieza con su sombra da
   líneas como las de una bandeja, pero lo que encierra no: una caja o una
   sombra oscurecen las "paredes" (-13 % o menos) y una pechuga aclara el
   "interior" (+45 %). Cada candidata confirmada es una bandeja.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import RimAndFloorTrayDetectorConfig, TrayConfig
from gt1.domain import Frame, TrayDetection, Zone
from gt1.geometry import FloatArray, wrap_axis_angle_deg
from gt1.perception.image_ops import SOBEL_3_GAIN, to_gray, zone_pixels_per_mm, zone_polygon_px

from .base import TrayDetector, tray_on_table

# Un mapa en la grilla (filas, columnas), o varias vistas giradas apiladas (vistas, filas, columnas).
Grid = NDArray[np.float32]
Flags = NDArray[np.bool_]
# Transformación afín de OpenCV (2, 3): de la grilla a una vista girada.
Affine = NDArray[np.float64]

# Se suma al brillo antes del logaritmo: en lo casi negro el ruido del sensor no cuenta como línea.
DARK_LEVEL_OFFSET = 10.0
# Puntaje de las poses que no son una bandeja (lados fuera de la imagen o sin la evidencia mínima).
NOT_A_TRAY = -1.0
# Una candidata que no se confirma descarta las poses a menos de esto de ella: son el mismo objeto. Es
# también la separación mínima entre las poses que la búsqueda gruesa entrega para afinar.
REJECTED_RADIUS_MM = 20.0
# Posición de los lados largos y de los cortos en ``_side_normals_deg``.
LONG_SIDES = (0, 2)
SHORT_SIDES = (1, 3)
# La proyección de la vista cenital a la imagen es suave: se proyecta un punto cada tantos de la grilla y
# los demás se interpolan (el error es de milésimas de píxel y proyectar todos tardaba ~70 ms por vista).
PROJECTED_EVERY_POINTS = 8


@dataclass(frozen=True)
class _TopView:
    """Vista cenital de la zona a una altura: puntos cada ``grid_mm`` y el píxel de la imagen de cada uno."""

    origin_mm: NDArray[np.float64]  # (2,) punto de la fila 0, columna 0
    grid_mm: float
    pixels: Grid  # (filas, columnas, 2)
    visible: Flags  # el punto cae dentro de la imagen
    in_zone: Flags  # el punto está dentro de la zona
    zone_xy: FloatArray  # (N, 2) vértices de la zona en la grilla: (columna, fila)

    def base_xy(self, row: float, column: float) -> tuple[float, float]:
        x_mm, y_mm = self.origin_mm + self.grid_mm * np.array([column, row])
        return float(x_mm), float(y_mm)

    def in_zone_at(self, row: float, column: float) -> bool:
        rows, columns = self.in_zone.shape
        row_index, column_index = round(row), round(column)
        inside = 0 <= row_index < rows and 0 <= column_index < columns
        return inside and bool(self.in_zone[row_index, column_index])


@dataclass(frozen=True)
class _Evidence:
    """Los mapas de evidencia de una vista (línea oscura o escalón, en %, ya ensanchados en la tolerancia),
    dónde se ven, y el rectángulo de la bandeja que se busca en ellos. Cada lado usa el mapa en que más
    contrasta."""

    maps: tuple[Grid, ...]
    visible: NDArray[np.uint8]  # 1 donde se ve
    length_mm: float
    width_mm: float
    grid_mm: float


@dataclass(frozen=True)
class _Lines:
    """La evidencia en vistas giradas del mismo tamaño (el lado largo a lo largo de las columnas), apiladas
    (vista, fila, columna), con su media a lo largo de los lados largos y de los cortos centrada en cada
    punto."""

    along_columns: tuple[Grid, ...]  # tramos de lado largo, uno por mapa de evidencia
    along_rows: tuple[Grid, ...]  # tramos de lado corto
    visible: NDArray[np.uint8]
    evidence: _Evidence


@dataclass(frozen=True)
class _Region:
    """Un rectángulo de puntos de la vista girada: dónde se calcula el puntaje."""

    top: int
    left: int
    rows: int
    columns: int

    def moved(self, rows: int, columns: int) -> _Region:
        return _Region(self.top + rows, self.left + columns, self.rows, self.columns)


@dataclass(frozen=True)
class _Pose:
    """Una bandeja candidata: centro en la grilla fina (fila, columna, fraccionarios), eje y puntaje."""

    row: float
    column: float
    angle_deg: float
    score: float


class RimAndFloorTrayDetector(TrayDetector):
    def __init__(
        self, config: RimAndFloorTrayDetectorConfig, tray: TrayConfig, projector: PixelProjector
    ) -> None:
        self._config = config
        self._tray = tray
        self._projector = projector

    def detect(self, frame: Frame, zone: Zone) -> list[TrayDetection]:
        floor_view = self._top_view(zone, frame.size_px, zone.surface_z_mm)
        rim_view = self._top_view(zone, frame.size_px, zone.surface_z_mm + self._tray.depth_mm)
        brightness = self._smoothed_brightness(frame, zone)
        floor_log = self._log_brightness(brightness, floor_view)
        rim_log = self._log_brightness(brightness, rim_view)
        tray = self._tray
        fold = self._evidence(
            (self._dark_lines(floor_log),), floor_view, tray.floor_length_mm, tray.floor_width_mm
        )
        rim = self._evidence(
            (self._dark_lines(rim_log), _steps(rim_log)), rim_view, tray.rim_length_mm, tray.rim_width_mm
        )
        coarse_poses = self._coarse_poses(fold, rim, floor_view.zone_xy)
        refined = (self._refined(fold, rim, pose) for pose in coarse_poses)
        poses = sorted((pose for pose in refined if pose is not None), key=lambda pose: -pose.score)
        body = _TrayBody(
            log_brightness=floor_log,
            view=floor_view,
            tray=self._tray,
            projector=self._projector,
            table_z_mm=zone.surface_z_mm,
            config=self._config,
        )
        return self._confirmed(poses, floor_view, body, zone)

    def _confirmed(
        self, poses: list[_Pose], view: _TopView, body: _TrayBody, zone: Zone
    ) -> list[TrayDetection]:
        """Las candidatas con el centro en la zona que encierran PET vacío, de la mejor a la peor."""
        trays: list[TrayDetection] = []
        rejected: list[tuple[float, float]] = []
        for pose in poses:
            if not view.in_zone_at(pose.row, pose.column):
                continue
            center_mm = view.base_xy(pose.row, pose.column)
            if _near(center_mm, [tray.center_mm[:2] for tray in trays], self._tray.rim_width_mm):
                continue  # dos bandejas no se traslapan: es la misma, con menos puntaje
            if _near(center_mm, rejected, REJECTED_RADIUS_MM):
                continue
            angle = wrap_axis_angle_deg(pose.angle_deg)
            wall_pct, floor_pct = body.brightness_vs_table_pct(center_mm, angle)
            if not body.looks_empty(wall_pct, floor_pct):
                rejected.append(center_mm)
                continue
            measurements = {
                "line_contrast_pct": round(pose.score, 1),
                "wall_vs_table_pct": round(wall_pct, 1),
                "floor_vs_table_pct": round(floor_pct, 1),
            }
            trays.append(tray_on_table(center_mm, angle, zone.surface_z_mm, self._tray, measurements))
        return trays

    def _top_view(self, zone: Zone, size_px: tuple[int, int], height_z_mm: float) -> _TopView:
        grid_mm = self._config.grid_mm
        origin_mm = zone.polygon_mm.min(axis=0)
        extent_mm = zone.polygon_mm.max(axis=0) - origin_mm
        columns, rows = (int(np.ceil(extent / grid_mm)) + 1 for extent in extent_mm)
        pixels = self._grid_pixels(origin_mm, (rows, columns), height_z_mm)
        width, height = size_px
        visible = (pixels[..., 0] >= 0) & (pixels[..., 0] <= width - 1)
        visible &= (pixels[..., 1] >= 0) & (pixels[..., 1] <= height - 1)
        zone_xy = (zone.polygon_mm - origin_mm) / grid_mm
        in_zone = np.zeros((rows, columns), dtype=np.uint8)
        cv2.fillPoly(in_zone, [np.round(zone_xy).astype(np.int32).reshape(-1, 1, 2)], 1)
        return _TopView(origin_mm, grid_mm, pixels, visible, in_zone.astype(bool), zone_xy)

    def _grid_pixels(
        self, origin_mm: NDArray[np.float64], shape: tuple[int, int], height_z_mm: float
    ) -> Grid:
        """El píxel (filas, columnas, 2) de cada punto de la grilla, interpolado entre los proyectados."""
        step, grid_mm = PROJECTED_EVERY_POINTS, self._config.grid_mm
        projected_shape = tuple(int(np.ceil((size - 1) / step)) + 1 for size in shape)
        row_index, column_index = np.indices(projected_shape)
        points_mm = np.column_stack(
            [
                origin_mm[0] + grid_mm * step * column_index.ravel(),
                origin_mm[1] + grid_mm * step * row_index.ravel(),
                np.full(row_index.size, height_z_mm),
            ]
        )
        projected = self._projector.base_to_pixels(points_mm).astype(np.float32)
        projected = projected.reshape(*projected_shape, 2)
        row_in_projected, column_in_projected = np.indices(shape, dtype=np.float32) / np.float32(step)
        return _as_grid(cv2.remap(projected, column_in_projected, row_in_projected, cv2.INTER_LINEAR))

    def _smoothed_brightness(self, frame: Frame, zone: Zone) -> Grid:
        """El brillo de la imagen suavizado a la escala de la grilla: si no, un reflejo de un solo píxel
        decide un punto."""
        pixels_per_mm = zone_pixels_per_mm(zone, zone_polygon_px(zone, self._projector))
        sigma_px = 0.5 * self._config.grid_mm * pixels_per_mm
        return _as_grid(cv2.GaussianBlur(to_gray(frame.image).astype(np.float32), (0, 0), sigma_px))

    def _log_brightness(self, brightness: Grid, view: _TopView) -> Grid:
        """Logaritmo del brillo en la vista cenital, sin los reflejos angostos del film."""
        top = cv2.remap(
            brightness,
            view.pixels[..., 0],
            view.pixels[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        log_brightness = np.log(top + np.float32(DARK_LEVEL_OFFSET)).astype(np.float32)
        # Entre los reflejos del film quedan huecos oscuros y angostos, como líneas: se borran los reflejos.
        without_glints = cv2.morphologyEx(
            log_brightness, cv2.MORPH_OPEN, _disk(self._config.glint_width_mm, view.grid_mm)
        )
        return _as_grid(without_glints)

    def _dark_lines(self, log_brightness: Grid) -> Grid:
        """Cuánto más oscuro que su entorno es cada punto, en %, si es angosto."""
        dark_line = _disk(self._config.dark_line_width_mm, self._config.grid_mm)
        return 100.0 * _as_grid(cv2.morphologyEx(log_brightness, cv2.MORPH_BLACKHAT, dark_line))

    def _evidence(
        self, maps: tuple[Grid, ...], view: _TopView, length_mm: float, width_mm: float
    ) -> _Evidence:
        """La evidencia de una vista, cero fuera de la imagen y ensanchada en la tolerancia."""
        tolerance = _disk(2.0 * self._config.side_tolerance_mm + view.grid_mm, view.grid_mm)
        widened = tuple(_as_grid(cv2.dilate(values * view.visible, tolerance)) for values in maps)
        return _Evidence(widened, view.visible.astype(np.uint8), length_mm, width_mm, view.grid_mm)

    def _coarse_poses(self, fold: _Evidence, rim: _Evidence, zone_xy: FloatArray) -> list[_Pose]:
        """Las poses que pasan los criterios en la grilla gruesa, a ``REJECTED_RADIUS_MM`` entre sí.

        Cada vista girada encierra solo la zona y, alrededor, lo que ocupan los lados de una bandeja
        centrada en ella: la mitad de la grilla que no es zona no se recorre en cada ángulo.
        """
        factor = round(self._config.coarse_grid_mm / self._config.grid_mm)
        coarse_fold, coarse_rim = _max_pooled(fold, factor), _max_pooled(rim, factor)
        coarse_zone_xy = (zone_xy - (factor - 1) / 2.0) / factor
        margin_px = self._sides_reach_px(coarse_rim)
        poses: list[_Pose] = []
        for angle_deg in np.arange(0.0, 180.0, self._config.coarse_angle_step_deg):
            to_view, size, region = _rotation_onto(coarse_zone_xy, float(angle_deg), margin_px)
            angles_deg = np.array([angle_deg])
            fold_lines = self._lines(coarse_fold, [to_view], size)
            rim_lines = self._lines(coarse_rim, [to_view], size)
            score = self._score(fold_lines, rim_lines, angles_deg, region)[0]
            back = cv2.invertAffineTransform(to_view)
            for row_in_region, column_in_region in _local_maxima(score):
                view_point = [region.left + column_in_region, region.top + row_in_region, 1.0]
                column, row = back @ np.array(view_point)
                # El centro de la celda gruesa, en la grilla fina.
                center = factor * row + (factor - 1) / 2.0, factor * column + (factor - 1) / 2.0
                poses.append(_Pose(*center, float(angle_deg), float(score[row_in_region, column_in_region])))
        return _apart(poses, REJECTED_RADIUS_MM / fold.grid_mm)

    def _refined(self, fold: _Evidence, rim: _Evidence, coarse: _Pose) -> _Pose | None:
        """La mejor pose en la grilla fina a una celda gruesa y a medio paso grueso de la pose gruesa."""
        config = self._config
        search_px = round(config.coarse_grid_mm / config.grid_mm)
        angle_steps = int(np.ceil(0.5 * config.coarse_angle_step_deg / config.angle_step_deg))
        steps = np.arange(-angle_steps, angle_steps + 1, dtype=np.float64)
        angles_deg = coarse.angle_deg + config.angle_step_deg * steps
        half_size_px = self._window_half_size_px(rim, search_px)
        to_views = [
            _rotation_around((coarse.row, coarse.column), float(angle), half_size_px) for angle in angles_deg
        ]
        size = (2 * half_size_px[0] + 1, 2 * half_size_px[1] + 1)
        search = 2 * search_px + 1
        region = _Region(half_size_px[1] - search_px, half_size_px[0] - search_px, search, search)
        score = self._score(
            self._lines(fold, to_views, size), self._lines(rim, to_views, size), angles_deg, region
        )
        view, row_in_region, column_in_region = np.unravel_index(int(np.argmax(score)), score.shape)
        best_score = float(score[view, row_in_region, column_in_region])
        if best_score <= NOT_A_TRAY:
            return None
        view_point = [region.left + column_in_region, region.top + row_in_region, 1.0]
        column, row = cv2.invertAffineTransform(to_views[view]) @ np.array(view_point)
        return _Pose(float(row), float(column), float(angles_deg[view]), best_score)

    def _window_half_size_px(self, rim: _Evidence, search_px: int) -> tuple[int, int]:
        """Media ventana (columnas, filas) que contiene los lados de la bandeja (su borde, el rectángulo más
        grande) corrida hasta ``search_px``, con sus paralelas."""
        reach_columns, reach_rows = self._sides_reach_px(rim)
        return reach_columns + search_px, reach_rows + search_px

    def _sides_reach_px(self, rim: _Evidence) -> tuple[int, int]:
        """Hasta cuántas columnas y filas del centro se lee la evidencia de una bandeja (su borde, el
        rectángulo más grande) en la vista girada: la paralela de afuera de un lado o el extremo del tramo
        del lado perpendicular."""
        neighbor_px = self._neighbor_px(rim.grid_mm)
        half_length_px, half_width_px = (
            round(size / 2.0 / rim.grid_mm) for size in (rim.length_mm, rim.width_mm)
        )
        long_reach_px, short_reach_px = (
            self._side_px(rim, index) // 2 for index in (LONG_SIDES[0], SHORT_SIDES[0])
        )
        return (
            max(half_length_px + neighbor_px, long_reach_px) + 1,
            max(half_width_px + neighbor_px, short_reach_px) + 1,
        )

    def _lines(self, evidence: _Evidence, to_views: Sequence[Affine], size: tuple[int, int]) -> _Lines:
        """La evidencia girada a cada vista y su media a lo largo de los lados largos y de los cortos."""

        def warped(values: Grid | NDArray[np.uint8], interpolation: int) -> NDArray[np.generic]:
            return np.stack(
                [cv2.warpAffine(values, to_view, size, flags=interpolation) for to_view in to_views]
            )

        maps = tuple(_as_grid(warped(values, cv2.INTER_LINEAR)) for values in evidence.maps)
        long_px, short_px = (self._side_px(evidence, index) for index in (LONG_SIDES[0], SHORT_SIDES[0]))
        return _Lines(
            along_columns=tuple(_box_filtered(views, (long_px, 1)) for views in maps),
            along_rows=tuple(_box_filtered(views, (1, short_px)) for views in maps),
            visible=np.asarray(warped(evidence.visible, cv2.INTER_NEAREST), np.uint8),
            evidence=evidence,
        )

    def _score(self, fold: _Lines, rim: _Lines, angles_deg: FloatArray, region: _Region) -> Grid:
        """Puntaje de una bandeja centrada en cada punto de la región de cada vista, con el lado largo al
        ángulo de su vista."""
        config = self._config
        contrasts: list[Grid] = []
        all_seen: Flags | None = None
        strongest_fold: Grid | None = None
        for index, normals_deg in enumerate(_side_normals_deg(angles_deg)):
            faces_the_shadow = np.cos(np.radians(normals_deg - config.shadow_toward_deg)) > 0.0
            contrast, seen = self._side_contrast_in_its_light(fold, rim, index, region, faces_the_shadow)
            contrasts.append(contrast)
            all_seen = seen if all_seen is None else all_seen & seen
            fold_contrast = np.where(faces_the_shadow[:, None, None], contrast, np.float32(-np.inf))
            strongest_fold = (
                fold_contrast if strongest_fold is None else np.maximum(strongest_fold, fold_contrast)
            )
        assert all_seen is not None and strongest_fold is not None
        best_long_side = np.maximum(contrasts[LONG_SIDES[0]], contrasts[LONG_SIDES[1]])
        best_short_side = np.maximum(contrasts[SHORT_SIDES[0]], contrasts[SHORT_SIDES[1]])
        is_tray = all_seen & (strongest_fold >= config.min_fold_contrast_pct)
        is_tray &= np.minimum(best_long_side, best_short_side) >= config.min_line_contrast_pct
        mean = 0.25 * functools.reduce(np.add, contrasts)
        return np.where(is_tray, mean, np.float32(NOT_A_TRAY)).astype(np.float32)

    def _side_contrast_in_its_light(
        self, fold: _Lines, rim: _Lines, index: int, region: _Region, faces_the_shadow: Flags
    ) -> tuple[Grid, Flags]:
        """El contraste del lado ``index`` con la evidencia que le corresponde en cada vista: el pliegue si
        mira hacia la sombra, el borde si no."""
        if faces_the_shadow.all():
            return self._side_contrast(fold, index, region)
        if not faces_the_shadow.any():
            return self._side_contrast(rim, index, region)
        fold_contrast, fold_seen = self._side_contrast(fold, index, region)
        rim_contrast, rim_seen = self._side_contrast(rim, index, region)
        in_shadow = faces_the_shadow[:, None, None]
        return np.where(in_shadow, fold_contrast, rim_contrast), np.where(in_shadow, fold_seen, rim_seen)

    def _side_contrast(self, lines: _Lines, index: int, region: _Region) -> tuple[Grid, Flags]:
        """Cuánto más marcado que sus paralelas es el lado ``index`` de la bandeja centrada en cada punto de
        la región, en el mapa en que más lo es; y si los extremos de la paralela de afuera se ven (la imagen
        vista en la mesa es convexa: entonces se ve todo el lado).

        En la vista girada los lados largos van a lo largo de las columnas, corridos en filas del centro; los
        cortos, a lo largo de las filas, corridos en columnas.
        """
        evidence = lines.evidence
        is_long = index in LONG_SIDES
        half_across_mm = (evidence.width_mm if is_long else evidence.length_mm) / 2.0
        outward = 1 if index < 2 else -1
        offset_px = outward * round(half_across_mm / evidence.grid_mm)
        neighbor_px = outward * self._neighbor_px(evidence.grid_mm)
        half_side_px = self._side_px(evidence, index) // 2

        def moved(across_px: int, along_px: int = 0) -> _Region:
            return region.moved(across_px, along_px) if is_long else region.moved(along_px, across_px)

        contrast = functools.reduce(
            np.maximum,
            (
                _cropped(means, moved(offset_px))
                - 0.5
                * (
                    _cropped(means, moved(offset_px - neighbor_px))
                    + _cropped(means, moved(offset_px + neighbor_px))
                )
                for means in (lines.along_columns if is_long else lines.along_rows)
            ),
        )
        outer_px = offset_px + neighbor_px
        seen = _cropped(lines.visible, moved(outer_px, -half_side_px)) & _cropped(
            lines.visible, moved(outer_px, half_side_px)
        )
        return _as_grid(contrast), seen.astype(bool)

    def _side_px(self, evidence: _Evidence, index: int) -> int:
        """Largo del tramo central del lado ``index`` que se mide, en puntos de la grilla (impar)."""
        side_mm = evidence.length_mm if index in LONG_SIDES else evidence.width_mm
        return _odd(self._config.side_length_fraction * side_mm / evidence.grid_mm)

    def _neighbor_px(self, grid_mm: float) -> int:
        """Distancia de cada lado a sus paralelas: más allá de la tolerancia en que la línea se ensanchó, al
        menos ``neighbor_offset_mm``."""
        config = self._config
        return int(np.ceil((config.side_tolerance_mm + config.neighbor_offset_mm) / grid_mm - 1e-9))


class _TrayBody:
    """Lo que encierra una bandeja en la vista cenital de la mesa, comparado con la mesa alrededor."""

    def __init__(
        self,
        *,
        log_brightness: Grid,
        view: _TopView,
        tray: TrayConfig,
        projector: PixelProjector,
        table_z_mm: float,
        config: RimAndFloorTrayDetectorConfig,
    ) -> None:
        self._log_brightness = log_brightness
        self._view = view
        self._tray = tray
        self._projector = projector
        self._table_z_mm = table_z_mm
        self._config = config

    def brightness_vs_table_pct(
        self, center_mm: tuple[float, float], angle_deg: float
    ) -> tuple[float, float]:
        """Cuánto más claras que la mesa alrededor son las paredes y el interior de la bandeja, en %.

        Todo en la vista de la mesa: el borde, a ``depth_mm`` de ella, se lleva a donde la cámara lo ve
        sobre la mesa. El interior es el fondo sin sus pliegues; la mesa, una franja por fuera del borde.
        """
        tray, margin_mm = self._tray, self._config.neighbor_offset_mm
        rim = self._as_seen_on_table(
            _rectangle_mm(center_mm, angle_deg, tray.rim_length_mm, tray.rim_width_mm)
        )
        floor = _rectangle_mm(
            center_mm,
            angle_deg,
            tray.floor_length_mm - 2.0 * margin_mm,
            tray.floor_width_mm - 2.0 * margin_mm,
        )
        around = _rectangle_mm(
            center_mm, angle_deg, tray.rim_length_mm + 4.0 * margin_mm, tray.rim_width_mm + 4.0 * margin_mm
        )
        rim_mask, floor_mask, around_mask = (self._mask(polygon) for polygon in (rim, floor, around))
        table = self._median(around_mask & ~rim_mask)
        wall = self._median(rim_mask & ~floor_mask)
        return 100.0 * (wall - table), 100.0 * (self._median(floor_mask) - table)

    def looks_empty(self, wall_pct: float, floor_pct: float) -> bool:
        """Paredes y fondo de PET vacío: ni oscuros como una caja o una sombra, ni claros como una pieza."""
        config = self._config
        return wall_pct >= config.min_wall_vs_table_pct and floor_pct <= config.max_floor_vs_table_pct

    def _as_seen_on_table(self, polygon_mm: FloatArray) -> FloatArray:
        """El polígono a la altura del borde, llevado a la mesa por el rayo de la cámara."""
        rim_z_mm = self._table_z_mm + self._tray.depth_mm
        on_rim = np.column_stack([polygon_mm, np.full(len(polygon_mm), rim_z_mm)])
        pixels = self._projector.base_to_pixels(on_rim)
        return self._projector.pixels_to_base(pixels, self._table_z_mm)[:, :2]

    def _mask(self, polygon_mm: FloatArray) -> Flags:
        view = self._view
        polygon_px = np.round((polygon_mm - view.origin_mm) / view.grid_mm).astype(np.int32)
        mask = np.zeros(view.in_zone.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_px.reshape(-1, 1, 2)], 1)
        return mask.astype(bool) & view.visible

    def _median(self, mask: Flags) -> float:
        return float(np.median(self._log_brightness[mask])) if mask.any() else float("nan")


def _near(point_mm: tuple[float, float], others_mm: Sequence[ArrayLike], distance_mm: float) -> bool:
    """El punto está a menos de ``distance_mm`` de alguno de los otros."""
    point = np.asarray(point_mm, dtype=np.float64)
    return any(float(np.linalg.norm(point - np.asarray(other))) < distance_mm for other in others_mm)


def _apart(poses: list[_Pose], radius_px: float) -> list[_Pose]:
    """Las poses de la mejor a la peor, sin las que están a menos de ``radius_px`` de una mejor."""
    kept: list[_Pose] = []
    for pose in sorted(poses, key=lambda pose: -pose.score):
        if not _near((pose.row, pose.column), [(other.row, other.column) for other in kept], radius_px):
            kept.append(pose)
    return kept


def _local_maxima(score: Grid) -> list[tuple[int, int]]:
    """Los puntos que son bandeja y no tienen un vecino con más puntaje."""
    neighborhood_max = cv2.dilate(score, np.ones((3, 3), np.uint8))
    peaks = np.argwhere((score > NOT_A_TRAY) & (score >= neighborhood_max))
    return [(int(row), int(column)) for row, column in peaks]


def _rectangle_mm(
    center_mm: tuple[float, float], angle_deg: float, length_mm: float, width_mm: float
) -> FloatArray:
    """Esquinas (4, 2) de un rectángulo centrado en ``center_mm`` con el largo a ``angle_deg``."""
    angle = np.radians(angle_deg)
    along = np.array([np.cos(angle), np.sin(angle)]) * length_mm / 2.0
    across = np.array([-np.sin(angle), np.cos(angle)]) * width_mm / 2.0
    center = np.asarray(center_mm, dtype=np.float64)
    return np.array([center + a * along + b * across for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _side_normals_deg(angle_deg: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Normales hacia afuera (en la base) de los lados 0-3 con el lado largo a ``angle_deg``.

    En la vista girada el lado largo queda a lo largo de las columnas: correrse en filas es moverse a
    lo ancho de la bandeja (a ``angle_deg + 90``) y en columnas, a lo largo (a ``angle_deg``). Los lados
    0 y 1 están hacia las filas y columnas crecientes; el 2 y el 3, hacia las decrecientes.
    """
    return (angle_deg + 90.0, angle_deg, angle_deg - 90.0, angle_deg + 180.0)


def _steps(log_brightness: Grid) -> Grid:
    """Escalón de brillo en cada punto: cuánto cambia, en %, entre los puntos de la grilla a sus lados."""
    gradient_x = cv2.Sobel(log_brightness, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(log_brightness, cv2.CV_32F, 0, 1, ksize=3)
    per_point = np.hypot(gradient_x, gradient_y) / SOBEL_3_GAIN
    return (100.0 * 2.0 * per_point).astype(np.float32)


def _max_pooled(evidence: _Evidence, factor: int) -> _Evidence:
    """La evidencia en una grilla ``factor`` veces más gruesa: cada celda con lo más marcado que tiene (una
    línea angosta no se pierde) y visible solo si se ven todos sus puntos."""
    block = np.ones((factor, factor), np.uint8)

    def pooled(values: NDArray[np.float32] | NDArray[np.uint8], operation: int) -> NDArray[np.generic]:
        # Con el ancla en (0, 0), cada celda junta los puntos de la grilla fina desde el suyo.
        return np.asarray(cv2.morphologyEx(values, operation, block, anchor=(0, 0))[::factor, ::factor])

    return replace(
        evidence,
        maps=tuple(_as_grid(pooled(values, cv2.MORPH_DILATE)) for values in evidence.maps),
        visible=np.asarray(pooled(evidence.visible, cv2.MORPH_ERODE), np.uint8),
        grid_mm=evidence.grid_mm * factor,
    )


def _rotation_onto(
    zone_xy: FloatArray, angle_deg: float, margin_px: tuple[int, int]
) -> tuple[Affine, tuple[int, int], _Region]:
    """El giro de la grilla en ``angle_deg`` a una vista que encierra la zona con ``margin_px`` (columnas,
    filas) alrededor: la transformación, el tamaño (ancho, alto) de la vista y la región de la zona en
    ella."""
    to_view = np.asarray(cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0), np.float64)
    rotated = zone_xy @ to_view[:, :2].T
    low, high = np.floor(rotated.min(axis=0)), np.ceil(rotated.max(axis=0))
    margin = np.array(margin_px, dtype=np.float64)
    to_view[:, 2] = margin - low
    columns, rows = (int(size) + 1 for size in high - low)
    margin_columns, margin_rows = margin_px
    size = (columns + 2 * margin_columns, rows + 2 * margin_rows)
    return to_view, size, _Region(margin_rows, margin_columns, rows, columns)


def _rotation_around(
    center_row_column: tuple[float, float], angle_deg: float, half_size_px: tuple[int, int]
) -> Affine:
    """El giro en ``angle_deg`` sobre el punto, que lo deja en el centro de una ventana de media ventana
    ``half_size_px`` (columnas, filas)."""
    row, column = center_row_column
    to_view = np.asarray(cv2.getRotationMatrix2D((column, row), angle_deg, 1.0), np.float64)
    to_view[:, 2] += (half_size_px[0] - column, half_size_px[1] - row)
    return to_view


def _cropped[Value: np.generic](values: NDArray[Value], region: _Region) -> NDArray[Value]:
    """Los valores de la región en cada vista (sin copiarlos si cae entera en la vista); cero donde cae
    fuera."""
    rows, columns = values.shape[-2:]
    top, left = max(region.top, 0), max(region.left, 0)
    bottom, right = min(region.top + region.rows, rows), min(region.left + region.columns, columns)
    if (top, left, bottom, right) == (
        region.top,
        region.left,
        region.top + region.rows,
        region.left + region.columns,
    ):
        return values[..., top:bottom, left:right]
    cropped = np.zeros((*values.shape[:-2], region.rows, region.columns), dtype=values.dtype)
    if top < bottom and left < right:
        cropped[..., top - region.top : bottom - region.top, left - region.left : right - region.left] = (
            values[..., top:bottom, left:right]
        )
    return cropped


def _box_filtered(views: Grid, size: tuple[int, int]) -> Grid:
    """La media en una caja de ``size`` (ancho, alto) en cada vista, sin mezclar vistas."""
    filtered: list[Grid] = []
    for index in range(len(views)):
        view: Grid = views[index]
        filtered.append(_as_grid(cv2.boxFilter(view, -1, size)))
    return _as_grid(np.stack(filtered))


def _disk(diameter_mm: float, grid_mm: float) -> NDArray[np.uint8]:
    """Disco de ``diameter_mm`` en la grilla de la vista cenital."""
    size_px = _odd(diameter_mm / grid_mm)
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_px, size_px)), dtype=np.uint8)


def _as_grid(values: object) -> Grid:
    """Lo que devuelve OpenCV, como grilla float32."""
    return np.asarray(values, dtype=np.float32)


def _odd(size: float) -> int:
    """Tamaño de kernel impar más cercano, al menos 1."""
    return max(round(size) // 2 * 2 + 1, 1)
