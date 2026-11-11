"""Qué pieza va a qué bandeja: en orden a lo largo de una dirección, una pieza por bandeja.

Las piezas y las bandejas se ordenan a lo largo de ``order_toward_deg`` (p. ej.
de izquierda a derecha en la imagen) y la primera pieza va a la primera
bandeja, la segunda a la segunda, y así: el robot avanza siempre en el mismo
sentido y los recorridos no se cruzan. Si sobran piezas, se quedan las
últimas; si sobran bandejas, las últimas quedan vacías.
"""

from __future__ import annotations

from collections.abc import Sequence

from gt1.domain import Piece, TrayDetection
from gt1.geometry import position_along_mm


def pair_pieces_with_trays(
    pieces: Sequence[Piece], trays: Sequence[TrayDetection], order_toward_deg: float
) -> list[tuple[Piece, TrayDetection]]:
    """Pares (pieza, bandeja) en el orden de carga, sin repetir bandeja."""
    ordered_pieces = sorted(pieces, key=lambda piece: position_along_mm(piece.center_mm, order_toward_deg))
    ordered_trays = sorted(trays, key=lambda tray: position_along_mm(tray.center_mm, order_toward_deg))
    # Sin strict: lo que sobra de la lista más larga se queda sin carga.
    return list(zip(ordered_pieces, ordered_trays, strict=False))
