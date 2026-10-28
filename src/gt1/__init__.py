"""Biblioteca del piloto GT1: carga robotizada de pechuga entera en bandeja.

Las capas, de abajo hacia arriba (cada una solo importa de las de abajo):

    geometry                     matemática pura (numpy)
    domain                       tipos del problema: poses, piezas, caras, planes
    config                       esquemas de los archivos de configuración y su carga
    hardware, calibration,       implementaciones de cada interfaz
    perception, planning
    runtime                      arma los componentes, consola de operación, registro de corridas

Los programas (``gt1_test/``, ``gt1_tools/``) solo usan esta biblioteca.
"""
