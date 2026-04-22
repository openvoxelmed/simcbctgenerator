"""Simplified projector module for CBCT ray casting.

This module provides minimal functionality for ray-based X-ray projection
without scatter simulation, complex attenuation, or spectral processing.
"""

from .core import Projector
from .volume import Volume
from .device import Device, CBCT
from . import geometry as geo

# Export main classes
__all__ = [
    "Projector",
    "Volume", 
    "Device",
    "CBCT",
    "geo",
]