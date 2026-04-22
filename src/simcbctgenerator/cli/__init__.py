"""Command-line entrypoint modules."""

from .reconstruction import main as reconstruction_main, pipeline as reconstruction_pipeline
from .regression import main as regression_main, pipeline as regression_pipeline
from .segmentation import main as segmentation_main, pipeline as segmentation_pipeline

__all__ = [
    "segmentation_main",
    "segmentation_pipeline",
    "reconstruction_main",
    "reconstruction_pipeline",
    "regression_main",
    "regression_pipeline",
]
