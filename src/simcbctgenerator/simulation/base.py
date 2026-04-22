"""Shared simulation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import SimpleITK as sitk

from simcbctgenerator.utils.config import CBCTSystemConfig, MotionConfig


@dataclass
class SimulationResult:
    cbct: sitk.Image
    fov_mask: Optional[sitk.Image]
    system_config: Optional[CBCTSystemConfig] = None
    projections: Optional[sitk.Image] = None
    sampled_motion_config: Optional[MotionConfig] = None


class BaseCBCTSimulator:
    """Minimal simulation backend interface."""

    def run(self, *args, **kwargs) -> SimulationResult:
        raise NotImplementedError
