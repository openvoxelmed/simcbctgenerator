"""Phantom simulation backend."""

from __future__ import annotations

from simcbctgenerator.phantom_generator import PhantomGenerator

from .base import BaseCBCTSimulator, SimulationResult


class PhantomCBCTSimulator(BaseCBCTSimulator):
    """Thin simulator wrapper around PhantomGenerator."""

    def __init__(self, phantom_config):
        self.generator = PhantomGenerator(phantom_config)

    def run(self, patient, *args, **kwargs) -> SimulationResult:
        self.generator.initialize(patient)
        cbct = self.generator.generate()
        return SimulationResult(
            cbct=cbct,
            fov_mask=getattr(self.generator, "fov_mask", None),
            system_config=None,
            projections=None,
            sampled_motion_config=None,
        )
