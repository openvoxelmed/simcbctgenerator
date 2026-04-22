###############################################################################
# simcbctgenerator
#
# Copyright 2025 Lukas Zimmermann and Michael Rauter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""
Synthetic CBCT Router - Routes between different generation methods
"""

import numpy as np
from pathlib import Path
from typing import Optional
import SimpleITK as sitk
from pydantic import BaseModel

from simcbctgenerator.generate_4d_ct import FourDCTGenerator
from simcbctgenerator.generate_projections import DRRGenerator
from simcbctgenerator.cbct_reconstruction import SyntheticCBCTReconstruction
from simcbctgenerator.phantom_generator import PhantomGenerator
from simcbctgenerator.utils.config import CBCTSystemConfig, MotionConfig, PhantomConfig
from simcbctgenerator.patient import Patient
from simcbctgenerator import utils

logger = utils.setup_logger()

class RouterConfig(BaseModel):
    """Configuration for synthetic CBCT router."""
    method: str = "standard"  # "standard" or "phantom"


class SyntheticCBCTRouter:
    """Router class that handles different synthetic CBCT generation methods."""

    def __init__(self,
                 router_config: RouterConfig,
                 drr_path: Optional[Path] = None,
                 motion_config: Optional[MotionConfig] = None,
                 system_config: Optional[CBCTSystemConfig] = None,
                 phantom_config: Optional[PhantomConfig] = None,
                 gpu: bool = True,
                 motion_enabled = True):

        self.router_config = router_config
        self.method = router_config.method.lower()
        self.patient = None
        self.synthetic_cbct = None
        self.motion_enabled = motion_enabled

        # Initialize generators based on method
        if self.method == "standard":
            if not all([motion_config, system_config, drr_path]):
                raise ValueError("Standard method requires motion_config, system_config, and drr_path")

            self.ct_generator = FourDCTGenerator(motion_config)
            self.drr_generator = DRRGenerator(drr_path, system_config)
            self.cbct_reconstructor = SyntheticCBCTReconstruction(system_config, gpu=gpu)
            self.drr_path = drr_path

            logger.info("Initialized router with standard method (4D CT + DRR + Reconstruction)")

        elif self.method == "phantom":
            if not phantom_config:
                raise ValueError("Phantom method requires phantom_config")

            # Validate that phantom_path is provided
            if not phantom_config.phantom_path:
                raise ValueError("Phantom method requires phantom_path to be specified")

            self.phantom_generator = PhantomGenerator(phantom_config)

            logger.info("Initialized router with phantom method")

        else:
            raise ValueError(f"Unknown method: {self.method}. Choose 'standard' or 'phantom'")

    def initialize(self, patient: Patient):
        """Initialize router with patient data."""
        self.patient = patient

        if self.method == "standard":
            self.ct_generator.initialize(patient)
        elif self.method == "phantom":
            self.phantom_generator.initialize(patient)

        logger.info(f"Initialized router for patient {patient.id} using {self.method} method")

    def generate(self) -> sitk.Image:
        """Generate synthetic CBCT using the selected method."""
        if self.patient is None:
            raise ValueError("Router not initialized. Call initialize() first.")

        logger.info(f"Generating synthetic CBCT using {self.method} method")

        if self.method == "standard":
            return self._generate_standard()
        elif self.method == "phantom":
            return self._generate_phantom()

    def generate_projections(self) -> sitk.Image:
        """Generate synthetic CBCT using the selected method."""
        if self.patient is None:
            raise ValueError("Router not initialized. Call initialize() first.")

        logger.info(f"Generating synthetic CBCT using {self.method} method")

        if self.method == "standard":
            projs = self.drr_generator.generate_all_projections(self.ct_generator.patient, self.ct_generator, return_projections=True)
            recon = self.cbct_reconstructor.reconstruct(self.drr_path)
            self.synthetic_cbct = recon
            return (projs, recon)
        elif self.method == "phantom":
            raise ValueError("Phantom method does not support projection generation")

    def _generate_standard(self) -> sitk.Image:
        """Generate synthetic CBCT using standard method."""
        if self.motion_enabled:
            # Generate 4D CT with motion
            self.drr_generator.generate_all_projections(self.ct_generator.patient, self.ct_generator)

            # Reconstruct CBCT from projections
            recon = self.cbct_reconstructor.reconstruct(self.drr_path)
        else:
            # Generate static 3D CT projections
            self.drr_generator.generate_all_projections(self.ct_generator.patient)

            # Reconstruct CBCT from projections
            recon = self.cbct_reconstructor.reconstruct(self.drr_path)


        self.synthetic_cbct = recon
        return recon

    def _generate_phantom(self) -> sitk.Image:
        """Generate synthetic CBCT using phantom method."""
        self.synthetic_cbct = self.phantom_generator.generate()
        return self.synthetic_cbct

    def save(self, output_path: Path, file_name: str, mask: Optional[np.ndarray] = None) -> None:
        """Save synthetic CBCT to specified path."""
        if self.synthetic_cbct is None:
            raise ValueError("No synthetic CBCT generated. Call generate() first.")

        if self.method == "standard":
            self.cbct_reconstructor.save(self.synthetic_cbct, output_path=output_path, file_name=file_name)
        elif self.method == "phantom":
            self.phantom_generator.save(output_path, file_name)

        logger.info(f"Saved synthetic CBCT using {self.method} method")

    def get_fov_mask(self) -> Optional[sitk.Image]:
        """Get field of view mask if available."""
        if self.method == "standard":
            return getattr(self.cbct_reconstructor, 'fov_img', None)
        elif self.method == "phantom":
            if self.synthetic_cbct is not None:
                return self.phantom_generator.fov_mask
            return None
        return None

    def cleanup(self):
        """Clean up temporary files and reset state."""
        if self.method == "standard":
            # Delete projections
            if hasattr(self.drr_generator, 'delete_projections'):
                self.drr_generator.delete_projections()

        elif self.method == "phantom":
            # Reset phantom generator
            if hasattr(self.phantom_generator, 'reset'):
                self.phantom_generator.reset()

        # Reset router state
        self.patient = None
        self.synthetic_cbct = None

        logger.debug(f"Cleaned up {self.method} method resources")

    def get_method(self) -> str:
        """Get the current generation method."""
        return self.method

    def get_patient(self) -> Optional[Patient]:
        """Get the current patient."""
        return self.patient

    def get_synthetic_cbct(self) -> Optional[sitk.Image]:
        """Get the generated synthetic CBCT."""
        return self.synthetic_cbct
