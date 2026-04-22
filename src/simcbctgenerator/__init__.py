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

"""SimCBCTGenerator - Framework for generating simulated CBCTs with realistic noise and motion."""

__version__ = "0.2.0"

from .patient import Patient
from .generate_4d_ct import FourDCTGenerator
from .generate_projections import DRRGenerator
from .cbct_reconstruction import SyntheticCBCTReconstruction
from .phantom_generator import PhantomGenerator
from .utils.config import MotionConfig, CBCTSystemConfig, GeometryConfig, PhysicsConfig, ReconstructionVolumeConfig, PatientConfig, PhantomConfig, Vendor
from .simulation import BaseCBCTSimulator, PhantomCBCTSimulator, SimulationResult, StandardCBCTSimulator
from .api import PhantomPipeline, ProjectionPipeline, SegmentationPipeline, RegressionPipeline

__all__ = [
    "__version__",
    "Patient",
    "PatientConfig",
    "FourDCTGenerator",
    "DRRGenerator",
    "CBCTSystemConfig",
    "PhysicsConfig",
    "GeometryConfig",
    "ReconstructionVolumeConfig",
    "SyntheticCBCTReconstruction",
    "PhantomGenerator",
    "PhantomConfig",
    "Vendor",
    "BaseCBCTSimulator",
    "SimulationResult",
    "StandardCBCTSimulator",
    "PhantomCBCTSimulator",
    "MotionConfig",
    "ProjectionPipeline",
    "PhantomPipeline",
    "SegmentationPipeline",
    "RegressionPipeline",
]
