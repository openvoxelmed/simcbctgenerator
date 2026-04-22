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

"""Configuration classes for registration."""

from pydantic import BaseModel
from pathlib import Path
from typing import Optional


class RegistrationConfig(BaseModel):
    """Configuration for CT-CBCT registration.

    Attributes:
        elastix_binary_path: Path to elastix executable
        parameter_map_path: Path to elastix parameter map file
        model_path: Path to neural network model file for registration (if used)
        models_dest_dir: Destination directory for models
        threads: Number of threads for elastix
        persistent_data_dir: Directory for persistent data (models, parameter maps)
        use_fixed_mask: Whether to use fixed image mask
        use_moving_mask: Whether to use moving image mask
        save_visualizations: Whether to save multi-planar comparison visualizations
        visualization_output_dir: Directory for saving visualizations
    """

    # Elastix settings
    elastix_binary_path: Path = Path("/usr/lib/elastix-install/bin/elastix")
    parameter_map_path: Optional[Path] = None
    threads: int = 24

    # Model settings (for Impact-based registration)
    model_path: Optional[Path] = None
    models_dest_dir: str = "Models/TS"

    # Data directories
    persistent_data_dir: Path = Path("/Data")

    # Mask settings
    use_fixed_mask: bool = False
    use_moving_mask: bool = False

    # Visualization settings
    save_visualizations: bool = True
    visualization_output_dir: Optional[Path] = None

    # Standardization settings
    clip_min: float = -1024.0
    clip_max: float = 276.0
    standardize_mean: float = -370.00039267657144
    standardize_std: float = 436.5998675471528
