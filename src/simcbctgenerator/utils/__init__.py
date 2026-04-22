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

from .arg_parser import (
    # New argument functions
    add_physics_arguments, add_geometry_arguments, add_reconstruction_volume_arguments,
    # Other argument functions
    add_4d_ct_arguments, add_patient_arguments,
    add_router_arguments, add_phantom_arguments, add_registration_arguments,
    # Config loading
    add_config_argument, merge_config_and_args, load_config, load_config_from_args,
    discover_config_presets, resolve_config_path, list_available_configs
)
from .logger_config import log_config, log_time, setup_logger
from .physics_config import (
    PhysicsConfig, load_physics_config, save_physics_config,
    load_cbct_physics_from_yaml, load_geometry_from_yaml,
    SpectralData, generate_spectral_data
)
from .config import (
    # New config classes
    GeometryConfig, ReconstructionVolumeConfig, CBCTSystemConfig,
    # Other existing config classes
    PatientConfig, MotionConfig, sample_motion_config, PhantomConfig, Vendor,
    ImageType, ImageCenter, ImagingModality, Errors
)
__all__ = [
    # New argument functions
    'add_physics_arguments',
    'add_geometry_arguments',
    'add_reconstruction_volume_arguments',
    # Other argument functions
    'add_4d_ct_arguments',
    'add_patient_arguments',
    'add_router_arguments',
    'add_phantom_arguments',
    'add_registration_arguments',
    # Config loading
    'add_config_argument',
    'merge_config_and_args',
    'load_config',
    'load_config_from_args',
    'discover_config_presets',
    'resolve_config_path',
    'list_available_configs',
    # Logger
    'log_config',
    'log_time',
    'setup_logger',
    # Config classes
    'PhysicsConfig',
    'load_physics_config',
    'save_physics_config',
    'load_cbct_physics_from_yaml',
    'load_geometry_from_yaml',
    'SpectralData',
    'generate_spectral_data',
    'GeometryConfig',
    'ReconstructionVolumeConfig',
    'CBCTSystemConfig',
    'PatientConfig',
    'MotionConfig',
    'sample_motion_config',
    'PhantomConfig',
    'Vendor',
    'ImageType',
    'ImageCenter',
    'ImagingModality',
    'Errors',
]
