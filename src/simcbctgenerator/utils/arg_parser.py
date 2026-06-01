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


from configparser import ConfigParser
from pathlib import Path
from typing import Dict, Optional
import json
import sys
import difflib


def auto_convert(value):
    """
    Automatically converts a string value to the appropriate Python type.
    """
    value = value.strip()  # Remove leading/trailing whitespace

    # Try to convert to boolean
    if value.lower() in ('true', 'false', 'yes', 'no'):
        return value.lower() in ('true', 'yes')  # True for 'true' and 'yes', False for 'false' and 'no'

    # Try to convert to integer
    if value.isdigit():
        return int(value)

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    # Try to convert to list (comma-separated values)
    if ',' in value:
        return [auto_convert(item) for item in value.split(',') if item != ""]

    # Try to convert to dictionary or list (JSON format)
    if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Return as string if no conversion is possible
    return value


def get_config_dir() -> Path:
    """Get the path to the framework's config directory."""
    # This file is in src/simcbctgenerator/utils/arg_parser.py
    # Config dir is in src/simcbctgenerator/configs/
    return Path(__file__).parent.parent / 'configs'


def discover_config_presets() -> Dict[str, Path]:
    """Auto-discover all .ini config files and create preset name mappings.

    Returns:
        Dictionary mapping preset names to config file paths.

    Example mappings:
        'standard' -> 'standard_config.ini'
        'standard-synrad' -> 'standard_synthrad_config.ini'
        'regression' -> 'regression_config.ini'
        'phantom' -> 'config_phantom.ini'
    """
    config_dir = get_config_dir()
    presets = {}

    if not config_dir.exists():
        return presets

    for config_file in config_dir.glob('*.ini'):
        # Get the filename without extension
        name = config_file.stem

        # Create preset name by removing '_config' suffix and replacing underscores with dashes
        preset_name = name.replace('_config', '').replace('config_', '').replace('_', '-')

        # Store both the clean name and the full stem as valid presets
        presets[preset_name] = config_file

        # Also allow using the full filename without extension
        if name != preset_name:
            presets[name] = config_file

    return presets


def resolve_config_path(config_input: str) -> Path:
    """Resolve a config preset name or custom path to an actual file path.

    Args:
        config_input: Either a preset name (e.g., 'standard-synrad') or a file path

    Returns:
        Path to the config file

    Raises:
        FileNotFoundError: If preset doesn't exist or file path is invalid
    """
    # First, check if it's a custom file path
    custom_path = Path(config_input)
    if custom_path.exists() and custom_path.suffix == '.ini':
        return custom_path

    # Try to resolve as a preset name
    presets = discover_config_presets()

    if config_input in presets:
        return presets[config_input]

    # Config not found - provide helpful error message
    available = sorted(presets.keys())
    error_msg = f"Config preset '{config_input}' not found.\n\n"

    # Try to suggest similar names
    suggestions = difflib.get_close_matches(config_input, available, n=3, cutoff=0.6)
    if suggestions:
        error_msg += "Did you mean one of these?\n"
        for suggestion in suggestions:
            error_msg += f"  - {suggestion}\n"
        error_msg += "\n"

    error_msg += "Available config presets:\n"
    for preset_name in available:
        error_msg += f"  - {preset_name}\n"

    error_msg += "\nOr provide a custom path to an .ini file."

    raise FileNotFoundError(error_msg)


def list_available_configs():
    """Print available config presets and exit."""
    presets = discover_config_presets()

    if not presets:
        print("No config presets found.")
        sys.exit(0)

    print("\nAvailable configuration presets:\n")

    # Group by base name for better organization
    grouped = {}
    for name, path in presets.items():
        # Use the shorter name as the display name
        if '-' in name or len(name) < 15:
            base = name.split('-')[0] if '-' in name else name
            if base not in grouped:
                grouped[base] = []
            grouped[base].append((name, path))

    # Display grouped configs
    for base in sorted(grouped.keys()):
        configs = sorted(grouped[base], key=lambda x: x[0])
        for name, path in configs:
            # Read first line of INI to get description if available
            description = ""
            try:
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#') or first_line.startswith(';'):
                        description = first_line.lstrip('#;').strip()
            except (OSError, UnicodeDecodeError):
                description = ""

            if description:
                print(f"  {name:25s} - {description}")
            else:
                print(f"  {name:25s} - {path.name}")
        print()

    print("Usage: --config <preset-name>")
    print("   Or: --config /path/to/custom.ini\n")
    sys.exit(0)


def load_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    return config


def add_patient_arguments(parser):
    group = parser.add_argument_group('Patient settings')
    group.add_argument('--plan_dir', type=str, help='Directory containing patient plans (ELEKTA format supported)')
    group.add_argument('--ct_dir', type=str, help='Directory containing patient CT scans (ELEKTA format supported)')
    group.add_argument('--cbct_dir', type=str, help='Directory containing patient CBCT scans (ELEKTA format supported)')
    group.add_argument('--export_structures', type=str, nargs='+', help='Directory to export patient structures')
    group.add_argument('--priority', type=int, nargs='+', help='Priority of the patient structures')
    group.add_argument('--cm_mask', type=str, help='mask used for correcting the contrast agent in the bowel.')
    group.add_argument('--use_totalsegmentator', action='store_true', help='Use TotalSegmentator for automatic organ segmentation (default: False)')
    group.add_argument('--image_modality', type=str, choices= ['DUMMY', 'SYNRAD', 'XVI'], help='Key for the image modality in the config file')


def add_4d_ct_arguments(parser):
    group = parser.add_argument_group('Motion CT settings')
    group.add_argument('--motion_type', type=str, choices=['PELVIS', 'THORAX'],
                      help='Motion type for 4D CT generation (PELVIS or THORAX). Motion surrogate organs are automatically inferred: PELVIS→bowel, THORAX→heart/aorta/lung/spine')
    group.add_argument('--amplitude_breathing', type=float, help='Amplitude of the breathing motion for 4D CT generation')
    group.add_argument('--amplitude_heart', type=float, help='Amplitude of the heart motion for 4D CT generation')
    group.add_argument('--phase_offset_breathing', type=float, default=0.0, help='Phase offset of the breathing motion for 4D CT generation (default 0.0)')
    group.add_argument('--phase_offset_heart', type=float, default=0.0, help='Phase offset of the heart motion for 4D CT generation (default 0.0)')
    group.add_argument('--contour_name', type=str, help='(DEPRECATED: Motion surrogate organs now inferred from motion_type) Name of the contour for 4D CT generation')
    group.add_argument('--frequency_breathing', type=float, help='Frequency of the breathing motion for 4D CT generation')
    group.add_argument('--frequency_heartbeat', type=float, help='Frequency of the heartbeat motion for 4D CT generation')
    group.add_argument('--time_per_projection', type=float, help='Time per projection for 4D CT generation')
    group.add_argument('--uncertainty', type=float, help='Uncertainty in the motion for 4D CT generation')
    # Range arguments for random sampling per patient
    group.add_argument('--amplitude_min', type=float, help='Min amplitude for random sampling')
    group.add_argument('--amplitude_max', type=float, help='Max amplitude for random sampling')
    group.add_argument('--frequency_min', type=float, help='Min frequency (breaths/min) for random sampling')
    group.add_argument('--frequency_max', type=float, help='Max frequency (breaths/min) for random sampling')
    group.add_argument('--time_per_projection_min', type=float, help='Min time per projection for random sampling')
    group.add_argument('--time_per_projection_max', type=float, help='Max time per projection for random sampling')
    group.add_argument('--no_motion', action='store_true', help='enable 4d CT motion generation')


def add_physics_arguments(parser):
    """Add X-ray physics and execution parameters for projection generation."""
    group = parser.add_argument_group('Physics and execution settings')
    # X-ray physics parameters
    group.add_argument('--photon_flux', type=int, help='Photon count per pixel per mAs')
    group.add_argument('--spr', type=float, help='Scatter-to-Primary Ratio')
    group.add_argument('--mas', type=float, help='Milliampere-seconds (exposure)')
    group.add_argument('--kv', type=float, help='Kilovoltage peak')
    group.add_argument('--saturation_factor', type=float, help='Detector saturation correction factor')
    group.add_argument('--bp_amplitude', type=float, help='Beam profile amplitude for response estimation')
    group.add_argument('--bp_std', type=float, help='Beam profile standard deviation for response estimation')
    # Polychromatic projection
    group.add_argument('--polychromatic', action='store_true',
                       help='Enable polychromatic spectral projection with beam hardening')
    group.add_argument('--T1', type=float,
                       help='Lower HU threshold for water-bone fuzzy transition (default: 200)')
    group.add_argument('--T2', type=float,
                       help='Upper HU threshold for water-bone fuzzy transition (default: 1500)')
    # Execution parameters
    group.add_argument('--threads', type=int, default=8, help='Number of CUDA threads')
    group.add_argument('--max_block_index', type=int, default=200, help='CUDA block limit')
    # Ablation flags
    group.add_argument('--no_scatter', action='store_true', help='Disable scatter simulation (scatter is enabled by default)')
    group.add_argument('--no_noise', action='store_true', help='Disable Poisson noise simulation (noise is enabled by default)')


def add_geometry_arguments(parser):
    """Add CBCT geometry parameters (detector, C-arm, acquisition angles)."""
    group = parser.add_argument_group('Geometry settings')
    # CBCT geometry
    group.add_argument('--source_origin_distance', type=float, help='Source to isocenter distance (mm)')
    group.add_argument('--source_detector_distance', type=float, help='Source to detector distance (mm)')
    group.add_argument('--detector_offset', type=float, help='Detector lateral offset (mm)')
    # Detector specifications
    group.add_argument('--detector_size_h', type=float, help='Detector height in mm')
    group.add_argument('--detector_size_w', type=float, help='Detector width in mm')
    group.add_argument('--detector_pixels_h', type=int, help='Number of detector pixels vertically')
    group.add_argument('--detector_pixels_w', type=int, help='Number of detector pixels horizontally')
    # Acquisition angles
    group.add_argument('--start_angle', type=float, help='Start gantry angle (degrees, 0-360 convention)')
    group.add_argument('--end_angle', type=float, help='End gantry angle (degrees, 0-360 convention)')
    group.add_argument('--angle_increments', type=float, help='Angle increment between projections')
    # Optional RTK XML override
    group.add_argument('--geometry_xml_path', type=str, help='Path to RTK XML geometry file (overrides all geometry)')


def add_reconstruction_volume_arguments(parser):
    """Add reconstruction volume parameters."""
    group = parser.add_argument_group('Reconstruction volume settings')
    group.add_argument('--recon_size', type=str, help='Reconstruction volume size [x,y,z] in voxels')
    group.add_argument('--recon_origin', type=str, help='Reconstruction volume origin [x,y,z] in mm')
    group.add_argument('--recon_spacing', type=str, help='Reconstruction voxel spacing [x,y,z] in mm')


def add_router_arguments(parser):
    group = parser.add_argument_group('Router settings')
    group.add_argument('--method', type=str, default='standard', choices=['standard', 'phantom'],  help='Synthetic CBCT generation method')


def add_phantom_arguments(parser):
    group = parser.add_argument_group('Rectangular phantom settings')
    group.add_argument('--phantom_path', type=str, help='Path to rectangular phantom MHA file')
    group.add_argument('--intensity_factor', type=float, help='Scaling factor for phantom intensity')
    group.add_argument('--water_threshold', type=float,  help='Threshold above which pixels are considered water')
    group.add_argument('--enhancement_factor', type=float,  help='Factor to enhance phantom artifacts')
    group.add_argument('--lower_threshold', type=float,  help='Threshold below which pixels are excluded')
    group.add_argument('--body_threshold', type=float,  help='Threshold above which pixels are considered body')
    group.add_argument('--gaussian_sigma', type=float, help='Sigma for gaussian smoothing of mask edges')
    group.add_argument('--noise_range_min', type=float, help='Minimum noise range for background')
    group.add_argument('--noise_range_max', type=float, help='Maximum noise range for background')


def add_registration_arguments(parser):
    group = parser.add_argument_group('Registration settings')
    group.add_argument('--elastix_binary_path', type=str, help='Path to elastix binary (inside Docker container)')
    group.add_argument('--parameter_map_path', type=str, help='Path to elastix parameter map file')
    group.add_argument('--model_path', type=str, help='Path to neural network model file for registration')
    group.add_argument('--models_dest_dir', type=str, help='Destination directory for models')
    group.add_argument('--persistent_data_dir', type=str, default="persistent_data", help='Directory for persistent data (models, parameter maps)')
    group.add_argument('--threads', type=int, help='Number of threads for elastix')
    group.add_argument('--use_fixed_mask', type=bool, help='Whether to use fixed image mask')
    group.add_argument('--use_moving_mask', type=bool, help='Whether to use moving image mask')
    group.add_argument('--save_visualizations', type=bool, help='Whether to save multi-planar comparison visualizations')
    group.add_argument('--visualization_output_dir', type=str, help='Directory for saving visualizations')
    group.add_argument('--clip_min', type=float, help='Minimum CT value for clipping')
    group.add_argument('--clip_max', type=float, help='Maximum CT value for clipping')
    group.add_argument('--standardize_mean', type=float, help='Mean for CT standardization')
    group.add_argument('--standardize_std', type=float, help='Standard deviation for CT standardization')


def add_config_argument(parser, default: Optional[str] = None):
    """Add --config argument with preset support to the parser.

    Args:
        parser: ArgumentParser instance
        default: Default config preset name or path (optional)

    This adds both --config (new) and --init (deprecated) arguments.
    """
    help_text = (
        'Config file or preset name. '
        'Use --list-configs to see available presets. '
        'Examples: --config standard-synrad, --config /path/to/custom.ini'
    )

    # Add --config as the primary argument
    parser.add_argument('--config', type=str, default=default,
                       dest='config', help=help_text)

    # Keep --init for backward compatibility (deprecated)
    parser.add_argument('--init', type=str, dest='config',
                       help='(Deprecated: use --config instead) ' + help_text)

    # Add --list-configs flag
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available config presets and exit')


def load_config_from_args(args) -> tuple[ConfigParser, Path]:
    """Load config file from parsed arguments, handling presets and --list-configs.

    Args:
        args: Parsed arguments from ArgumentParser

    Returns:
        Tuple of (loaded ConfigParser instance, Path to config file)

    Raises:
        FileNotFoundError: If config preset or file doesn't exist
    """
    # Handle --list-configs flag
    if hasattr(args, 'list_configs') and args.list_configs:
        list_available_configs()  # This will exit

    # Resolve config path (preset name or custom path)
    if hasattr(args, 'config') and args.config:
        config_path = resolve_config_path(args.config)
    else:
        raise ValueError("No config file specified. Use --config <preset-name> or --config <path>")

    return load_config(str(config_path)), config_path


def merge_config_and_args(config, args):
    """
    Merges configuration from INI file with command-line arguments.
    """
    args_dict = vars(args)
    for subgroup, group in config.items():
        for key, value in group.items():
            converted = auto_convert(value)
            if args_dict[key] is None or (isinstance(args_dict[key], bool) and isinstance(converted, bool)):
                args_dict[key] = converted

    return args_dict
