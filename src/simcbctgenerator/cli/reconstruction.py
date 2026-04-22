#
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
#

"""Pipeline for generating projections from CT and reconstructing CBCT.

This pipeline processes patient data, loading physics and geometry
from YAML metadata, generating projections from CT, reconstructing to CBCT,
and comparing with the real clinical CBCT.
"""
from simcbctgenerator.registration.visualization import save_cbct_comparison
from simcbctgenerator.cli.segmentation import load_patients
from simcbctgenerator.utils.config import CBCTSystemConfig, MotionConfig, sample_motion_config
from simcbctgenerator.api.reconstruction import ProjectionPipeline
from simcbctgenerator.utils.physics_config import MANUFACTURER_DEFAULTS
from pathlib import Path
from argparse import ArgumentParser
from pydantic import BaseModel
from typing import Dict, Any, Optional
import numpy as np
from simcbctgenerator import utils
import SimpleITK as sitk
import json
import yaml
import os

# Set matplotlib to non-interactive backend before any imports that might use it
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logger = utils.setup_logger()


class ChallengePatientConfig(BaseModel):
    """Configuration for challenge data format."""
    ct_filename: str = "ct_def.mha"
    projections_filename: str = "projections.mha"
    cbct_filename: str = "cbct_clinical.mha"
    geometry_filename: str = "geometry.xml"
    metadata_filename: str = "metadata.yaml"


def save_projections_config(
    output_dir: Path,
    system_config: CBCTSystemConfig,
    physics_config=None
):
    """Save the projection and system config in a JSON file."""
    config: Dict[str, Any] = {
        "source_origin_distance": system_config.effective_source_origin_distance,
        "source_detector_distance": system_config.effective_source_detector_distance,
        "detector_size_h": system_config.geometry.detector_size_h,
        "detector_size_w": system_config.geometry.detector_size_w,
        "detector_pixels_h": system_config.geometry.detector_pixels_h,
        "detector_pixels_w": system_config.geometry.detector_pixels_w,
        "recon_size": system_config.reconstruction_volume.recon_size,
        "recon_origin": system_config.reconstruction_volume.recon_origin,
        "recon_spacing": system_config.reconstruction_volume.recon_spacing,
        "angles": system_config.effective_angles.tolist(),
    }

    # Add per-angle offsets from geometry data
    geo_data = system_config.geometry_data
    config["offset_x"] = geo_data.offset_x.tolist()
    config["offset_y"] = geo_data.offset_y.tolist()

    # Add physics parameters if provided
    if physics_config is not None:
        config["physics"] = {
            "photon_flux": physics_config.photon_flux,
            "spr": physics_config.spr,
            "mAs": physics_config.mAs,
            "kv": physics_config.kv,
            "saturation_factor": physics_config.saturation_factor,
            "bp_amplitude": physics_config.bp_amplitude,
            "bp_std": physics_config.bp_std,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'projection_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f'Saved projection config to {output_dir / "projection_config.json"}')


def main(
    patient_list_dir: Path,
    output_path: Path,
    config: Optional[ChallengePatientConfig] = None,
    threads: int = 8,
    max_block_index: int = 200,
    gpu: bool = True,
    enable_motion: bool = False,
    motion_type: str = "PELVIS",
    contour_name: str = "bowel",
    use_totalsegmentator: bool = False,
    motion_seed: Optional[int] = None,
    correct_contrast_media: bool = False,
    polychromatic: bool = False
):
    if config is None:
        config = ChallengePatientConfig()

    patients = load_patients(patient_list_dir)
    logger.info(f'{len(patients)} unique patient paths found.')

    output_dir = output_path
    for patient_dir in patients:
        patient_id = patient_dir.name
        logger.info(f'Processing patient: {patient_id}')

        ct_path       = patient_dir / config.ct_filename
        cbct_path     = patient_dir / config.cbct_filename
        geometry_path = patient_dir / config.geometry_filename
        metadata_path = patient_dir / config.metadata_filename

        for path, name in [(ct_path, "CT"), (geometry_path, "Geometry"), (metadata_path, "Metadata")]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        has_cbct = cbct_path.exists()
        if not has_cbct:
            logger.info(f'No real CBCT found at {cbct_path}, skipping comparison.')

        with open(metadata_path) as _f:
            _meta = yaml.safe_load(_f)
        _manufacturer = _meta.get('cbct', {}).get('Manufacturer', 'Elekta').lower()
        vendor = 'varian' if 'varian' in _manufacturer else 'elekta'
        logger.info(f'Detected vendor: {vendor}')

        pipeline = ProjectionPipeline(
            vendor=vendor,
            correct_contrast_media=correct_contrast_media,
            polychromatic=polychromatic,
            gpu=gpu,
            threads=threads,
            max_block_index=max_block_index,
        )

        output_dir = output_path / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        ct_image   = sitk.ReadImage(str(ct_path))
        cbct_image = sitk.ReadImage(str(cbct_path)) if has_cbct else None

        if enable_motion:
            # Motion path: manual setup; 4D CT not yet supported in the API.
            tpp = MANUFACTURER_DEFAULTS[pipeline.vendor.value]['time_per_projection']
            rng = np.random.default_rng(motion_seed)
            motion_config = sample_motion_config(
                motion_type=MotionConfig.MotionType[motion_type.upper()],
                amplitude_range=(2.5, 7.5),
                frequency_range=(12, 20),
                time_per_projection=tpp,
                uncertainty_range=(0.02, 0.02),
                phase_offset_breathing=rng.uniform(0, 2 * np.pi),
                rng=rng
            )
            logger.info(
                f'Motion config: amplitude={motion_config.amplitude_breathing:.1f}mm, '
                f'frequency={motion_config.frequency_breathing}bpm'
            )
            simulated_cbct = pipeline.run(
                ct_image=ct_image,
                cbct_image=cbct_image,
                geometry_xml=geometry_path,
                metadata_yaml=metadata_path,
                output_dir=output_dir,
                motion_config=motion_config,
                cleanup_temp=True,
            )

        else:
            # Static CT path via API — isocenter derived from cbct_image, no resampling.
            simulated_cbct = pipeline.run(
                ct_image=ct_image,
                cbct_image=cbct_image,
                geometry_xml=geometry_path,
                metadata_yaml=metadata_path,
                output_dir=output_dir,
                cleanup_temp=True,
            )

        system_config = pipeline._create_simulator().build_system_config(geometry_path, metadata_path)

        if cbct_image is not None:
            real_cbct = cbct_image
            if real_cbct.GetSize() != simulated_cbct.GetSize():
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(simulated_cbct)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(-1000)
                real_cbct = resampler.Execute(real_cbct)
            save_cbct_comparison(simulated_cbct, real_cbct, output_dir / f"comparison_{patient_id}.png", patient_id)

        save_projections_config(output_dir, system_config, system_config.physics)
        logger.info(f'Successfully processed patient {patient_id}')

    return output_dir


def pipeline():
    """Entry point for the A001 challenge projection/reconstruction pipeline."""
    parser = ArgumentParser('A001 Challenge projection generation and reconstruction pipeline')

    # Required arguments
    parser.add_argument('--patient_dir', type=str, default='list/test_COBRA.txt', #required=True,
                       help='Path to A001-style patient directory')
    parser.add_argument('--output_path', type=str, default='output_test_recon_pipeline/', #required=True,
                       help='Output folder for generated data')

    # Patient folder structure options
    parser.add_argument('--ct_filename', type=str, default='ct_def.mha',
                       help='CT filename in patient folder')
    parser.add_argument('--projections_filename', type=str, default='projections.mha',
                       help='Original projections filename (for reference)')
    parser.add_argument('--cbct_filename', type=str, default='cbct_clinical.mha',
                       help='Real CBCT filename for comparison')
    parser.add_argument('--geometry_filename', type=str, default='geometry.xml',
                       help='RTK geometry XML filename')
    parser.add_argument('--metadata_filename', type=str, default='metadata.yaml',
                       help='Metadata YAML filename')

    # Processing options
    parser.add_argument('--threads', type=int, default=8,
                       help='CUDA threads (default: 8)')
    parser.add_argument('--max_block_index', type=int, default=200,
                       help='CUDA block limit (default: 200)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU for reconstruction (default: True)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Use CPU for reconstruction')

    # Motion simulation options
    parser.add_argument('--enable_motion', action='store_true', #default=False,
                       help='Enable respiratory motion simulation')
    parser.add_argument('--motion_type', type=str, default='PELVIS',
                       choices=['PELVIS', 'THORAX'],
                       help='Type of motion model (default: PELVIS)')
    parser.add_argument('--contour_name', type=str, default='bowel',
                       help='Contour name for motion model (default: bowel)')
    parser.add_argument('--use_totalsegmentator', action='store_true', #default=False,
                       help='Use TotalSegmentator for automatic bowel mask generation')
    parser.add_argument('--motion_seed', type=int, default=None,
                       help='Random seed for motion sampling (default: None for random)')
    parser.add_argument('--correct_contrast_media', action='store_true', #default=False,
                       help='Apply contrast media correction to CT (only with --enable_motion)')
    parser.add_argument('--polychromatic', action='store_true', #default=False,
                       help='Use polychromatic projection simulation (default: False)')

    args = parser.parse_args()

    # Create patient config
    patient_config = ChallengePatientConfig(
        ct_filename=args.ct_filename,
        projections_filename=args.projections_filename,
        cbct_filename=args.cbct_filename,
        geometry_filename=args.geometry_filename,
        metadata_filename=args.metadata_filename,
    )

    main(
        patient_list_dir=Path(args.patient_dir),
        output_path=Path(args.output_path),
        config=patient_config,
        threads=args.threads,
        max_block_index=args.max_block_index,
        gpu=args.gpu,
        enable_motion=args.enable_motion,
        motion_type=args.motion_type,
        contour_name=args.contour_name,
        use_totalsegmentator=args.use_totalsegmentator,
        motion_seed=args.motion_seed,
        correct_contrast_media=args.correct_contrast_media,
        polychromatic=args.polychromatic
    )


if __name__ == '__main__':
    pipeline()
