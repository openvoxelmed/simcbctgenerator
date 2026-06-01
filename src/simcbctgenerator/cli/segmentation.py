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

"""CLI pipeline for synthetic CBCT generation for segmentation tasks."""

from __future__ import annotations

import SimpleITK as sitk
from simcbctgenerator.api.segmentation import SegmentationPipeline
from simcbctgenerator.patient import Patient
from simcbctgenerator.patient_setup import get_patient_loader
from simcbctgenerator.utils.config import PatientConfig
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from simcbctgenerator import utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simcbctgenerator.utils.config import MotionConfig, CBCTSystemConfig, PhantomConfig

logger = utils.setup_logger()


def load_patients(path: Path) -> np.ndarray:
    """Load patient paths from a text file."""
    if Path(path).exists():
        with open(path, 'r') as f:
            paths = [Path(line.strip()) for line in f.readlines()]
        return np.unique(paths)  # prevents array to be non iterable in case of one patient
    raise FileNotFoundError(f"Patient list file not found: {path}")


def main(patient_path: Path,
         output_path: Path,
         patient_config: PatientConfig,
         method: str,
         motion_config: MotionConfig | None,
         system_config: CBCTSystemConfig | None,
         phantom_config: PhantomConfig | None,
         store_real_cbct: bool,
         store_ct: bool,
         no_motion: bool,
         correct_contrast_media: bool = False,
         ):
    ct_path = output_path / "imagesTr"
    exported_patients = {}
    if ct_path.exists():
        exported_patients = {file.stem.split("_")[0] for file in ct_path.iterdir() if file.suffix == '.gz'}

    segmentation_path = output_path / "labelsTr"
    real_cbct_path = output_path / "realImagesTr"
    real_ct_path = output_path / "labelsTrRegression"
    patients = load_patients(patient_path)
    logger.info(f'{len(patients)} unique patient paths found.')
    logger.info(f'Initialized synthetic CBCT pipeline with {method} method')

    pipeline = SegmentationPipeline(
        method=method,
        organ_list=list(patient_config.export_structures),
        priority=list(patient_config.priority),
        use_totalsegmentator=patient_config.use_totalsegmentator,
        phantom_config=phantom_config,
        correct_contrast_media=correct_contrast_media
    )

    for patient_dir in patients:
        ID = patient_config.image_modality.value.get_id(patient_dir)
        if ID in exported_patients:
            if store_ct and not (real_ct_path / f'{ID}.nii.gz').exists():
                logger.info(f'{ID} CBCT already exported but resampled CT missing, re-exporting CT')
            elif store_real_cbct and not (real_cbct_path / f'{ID}.nii.gz').exists():
                logger.info(f'{ID} CBCT already exported but real CBCT missing, re-exporting')
            else:
                logger.info(f'{ID} already exported into {output_path}')
                continue
        patient_dir = Path(patient_dir)
        patient = Patient(patient_config, patient_dir)
        if not patient.valid:
            continue
        try:
            cm_mask_image = None
            if patient_config.cm_mask:
                cm_mask_image = patient.mask_dictionary.get(patient_config.cm_mask)
                if cm_mask_image is not None:
                    logger.info(f"Using '{patient_config.cm_mask}' from RTStruct as contrast media mask")
            results = pipeline.run(
                ct_image=patient.ct_image,
                system_config=system_config,
                output_dir=output_path / patient.id,
                mask_image=patient.combined_label_mask(),
                cm_mask_image=cm_mask_image,
                motion_config=None if no_motion else motion_config,
                motion_surrogate=getattr(patient, 'motion_surrogate', None),
                phantom_config=phantom_config,
                cleanup_temp=True,
            )

            recon = results['simulated_cbct']
            ct_path.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(recon, str(ct_path / f"{patient.id}_0000.nii.gz"))

            segmentation_path.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(results['label_mask'], str(segmentation_path / f"{patient.id}.nii.gz"))

            if store_real_cbct:
                get_patient_loader(patient.config).save_real_cbct(patient, real_cbct_path)
            if store_ct:
                real_ct_path.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(results['resampled_ct'], str(real_ct_path / f"{patient.id}.nii.gz"))
        except MemoryError:
            logger.error(f'Out of Memory error for patient {ID}', exc_info=True)

def pipeline():
    parser = ArgumentParser('synthetic CBCT generation pipeline')

    # Add config argument with preset support
    utils.add_config_argument(parser, default='segmentation-synthrad')

    parser.add_argument('--patient_path', type=str, default='./list/synthrad2025-TH.txt', help='a file where all the patient paths are stored.')
    parser.add_argument('--drr_path', type=str, default='projection', help='destination to store the projection files.')
    parser.add_argument('--output_path', type=str, default='Dataset0xx_simCBCT', help='output folder where to store the generated synthetic cts.')
    parser.add_argument('--store_real_cbct', action='store_true', help='store the real cbct images.')
    parser.add_argument('--store_ct', action='store_true', help='store the resampled ct images.')
    parser.add_argument('--correct_contrast_media', action='store_true', #default=False,
                       help='Apply contrast media correction to CT (only effective when motion is enabled)')

    utils.add_patient_arguments(parser)
    utils.add_router_arguments(parser)
    utils.add_4d_ct_arguments(parser)
    utils.add_physics_arguments(parser)
    utils.add_geometry_arguments(parser)
    utils.add_reconstruction_volume_arguments(parser)
    utils.add_phantom_arguments(parser)
    args = parser.parse_args()
    config, _ = utils.load_config_from_args(args)

    args_dict = utils.merge_config_and_args(config, args)
    #args_dict = {k: v for k, v in args_dict.items() if v is not None}
    utils.log_config(logger, args_dict)



    # Initialize configs from args_dict
    patient_config = PatientConfig(
    plan_dir=args_dict['plan_dir'],
    ct_dir=args_dict['ct_dir'],
    cbct_dir=args_dict['cbct_dir'],
    cm_mask=args_dict['cm_mask'],
    export_structures=args_dict['export_structures'],
    priority=args_dict['priority'],
    image_modality=args_dict['image_modality'],
    use_totalsegmentator=args_dict['use_totalsegmentator']
    )

    method = args_dict['method']

    if method == 'standard':
        from simcbctgenerator.utils import PhysicsConfig, GeometryConfig, MotionConfig, ReconstructionVolumeConfig, CBCTSystemConfig

        motion_config = MotionConfig(
            motion_type=args_dict['motion_type'],
            amplitude_breathing=args_dict['amplitude_breathing'],
            amplitude_heart=args_dict['amplitude_heart'],
            phase_offset_breathing=args_dict['phase_offset_breathing'],
            phase_offset_heart=args_dict['phase_offset_heart'],
            contour_name=args_dict['contour_name'],
            frequency_breathing=args_dict['frequency_breathing'],
            frequency_heartbeat=args_dict['frequency_heartbeat'],
            time_per_projection=args_dict['time_per_projection'],
            uncertainty=args_dict['uncertainty']
        )

        # Create sub-configs
        physics_config = PhysicsConfig(
            photon_flux=args_dict['photon_flux'],
            mAs=args_dict['mas'],
            spr=args_dict['spr'],
            kv=args_dict['kv'],
            saturation_factor=args_dict.get('saturation_factor', 1.0),
            bp_amplitude=args_dict['bp_amplitude'],
            bp_std=args_dict['bp_std'],
            polychromatic=args_dict['polychromatic'],
            T1=args_dict['T1'],
            T2=args_dict['T2'],
            threads=args_dict.get('threads', 8),
            max_block_index=args_dict.get('max_block_index', 200),
            no_scatter=args_dict.get('no_scatter', False),
            no_noise=args_dict.get('no_noise', False),
        )

        geometry_config = GeometryConfig(
            source_origin_distance=args_dict['source_origin_distance'],
            source_detector_distance=args_dict['source_detector_distance'],
            detector_offset=args_dict['detector_offset'],
            detector_size_h=args_dict['detector_size_h'],
            detector_size_w=args_dict['detector_size_w'],
            detector_pixels_h=args_dict['detector_pixels_h'],
            detector_pixels_w=args_dict['detector_pixels_w'],
            start_angle=args_dict.get('start_angle', 0.0),
            end_angle=args_dict.get('end_angle', 360.0),
            angle_increments=args_dict.get('angle_increments', 1.0),
            geometry_xml_path=args_dict.get('geometry_xml_path')
        )

        reconstruction_volume_config = ReconstructionVolumeConfig(
            recon_size=args_dict['recon_size'],
            recon_origin=args_dict['recon_origin'],
            recon_spacing=args_dict['recon_spacing']
        )

        # Create composite system config
        system_config = CBCTSystemConfig(
            physics=physics_config,
            geometry=geometry_config,
            reconstruction_volume=reconstruction_volume_config
        )
        phantom_config = None

    elif method == 'phantom':
        from simcbctgenerator.utils.config import PhantomConfig

        phantom_config = PhantomConfig(
            phantom_path=args_dict['phantom_path'],
            intensity_factor=args_dict['intensity_factor'],
            water_threshold=args_dict['water_threshold'],
            enhancement_factor=args_dict['enhancement_factor'],
            lower_threshold=args_dict['lower_threshold'],
            body_threshold=args_dict['body_threshold'],
            gaussian_sigma=args_dict['gaussian_sigma'],
            noise_range=(args_dict['noise_range_min'], args_dict['noise_range_max'])
        )

        system_config = None
        motion_config = None

    else:
        raise NotImplementedError(f"{args_dict['method']} is not implemented.")

    main(patient_path=Path(args_dict['patient_path']),
         output_path=Path(args_dict['output_path']),
         patient_config=patient_config,
         method=method,
         motion_config=motion_config,
         system_config=system_config,
         phantom_config=phantom_config,
         store_real_cbct=args_dict['store_real_cbct'],
         store_ct=args_dict['store_ct'],
         no_motion=args_dict['no_motion'],
         correct_contrast_media=args.correct_contrast_media,
         )

if __name__ == '__main__':
    pipeline()
