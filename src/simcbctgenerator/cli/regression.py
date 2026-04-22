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

"""Regression pipeline for generating aligned CT-CBCT pairs for regression tasks."""

from simcbctgenerator.patient import Patient, PatientConfig
from simcbctgenerator.patient_setup import get_patient_loader
from simcbctgenerator.registration import RegistrationConfig, assert_docker_prerequisites
from simcbctgenerator.registration.visualization import save_multiplanar_comparison
from simcbctgenerator.api.regression import RegressionPipeline
from simcbctgenerator.preprocessing.elekta_cbct_exporter import resample
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from simcbctgenerator import utils
import SimpleITK as sitk

logger = utils.setup_logger()

def load_patients(path: Path) -> np.ndarray:
    """Load patient paths from text file.

    Args:
        path: Path to file containing patient paths

    Returns:
        Array of unique patient paths
    """
    if Path(path).exists():
        with open(path, 'r') as f:
            paths = [Path(line.strip()) for line in f.readlines()]
        return np.unique(paths)
    raise FileNotFoundError(f"Patient list file not found: {path}")


def save_nnunet_format(patient_id: str, cbct_image: sitk.Image, ct_image: sitk.Image, mask_image: sitk.Image,
                       images_path: Path, labels_path: Path):
    """Save images in nnUNet format for regression tasks.

    Args:
        patient_id: Patient identifier
        cbct_image: CBCT image (channel 0)
        ct_image: Registered CT image (channel 1)
        images_path: Path to imagesTr directory
        labels_path: Path to labelsTr directory
    """
    # Create directories
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    # Save mask in labelsTr
    mask_filename = labels_path / f"{patient_id}.nii.gz"
    sitk.WriteImage(mask_image, str(mask_filename))
    logger.info(f"Saved mask: {mask_filename}")

    # Save CBCT as channel 0000
    cbct_filename = images_path / f"{patient_id}_0000.nii.gz"
    cbct_cast = sitk.Cast(cbct_image, sitk.sitkInt16)
    masked_cbct = sitk.Mask(cbct_cast, mask_image, -1024)
    sitk.WriteImage(masked_cbct, str(cbct_filename))
    logger.info(f"Saved CBCT: {cbct_filename}")

    # Save CT as channel 0001
    ct_filename = images_path / f"{patient_id}_0001.nii.gz"
    ct_cast = sitk.Cast(ct_image, sitk.sitkInt16)
    masked_ct = sitk.Mask(ct_cast, mask_image, -1024)
    sitk.WriteImage(masked_ct, str(ct_filename))
    logger.info(f"Saved CT: {ct_filename}")




def process_patient_registration(patient_dir: Path,
                                 patient_config: PatientConfig,
                                 registration_config: RegistrationConfig,
                                 output_path: Path,
                                 overwrite: bool = False,
                                 use_rigid: bool = True,
                                 use_deformable: bool = True) -> bool:
    """Process a single patient for registration.

    Args:
        patient_dir: Path to patient directory
        patient_config: Patient configuration
        registration_config: Registration configuration
        output_path: Output directory
        overwrite: Whether to overwrite existing results

    Returns:
        True if the patient produced an aligned CBCT/CT pair, False otherwise.
    """
    try:
        patient = Patient(patient_config, patient_dir, allow_multi_plan=True)
    except KeyError as e:
        logger.error(e)
        return False
    if not patient.valid:
        logger.warning(f"Patient {patient.id} is not valid, skipping...")
        return False

    logger.info(f"Processing patient {patient.id}")

    # Create output paths
    images_path = output_path / "imagesTr"
    labels_path = output_path / "labelsTr"
    temp_output = output_path / "temp_registration" / patient.id

    # Check if already processed
    cbct_output = images_path / f"{patient.id}_0000.nii.gz"
    ct_output = images_path / f"{patient.id}_0001.nii.gz"
    if cbct_output.exists() and ct_output.exists() and not overwrite:
        logger.info(f"Patient {patient.id} already processed, skipping...")
        return True

    try:
        patient.correct_CM()
        ct_image = patient.ct_image
        ct_image.SetOrigin(patient.original_origin)

        values = get_patient_loader(patient.config).load_cbct(
            patient, apply_correction=False, return_transform=True
        )
        cbct_image, transform = values.get('img'), values.get('transform')
        if cbct_image is None:
            logger.warning(f"Could not load CBCT for patient {patient.id}, skipping...")
            return False

        # XVI stores the rigid alignment transform separately; pre-resample the
        # CT into CBCT space before deformable registration runs.
        if transform is not None:
            logger.info("Applying inverse transform to resample CT to CBCT space")
            ct_image = resample(ct_image, cbct_image, transform.GetInverse())

        pipeline = RegressionPipeline(
            use_rigid=use_rigid,
            use_deformable=use_deformable,
            registration_config=registration_config,
        )
        results = pipeline.run(
            ct_image=ct_image,
            cbct_image=cbct_image,
            output_dir=temp_output,
            crop_to_fov=True,
            simulate_cbct=False,
        )

        save_nnunet_format(
            patient_id=patient.id,
            cbct_image=results['cbct_image'],
            ct_image=results['registered_ct'],
            mask_image=results['fov_mask'],
            images_path=images_path,
            labels_path=labels_path,
        )

        if registration_config.save_visualizations:
            viz_output_dir = registration_config.visualization_output_dir or (output_path / "visualizations")
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            save_multiplanar_comparison(
                ct_orig=None,
                cbct=results['cbct_image'],
                ct_def=results['registered_ct'],
                output_path=viz_output_dir / f"registration_{patient.id}.png",
                patient_id=patient.id,
            )

        return True

    except Exception as e:
        logger.error(f"Error processing patient {patient.id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main(patient_path: Path,
         output_path: Path,
         patient_config: PatientConfig,
         registration_config: RegistrationConfig,
         overwrite: bool = False,
         use_rigid: bool = True,
         use_deformable: bool = True):
    """Main regression pipeline function.

    Args:
        patient_path: Path to patient list file
        output_path: Output directory
        patient_config: Patient configuration
        registration_config: Registration configuration
        overwrite: Whether to overwrite existing results
        use_deformable: Whether deformable (Docker-based) registration will run.
            Used here only to gate the Docker prerequisite assertion.

    Raises:
        RegistrationError: If deformable registration is requested but Docker /
            the required registration image are unavailable.
        RuntimeError: If every patient in the list failed to produce output.
    """
    # Fail fast on missing environment so users get a clear error BEFORE any
    # per-patient work runs (and before the per-patient except-block swallows it).
    if use_deformable:
        assert_docker_prerequisites()

    patients = load_patients(patient_path)
    logger.info(f'{len(patients)} unique patient paths found.')

    if overwrite:
        logger.info("OVERWRITE mode enabled - will reprocess existing results")

    succeeded = 0
    failed = 0
    for patient_dir in patients:
        ok = process_patient_registration(
            patient_dir=patient_dir,
            patient_config=patient_config,
            registration_config=registration_config,
            output_path=output_path,
            overwrite=overwrite,
            use_rigid=use_rigid,
            use_deformable=use_deformable,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1

    logger.info(f"Regression pipeline completed: {succeeded} succeeded, {failed} failed")

    if succeeded == 0 and len(patients) > 0:
        raise RuntimeError(
            f"All {len(patients)} patient(s) failed regression. See per-patient errors above."
        )


def pipeline():
    """Command-line interface for regression pipeline."""
    parser = ArgumentParser('CT-CBCT registration pipeline for regression tasks')

    # Add config argument with preset support
    utils.add_config_argument(parser, default='regression-synthrad')

    parser.add_argument('--patient_path', type=str, required=True,
                       help='File containing list of patient paths')
    parser.add_argument('--output_path', type=str, default='Dataset0xx_CBCT_regression',
                       help='Output directory for registered images')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--no-rigid', dest='use_rigid', action='store_false',
                       help='Skip rigid (Elastix) registration step')
    parser.add_argument('--no-deformable', dest='use_deformable', action='store_false',
                       help='Skip deformable (Docker Impact) registration step. '
                            'Use when the vboussot/elastix_impact container is unavailable.')
    parser.set_defaults(use_rigid=True, use_deformable=True)

    # Add patient configuration arguments
    utils.add_patient_arguments(parser)

    # Add registration-specific arguments
    utils.add_registration_arguments(parser)

    args = parser.parse_args()
    config, config_path = utils.load_config_from_args(args)

    args_dict = utils.merge_config_and_args(config, args)

    utils.log_config(logger, args_dict)

    # Helper function to resolve paths relative to config directory
    def resolve_path_relative_to_config(path_str: str) -> Path:
        """Resolve path relative to config directory if it's just a filename."""
        if not path_str:
            return None
        path = Path(path_str)
        # If path is absolute or already exists, use it as-is
        if path.is_absolute() or path.exists():
            return path
        # Otherwise, try to resolve relative to config directory
        config_dir_path = config_path.parent / path_str
        if config_dir_path.exists():
            return config_dir_path
        # Fall back to original path (might be relative to cwd)
        return path

    # Initialize patient config
    patient_config = PatientConfig(
        plan_dir=args_dict['plan_dir'],
        ct_dir=args_dict['ct_dir'],
        cbct_dir=args_dict['cbct_dir'],
        cm_mask=args_dict.get('cm_mask'),
        export_structures=args_dict['export_structures'],
        priority=args_dict['priority'],
        image_modality=args_dict['image_modality'],
    )

    # Resolve registration file paths relative to config directory
    parameter_map_path = resolve_path_relative_to_config(args_dict.get('parameter_map_path'))
    model_path = resolve_path_relative_to_config(args_dict.get('model_path'))

    # Initialize registration config
    registration_config = RegistrationConfig(
        elastix_binary_path=Path(args_dict.get('elastix_binary_path', '/usr/lib/elastix-install/bin/elastix')),
        parameter_map_path=parameter_map_path,
        model_path=model_path,
        models_dest_dir=args_dict.get('models_dest_dir') or 'Models/TS',
        persistent_data_dir=Path(args_dict.get('persistent_data_dir', '/Data')),
        threads=args_dict.get('threads', 24),
        use_fixed_mask=args_dict.get('use_fixed_mask', False),
        use_moving_mask=args_dict.get('use_moving_mask', False),
        save_visualizations=args_dict.get('save_visualizations', True),
        visualization_output_dir=Path(args_dict['visualization_output_dir']) if args_dict.get('visualization_output_dir') else None,
        clip_min=args_dict.get('clip_min', -1024.0),
        clip_max=args_dict.get('clip_max', 276.0),
        standardize_mean=args_dict.get('standardize_mean', -370.00039267657144),
        standardize_std=args_dict.get('standardize_std', 436.5998675471528),
    )

    main(
        patient_path=Path(args_dict['patient_path']),
        output_path=Path(args_dict['output_path']),
        patient_config=patient_config,
        registration_config=registration_config,
        overwrite=args_dict.get('overwrite', False),
        use_rigid=args_dict.get('use_rigid', True),
        use_deformable=args_dict.get('use_deformable', True),
    )


if __name__ == '__main__':
    pipeline()
