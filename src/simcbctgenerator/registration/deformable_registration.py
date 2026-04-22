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

"""Deformable registration engine for CT-CBCT alignment.

Uses SimpleITK-SimpleElastix for rigid registration and Docker container for deformable registration.
"""

from __future__ import annotations

import SimpleITK as sitk
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from simcbctgenerator.registration.registration_config import RegistrationConfig

logger = logging.getLogger(__name__)

IMPACT_DOCKER_IMAGE = "vboussot/elastix_impact"


class RegistrationError(Exception):
    """Custom exception for registration errors."""
    pass


def assert_docker_prerequisites(docker_image: str = IMPACT_DOCKER_IMAGE) -> None:
    """Assert that Docker and the required registration image are available.

    Raises RegistrationError with an actionable message if either is missing.
    Call this once up front (before looping over patients) so users who lack
    the container get a clear error instead of per-patient failures that the
    CLI otherwise swallows.
    """
    if shutil.which("docker") is None:
        raise RegistrationError(
            "Docker is required for deformable registration but was not found on PATH. "
            "Install Docker (Desktop or Engine) and ensure the daemon is running, or "
            "construct RegressionPipeline with use_deformable=False to skip this step."
        )

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise RegistrationError(
            f"Failed to invoke 'docker image inspect': {exc}. "
            "Verify the Docker daemon is running."
        ) from exc

    if result.returncode != 0:
        raise RegistrationError(
            f"Required Docker image '{docker_image}' is not available locally.\n"
            f"Pull it first with:\n"
            f"    docker pull {docker_image}\n"
            f"or construct RegressionPipeline with use_deformable=False to skip "
            f"the deformable registration step."
        )


class RegistrationEngine:
    """Engine for performing rigid and deformable registration between CT and CBCT images."""

    def __init__(self, config: RegistrationConfig):
        """Initialize registration engine.

        Args:
            config: Registration configuration
        """
        self.config = config
        self.persistent_data_dir = None

    def _check_elastix_available(self):
        """Check if SimpleElastix is available.

        Raises:
            RegistrationError: If SimpleElastix is not available
        """
        try:
            # Test if elastix functions are available
            sitk.ElastixImageFilter()
        except AttributeError:
            raise RegistrationError(
                "SimpleElastix is required for image registration. "
                "Install it with: uv sync --extra registration"
            )

    def clip_and_standardize_CT(self, image: sitk.Image) -> sitk.Image:
        """Clip and standardize CT image values.

        Args:
            image: Input CT image

        Returns:
            Standardized CT image
        """
        data = sitk.GetArrayFromImage(image)
        data[data < self.config.clip_min] = self.config.clip_min
        data[data > self.config.clip_max] = self.config.clip_max
        data = (data - self.config.standardize_mean) / self.config.standardize_std
        result = sitk.GetImageFromArray(data)
        result.CopyInformation(image)
        return result

    def standardize_z_score(self, image: sitk.Image) -> sitk.Image:
        data = sitk.GetArrayFromImage(image)
        data = (data - data.mean()) / data.std()
        result = sitk.GetImageFromArray(data)
        result.CopyInformation(image)
        return result


    def mask_image(self, img: sitk.Image, mask: sitk.Image,
                   default_value: float = -1000) -> sitk.Image:
        """Mask image with binary mask.

        Args:
            img: Input image
            mask: Binary mask
            default_value: Value for pixels outside mask

        Returns:
            Masked image
        """
        return sitk.Mask(img, mask, outsideValue=default_value)

    def setup_persistent_data_dir(self) -> Path:
        """Setup persistent data directory with models and parameter maps.

        Returns:
            Path to persistent data directory
        """
        if self.persistent_data_dir is None:
            self.persistent_data_dir = Path(self.config.persistent_data_dir)
            self.persistent_data_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file if specified
            if self.config.model_path is not None and self.config.model_path.exists():
                models_dest_dir = self.persistent_data_dir / self.config.models_dest_dir
                models_dest_file = models_dest_dir / self.config.model_path.name

                logger.info(f"Copying model to persistent directory: {models_dest_file}")
                models_dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(self.config.model_path), str(models_dest_file))
            else:
                logger.info("No model path specified")

            # Copy parameter map if specified
            if self.config.parameter_map_path is not None and self.config.parameter_map_path.exists():
                param_map_dest = self.persistent_data_dir / "ParameterMap.txt"
                shutil.copy(str(self.config.parameter_map_path), str(param_map_dest))
                logger.info(f"Copied parameter map to {param_map_dest}")
            else:
                logger.warning("Parameter map not specified, will create basic one")

        return self.persistent_data_dir

    def cleanup_persistent_data_dir(self):
        """Clean up persistent data directory."""
        if self.persistent_data_dir and self.persistent_data_dir.exists():
            shutil.rmtree(self.persistent_data_dir, ignore_errors=True)
            self.persistent_data_dir = None

    def apply_deformation(self, img: sitk.Image, transform_file: Path) -> sitk.Image:
        """Apply deformation field to image using transformix.

        Args:
            img: Image to deform
            transform_file: Path to transform parameter file

        Returns:
            Deformed image
        """
        self._check_elastix_available()

        transform = sitk.TransformixImageFilter()
        param_map = transform.ReadParameterFile(str(transform_file.absolute()))
        transform.SetTransformParameterMap(param_map)
        transform.SetMovingImage(img)
        transform.SetLogToConsole(False)
        transform.SetLogToFile(False)
        transform.Execute()
        deformed_img = transform.GetResultImage()
        return deformed_img

    def get_rigid_parameter_map(self) -> sitk.ParameterMap:
        """Get rigid parameter map for elastix registration.

        Returns:
            SimpleITK parameter map for rigid registration
        """
        self._check_elastix_available()

        parameter_map = sitk.GetDefaultParameterMap("rigid")

        # Customize parameters for CT-CBCT registration based on working configuration
        parameter_map["AutomaticParameterEstimation"] = ["true"]
        parameter_map["AutomaticScalesEstimation"] = ["false"]
        parameter_map["AutomaticTransformInitialization"] = ["true"]
        parameter_map["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]
        parameter_map["CheckNumberOfSamples"] = ["true"]
        parameter_map["DefaultPixelValue"] = ["-1024.000000"]
        parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
        parameter_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
        parameter_map["ImageSampler"] = ["RandomCoordinate"]
        parameter_map["Interpolator"] = ["LinearInterpolator"]
        parameter_map["MaximumNumberOfIterations"] = ["256"]
        parameter_map["MaximumNumberOfSamplingAttempts"] = ["8"]
        parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]
        parameter_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
        parameter_map["NewSamplesEveryIteration"] = ["true"]
        parameter_map["NumberOfResolutions"] = ["4"]
        parameter_map["NumberOfSamplesForExactGradient"] = ["4096"]
        parameter_map["NumberOfSpatialSamples"] = ["2048"]
        parameter_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        parameter_map["Registration"] = ["MultiResolutionRegistration"]
        parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        parameter_map["Resampler"] = ["DefaultResampler"]
        parameter_map["ResultImageFormat"] = ["nii"]
        parameter_map["Transform"] = ["EulerTransform"]
        parameter_map["WriteIterationInfo"] = ["false"]
        parameter_map["WriteResultImage"] = ["true"]

        return parameter_map

    def rigid_registration_elastix(self, fixed_image: sitk.Image,
                                   moving_image: sitk.Image,
                                   fixed_mask: Optional[sitk.Image] = None,
                                   moving_mask: Optional[sitk.Image] = None) -> Tuple[sitk.Image, List[sitk.ParameterMap]]:
        """Perform rigid registration using SimpleITK Elastix.

        Args:
            fixed_image: Fixed (reference) image
            moving_image: Moving image to align
            fixed_mask: Optional mask for fixed image
            moving_mask: Optional mask for moving image

        Returns:
            Tuple of (registered image, transform parameter maps)
        """
        self._check_elastix_available()

        logger.info("Performing rigid registration with SimpleITK Elastix...")

        # Initialize Elastix
        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fixed_image)
        elastix.SetMovingImage(moving_image)

        # Set masks if provided
        if fixed_mask is not None:
            logger.info("Using fixed image mask for rigid registration")
            elastix.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            logger.info("Using moving image mask for rigid registration")
            elastix.SetMovingMask(moving_mask)

        # Set rigid parameter map
        parameter_map = self.get_rigid_parameter_map()
        elastix.SetParameterMap(parameter_map)

        # Disable logging to console
        elastix.LogToConsoleOn()

        # Execute registration
        logger.info("Executing rigid elastix registration...")
        elastix.Execute()

        # Get result image
        result_image = elastix.GetResultImage()

        # Get transform parameter maps
        transform_parameter_maps = elastix.GetTransformParameterMap()

        logger.info("Rigid registration completed successfully")

        return result_image, transform_parameter_maps

    def apply_rigid_transform(self, image: sitk.Image,
                             transform_parameter_maps: List[sitk.ParameterMap]) -> sitk.Image:
        """Apply rigid transformation to image using transformix.

        Args:
            image: Image to transform
            transform_parameter_maps: Transform parameter maps from elastix

        Returns:
            Transformed image
        """
        self._check_elastix_available()

        logger.info("Applying rigid transformation with transformix...")

        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(image)
        transformix.SetTransformParameterMap(transform_parameter_maps)
        transformix.LogToConsoleOff()
        transformix.Execute()

        return transformix.GetResultImage()

    def deformable_registration_docker(self, fixed_image_path: Path,
                                      moving_image_path: Path,
                                      output_dir: Path,
                                      fixed_mask_path: Optional[Path] = None,
                                      moving_mask_path: Optional[Path] = None) -> Path:
        """Run deformable registration using Docker container with elastix.

        Args:
            fixed_image_path: Path to fixed image file
            moving_image_path: Path to moving image file
            output_dir: Directory for output
            fixed_mask_path: Optional path to fixed image mask
            moving_mask_path: Optional path to moving image mask

        Returns:
            Path to transform parameter file
        """
        # Setup persistent data directory
        data_dir = self.setup_persistent_data_dir()
        temp_out_dir = Path('Out')
        temp_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load and standardize images
            fixed_image = sitk.ReadImage(str(fixed_image_path))
            moving_image = sitk.ReadImage(str(moving_image_path))

            # Standardize images
            fixed_std = self.standardize_z_score(fixed_image)
            # fixed_std = self.clip_and_standardize_CT(fixed_image)
            moving_std = self.clip_and_standardize_CT(moving_image)

            # Write standardized images to persistent data directory
            sitk.WriteImage(fixed_std, str(data_dir / "Fixed_image.mha"))
            sitk.WriteImage(moving_std, str(data_dir / "Moving_image.mha"))

            # Parameter map handling
            param_map_path = data_dir / "ParameterMap.txt"
            if not param_map_path.exists():
                logger.warning("Creating basic parameter map")
                self._create_basic_parameter_map(param_map_path)

            # Build elastix command (runs inside Docker container)
            # Note: Paths inside the container use the mounted directories
            elastix_cmd = [
                "docker",
                "run", "--rm", "--gpus", "all",
                "-v", f"{data_dir.absolute()}:/Data",
                "-v", f"{temp_out_dir.absolute()}:/Out",
                IMPACT_DOCKER_IMAGE,
                str(self.config.elastix_binary_path),
                "-f", "/Data/Fixed_image.mha",
                "-m", "/Data/Moving_image.mha",
                "-p", "/Data/ParameterMap.txt",
                "-out", "/Out",
                "-threads", str(self.config.threads)
            ]

            # Add masks if specified and exist
            if self.config.use_fixed_mask and fixed_mask_path is not None and fixed_mask_path.exists():
                elastix_cmd.extend(["-fMask", str(fixed_mask_path)])
            if self.config.use_moving_mask and moving_mask_path is not None and moving_mask_path.exists():
                elastix_cmd.extend(["-mMask", str(moving_mask_path)])

            logger.info(f"Running elastix in Docker: {' '.join(elastix_cmd)}")

            # Run elastix (assumes running inside Docker container)
            import subprocess
            subprocess.run(elastix_cmd, check=True)

            logger.info("Elastix deformable registration completed successfully")

            # Copy transformation file to output directory
            transform_file = temp_out_dir / "TransformParameters.0.txt"
            if transform_file.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                output_transform = output_dir / "TransformParameters.0.txt"
                shutil.copy(str(transform_file), str(output_transform))
                return output_transform
            else:
                raise FileNotFoundError("TransformParameters.0.txt not generated by elastix")

        except Exception as e:
            logger.error(f"Elastix deformable registration failed: {e}")
            raise
        finally:
            # Clean up temporary output directory
            if temp_out_dir.exists():
                shutil.rmtree(temp_out_dir, ignore_errors=True)

    def _create_basic_parameter_map(self, output_path: Path):
        """Create a basic elastix parameter map for deformable registration.

        Args:
            output_path: Path to save parameter map
        """
        with open(output_path, 'w') as f:
            f.write("// Basic B-Spline parameter map\n")
            f.write("(Registration \"MultiResolutionRegistration\")\n")
            f.write("(Transform \"BSplineTransform\")\n")
            f.write("(FinalBSplineInterpolationOrder 3)\n")
            f.write("(GridSpacingSchedule 4.0 2.0 1.0)\n")
            f.write("(HowToCombineTransforms \"Compose\")\n")
            f.write("(Metric \"AdvancedMattesMutualInformation\")\n")
            f.write("(NumberOfHistogramBins 32)\n")
            f.write("(NumberOfResolutions 3)\n")
            f.write("(MaximumNumberOfIterations 500)\n")
            f.write("(ImageSampler \"Random\")\n")
            f.write("(NumberOfSpatialSamples 2048)\n")
            f.write("(Optimizer \"AdaptiveStochasticGradientDescent\")\n")
            f.write("(ResampleInterpolator \"FinalBSplineInterpolator\")\n")
            f.write("(Resampler \"DefaultResampler\")\n")
            f.write("(ResultImageFormat \"mha\")\n")
            f.write("(WriteResultImage \"false\")\n")

    def register_ct_to_cbct(self, ct_image: sitk.Image, cbct_image: sitk.Image,
                           output_dir: Path,
                           use_rigid: bool = True,
                           use_deformable: bool = True,
                           fixed_mask: Optional[sitk.Image] = None,
                           moving_mask: Optional[sitk.Image] = None) -> Tuple[sitk.Image, Optional[Path]]:
        """Register CT to CBCT (CBCT as fixed, CT as moving).

        Uses SimpleITK Elastix for rigid registration and Docker container for deformable registration.

        Args:
            ct_image: CT image to register
            cbct_image: CBCT image (reference)
            output_dir: Directory for output files
            use_rigid: Whether to perform rigid registration first
            use_deformable: Whether to perform deformable registration
            fixed_mask: Optional mask for CBCT (fixed) image
            moving_mask: Optional mask for CT (moving) image

        Returns:
            Tuple of (registered CT image, path to deformation transform or None)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        transform_file = None

        # Save original images
        ct_path = output_dir / "ct_original.mha"
        cbct_path = output_dir / "cbct_original.mha"
        sitk.WriteImage(ct_image, str(ct_path))
        sitk.WriteImage(cbct_image, str(cbct_path))

        current_ct = ct_image

        # Rigid registration using SimpleITK Elastix
        if use_rigid:
            logger.info("Performing rigid registration with Elastix (CBCT fixed, CT moving)")
            rigid_ct, rigid_transform_maps = self.rigid_registration_elastix(
                fixed_image=cbct_image,
                moving_image=current_ct,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask
            )
            sitk.WriteImage(rigid_ct, str(output_dir / "ct_rigid.mha"))

            # Save rigid transform
            for i, param_map in enumerate(rigid_transform_maps):
                sitk.WriteParameterFile(param_map, str(output_dir / f"RigidTransformParameters.{i}.txt"))

            current_ct = rigid_ct

        # Deformable registration using Docker container
        if use_deformable:
            logger.info("Performing deformable registration with Docker (CBCT fixed, CT moving)")

            # Save current CT for Docker elastix
            moving_path = output_dir / "ct_for_deformable.mha"
            sitk.WriteImage(current_ct, str(moving_path))

            transform_file = self.deformable_registration_docker(
                fixed_image_path=cbct_path,
                moving_image_path=moving_path,
                output_dir=output_dir
            )

            # Apply deformation to get final registered CT
            deformed_ct = self.apply_deformation(current_ct, transform_file)
            sitk.WriteImage(deformed_ct, str(output_dir / "ct_deformed.mha"))
            current_ct = deformed_ct

        logger.info(f"Registration complete. Output saved to {output_dir}")

        return current_ct, transform_file
