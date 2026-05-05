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

"""Module for CBCT reconstruction using RTK FDK algorithm."""

import itk
from itk import RTK as rtk
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple
import numpy as np
import logging
from simcbctgenerator.utils import log_time
from simcbctgenerator.utils.config import CBCTSystemConfig

logger = logging.getLogger(__name__)

UINT16_MAX = 65535




class SyntheticCBCTReconstruction:
    """CBCT reconstruction using RTK FDK algorithm."""
    def __init__(self, system_config: CBCTSystemConfig, gpu: bool = False):
        self.system_config: CBCTSystemConfig = system_config
        self.gpu: bool = gpu

        self.CPUImageType = rtk.Image[itk.F,3]
        self.GPUImageType = rtk.CudaImage[itk.F,3]

        self.geometry = None
        self.fov_img = None

        self._init()
        self.build_graph(gpu=gpu)

        self.build_fov()


    def _init(self):
        if self.system_config.effective_angles is not None:
            self.construct_geometry(self.system_config.effective_angles)

    def build_fov(self):
        source = rtk.ConstantImageSource[self.CPUImageType].New()
        source.SetSpacing(self.system_config.recon_spacing)
        source.SetOrigin(self.system_config.recon_origin)
        source.SetSize(self.system_config.recon_size)
        source.SetConstant(1)
        source.Update()

        projection = rtk.ConstantImageSource[self.CPUImageType].New()
        projection.SetSpacing((self.system_config.pixel_size[0], self.system_config.pixel_size[1], 1))
        projection.SetSize((self.system_config.detector_pixels_w,
                            self.system_config.detector_pixels_h,
                            len(self.system_config.effective_angles)))
        projection.SetOrigin((-(self.system_config.detector_pixels_w-1) / 2 * self.system_config.pixel_size[0],
                      -(self.system_config.detector_pixels_h-1) / 2 * self.system_config.pixel_size[1],
                      0.0))
        projection.SetConstant(1)

        FOVFilterType = rtk.FieldOfViewImageFilter[self.CPUImageType, self.CPUImageType]
        fieldofview = FOVFilterType.New()
        fieldofview.SetInput(0, source.GetOutput())
        fieldofview.SetProjectionsStack(projection.GetOutput())
        fieldofview.SetMask(True)
        fieldofview.SetDisplacedDetector(True)
        fieldofview.SetGeometry(self.geometry)
        fieldofview.Update()

        fov = fieldofview.GetOutput()

        fov_sitk = self.itk_to_sitk(fov)

        self.fov_img = sitk.Cast(fov_sitk, sitk.sitkUInt8)

    def construct_geometry(self, angles):
        """Construct RTK geometry from gantry angles.

        Args:
            angles: Gantry angles in degrees (0-360 convention)

        Note: A +180 offset is applied to match the DRR projector's coordinate
        convention (see generate_projections.py create_projections method).
        """
        logger.info("construct geometry")

        # Normalize all angles to [0, 360) range
        # angles = np.asarray(angles) % 360

        self.geometry = rtk.ThreeDCircularProjectionGeometry.New()

        # Get geometry data for per-angle offsets
        geo_data = self.system_config.geometry_data

        for i, angle in enumerate(angles):
            # Get per-angle offsets using the original angle (for XML geometry matching)
            if self.system_config.has_xml_geometry:
                # Use angle-based lookup for XML geometry
                offset_x, offset_y = geo_data.get_offsets_at_angle(angle)
            else:
                # Use index-based lookup for config-based geometry
                if i < geo_data.num_projections:
                    offset_x, offset_y = geo_data.get_offsets_at_index(i)
                else:
                    offset_x, offset_y = geo_data.get_offsets_at_index(geo_data.num_projections - 1)

            # Use effective distances (from XML if available, else config)
            # Convert to native Python floats for RTK compatibility
            # XML offsets are already in RTK convention, config offsets need negation
            if self.system_config.has_xml_geometry:
                proj_offset_x = float(offset_x)
                proj_offset_y = float(offset_y)
            else:
                proj_offset_x = float(-offset_x)
                proj_offset_y = float(-offset_y)

            # Apply vendor-specific angle offset to match _drr_alpha() in generate_projections.py.
            # Elekta XVI angles need +180 to align with the DRR coordinate system;
            # Varian OBI angles are already aligned (no offset).
            is_elekta = self.system_config.physics.filtration_material.lower() == 'al'
            physical_angle = (angle + 180) % 360 if is_elekta else float(angle + 90.0) % 360

            self.geometry.AddProjection(
                float(self.system_config.effective_source_origin_distance),
                float(self.system_config.effective_source_detector_distance),
                float(physical_angle),
                proj_offset_x,
                proj_offset_y,
                0.0,             # in-plane angle
                0.0              # out-of-plane angle
            )

        self.geometry.Update()

    def correct_and_crop(self, recon: sitk.Image):
        recon = sitk.Multiply(recon, UINT16_MAX)
        recon = sitk.Add(recon, -1000.0)
        recon = sitk.Clamp(recon, lowerBound=-1000.0, upperBound=3071.0, outputPixelType=sitk.sitkFloat32)
        recon = sitk.Cast(recon, sitk.sitkInt16)
        if self.fov_img:
            return sitk.Mask(recon, self.fov_img, -1024)
        return recon

    def read_projections(self, proj_dir:Path)-> Tuple["rtk.Image|rtk.CudaImage", List[float]]:
        """Read projection files by index.

        Files are named {index:04d}.mhd and sorted numerically.
        Angles are taken from system_config.effective_angles.
        """
        proj_dir = Path(proj_dir)
        filePaths = list(proj_dir.glob("*.mhd"))

        if len(filePaths) == 0:
            raise FileNotFoundError(f"No projection files (*.mhd) found in: {proj_dir}")

        # Parse index from filename and sort
        file_data = []  # List of (file, index)
        for file in filePaths:
            try:
                idx = int(file.stem)
                file_data.append((file, idx))
            except ValueError:
                logger.warning(f"Skipping file with non-numeric name: {file.name}")

        if len(file_data) == 0:
            raise FileNotFoundError(
                f"Projection files found in {proj_dir}, but none have numeric stems (expected 0000.mhd, 0001.mhd, ...)"
            )

        file_data.sort(key=lambda x: x[1])
        files = [str(item[0].absolute()) for item in file_data]

        # Get angles from config (indexed by file order)
        angles = self.system_config.effective_angles[:len(file_data)]

        if len(file_data) != len(self.system_config.effective_angles):
            logger.warning(f"Projection count mismatch: found {len(file_data)} files "
                          f"but config has {len(self.system_config.effective_angles)} angles")

        projReader = rtk.ProjectionsReader[self.CPUImageType].New()
        projReader.SetFileNames(files)
        projReader.SetSpacing((self.system_config.pixel_size[0], self.system_config.pixel_size[1], 1))
        projReader.SetOrigin((-0.5*(self.system_config.detector_pixels_w-1) * self.system_config.pixel_size[0],
                              -0.5*(self.system_config.detector_pixels_h-1) * self.system_config.pixel_size[1],
                              0))
        projReader.Update()

        if self.gpu:
            projections = self.GPUImageType.New()
            projections.SetPixelContainer(projReader.GetOutput().GetPixelContainer())
            projections.CopyInformation(projReader.GetOutput())
            projections.SetBufferedRegion(projReader.GetOutput().GetBufferedRegion())
            projections.SetRequestedRegion(projReader.GetOutput().GetRequestedRegion())
            return (projections, angles)


        return (projReader.GetOutput(), angles)

    def build_graph(self, gpu=False):

        logger.info("Start building graph")

        ImageType = rtk.Image[itk.F,3]
        if self.gpu:
            ImageType = rtk.CudaImage[itk.F,3]

            self.offsetFilter = rtk.CudaDisplacedDetectorImageFilter.New()
            self.offsetFilter.SetGeometry(self.geometry)
            self.parkerFilter = rtk.CudaParkerShortScanImageFilter.New()
        else:
            self.offsetFilter = rtk.DisplacedDetectorImageFilter[ImageType].New()
            self.offsetFilter.SetGeometry(self.geometry)
            self.parkerFilter = rtk.ParkerShortScanImageFilter[ImageType].New()

        self.parkerFilter.SetInput(self.offsetFilter.GetOutput())
        self.parkerFilter.SetGeometry(self.geometry)
        self.parkerFilter.InPlaceOff()
        self.parkerFilter.SetAngularGapThreshold(20 * 3.14159265359 / 180.0)

        self.constantImageSource = rtk.ConstantImageSource[ImageType].New()
        self.constantImageSource.SetOrigin( self.system_config.recon_origin )
        self.constantImageSource.SetSpacing( self.system_config.recon_spacing )
        self.constantImageSource.SetSize( self.system_config.recon_size )
        self.constantImageSource.SetConstant(0.)
        # FDK reconstruction
        FDKType = rtk.FDKConeBeamReconstructionFilter
        if gpu:
            FDKType = rtk.CudaFDKConeBeamReconstructionFilter
        self.feldkamp = FDKType.New()
        self.feldkamp.SetInput(0, self.constantImageSource.GetOutput())
        self.feldkamp.SetInput(1, self.parkerFilter.GetOutput())
        self.feldkamp.SetGeometry(self.geometry)
        # Vendor-specific ramp filter settings (matched to COBRA preprocessing):
        #   Varian OBI:  Hann 0.5 / 0.5  (aggressive smoothing, high noise raw data)
        #   Elekta XVI:  Hann 0.99 / 0.0 (near-Ram-Lak in x, no Y smoothing)
        is_varian = self.system_config.physics.filtration_material.lower() == 'ti'
        if is_varian:
            self.feldkamp.GetRampFilter().SetTruncationCorrection(0.5)
            self.feldkamp.GetRampFilter().SetHannCutFrequency(0.5)
            self.feldkamp.GetRampFilter().SetHannCutFrequencyY(0.5)
        else:
            self.feldkamp.GetRampFilter().SetTruncationCorrection(0.2)
            self.feldkamp.GetRampFilter().SetHannCutFrequency(0.99)
            self.feldkamp.GetRampFilter().SetHannCutFrequencyY(0.0)
        self.feldkamp.SetProjectionSubsetSize(64)


    @log_time(logger)
    def fdk(self, proj):
        logger.info("Reconstruction started.")
        self.offsetFilter.SetInput(proj)
        self.feldkamp.Update()
        if self.gpu:
            fdk = rtk.Image[itk.F,3].New()
            fdk.SetPixelContainer(self.feldkamp.GetOutput().GetPixelContainer())
            fdk.CopyInformation(self.feldkamp.GetOutput())
            fdk.SetBufferedRegion(self.feldkamp.GetOutput().GetBufferedRegion())
            fdk.SetRequestedRegion(self.feldkamp.GetOutput().GetRequestedRegion())
            return fdk
        return self.feldkamp.GetOutput()

    def reconstruct(self, proj_dir):
        proj_reader, angles = self.read_projections(proj_dir)
        if self.geometry is None:
            self.construct_geometry(angles)
        recon = self.fdk(proj_reader)
        sitk_img = self.itk_to_sitk(recon)
        sitk_img = self.correct_and_crop(sitk_img)
        return sitk_img

    def itk_to_sitk(self, img):
        sitk_img = sitk.GetImageFromArray(itk.array_from_image(img))

        sitk_img.SetDirection((0.0, 0.0, -1.0,
                               -1.0, 0.0, 0.0,
                               0.0, -1.0, 0.0))

        sitk_img_corrected = sitk.DICOMOrient(sitk_img)
        origin = getattr(self, 'export_origin', None)
        spacing = getattr(self, 'export_spacing', None)
        if origin is None:
            self.export_origin = np.array(self.system_config.recon_origin)[[0, 2, 1]]
            self.export_spacing = np.array(self.system_config.recon_spacing)[[0, 2, 1]]
            origin = self.export_origin
            spacing = self.export_spacing
        sitk_img_corrected.SetOrigin(origin)
        sitk_img_corrected.SetSpacing(spacing)

        return sitk_img_corrected


    def save(self, recon, output_path:Path, file_name:str='recon'):
        if not output_path.exists():
            output_path.mkdir(exist_ok=True, parents=True)

        sitk.WriteImage(recon, output_path/f'{file_name}_0000.nii.gz')
