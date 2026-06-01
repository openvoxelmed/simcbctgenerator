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

"""Image-centric patient model for CBCT generation pipelines."""

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from DicomRTTool.ReaderWriter import DicomReaderWriter
from simcbctgenerator.utils.config import ImageCenter, PatientConfig
from pathlib import Path
import logging
import numpy as np
import SimpleITK as sitk
from simcbctgenerator.patient_setup import get_patient_loader

logger = logging.getLogger(__name__)


class Patient:
    """Patient data handler for CBCT generation pipeline."""

    def __init__(self, config: PatientConfig, path: Path, allow_multi_plan:bool=False):
        loaded = self.from_folder(config=config, path=path, allow_multi_plan=allow_multi_plan)
        self.__dict__.update(loaded.__dict__)

    @classmethod
    def _create_empty(
        cls,
        config: PatientConfig,
        path: Path,
        allow_multi_plan: bool,
        patient_id: str | None = None,
    ):
        patient = cls.__new__(cls)
        patient.config = config
        patient.path = Path(path)
        patient.id = patient_id or config.image_modality.value.get_id(patient.path)
        patient.plan_dir = patient.path / config.plan_dir
        patient.ct_dir = patient.path / config.ct_dir
        patient.cbct_dir = patient.path / config.cbct_dir
        patient.dicom_reader = DicomReaderWriter(arg_max=False, verbose=False)
        patient.export_structures_set = set(config.export_structures)
        patient.all_rois = set()
        patient._valid = False
        patient.allow_multi_plan = allow_multi_plan
        patient.motion_surrogate = None
        return patient

    @classmethod
    def from_folder(cls, config: PatientConfig, path: Path, allow_multi_plan: bool = False):
        patient = cls._create_empty(config=config, path=path, allow_multi_plan=allow_multi_plan)
        logger.info(f"construct patient {patient.id}")
        loader = get_patient_loader(config)
        loader.load(patient, Path(path))
        return patient

    @classmethod
    def from_images(
        cls,
        ct_image: sitk.Image,
        config: PatientConfig,
        mask_image: sitk.Image | None = None,
        reference_cbct: sitk.Image | None = None,
        patient_id: str = "challenge_patient",
    ):
        patient = cls._create_empty(config=config, path=Path(""), allow_multi_plan=False, patient_id=patient_id)

        ct_image = sitk.Image(ct_image)

        patient.dicom_reader = DicomReaderWriter(description="Challenge", arg_max=True, verbose=False)
        patient.dicom_reader.dicom_handle = ct_image
        patient.dicom_reader.ArrayDicom = sitk.GetArrayFromImage(ct_image).astype(np.float32)
        patient.dicom_reader.Contour_Names = []
        patient.dicom_reader.mask_dictionary = {}
        patient.dicom_reader.annotation_handle = mask_image

        if config.export_structures and not config.use_totalsegmentator and mask_image is None:
            raise ValueError(
                "export_structures is set but use_totalsegmentator=False and no mask_image provided. "
                "Either set use_totalsegmentator=True or provide a mask_image with labeled structures."
            )

        if config.export_structures:
            if config.use_totalsegmentator:
                from simcbctgenerator.organ_mask_generator import OrganMaskGenerator

                logger.info(f"Generating export structures using TotalSegmentator: {config.export_structures}")
                organ_generator = OrganMaskGenerator(fast_mode=True, device="gpu")
                organ_masks = organ_generator.generate_multi_organ_masks(ct_image, config.export_structures)
                combined_mask = organ_generator.create_combined_mask(organ_masks, config.priority)
                organ_names = config.export_structures
            else:
                logger.info(f"Loading export structures from provided mask image: {config.export_structures}")
                combined_mask = mask_image
                organ_names = config.export_structures

            patient.dicom_reader.mask_dictionary = {}
            for i, name in enumerate(organ_names):
                organ_array = (sitk.GetArrayFromImage(combined_mask) == i + 1).astype(np.uint8)
                organ_mask = sitk.GetImageFromArray(organ_array)
                organ_mask.CopyInformation(combined_mask)
                patient.dicom_reader.mask_dictionary[name] = sitk.Cast(organ_mask, sitk.sitkUInt8)

            patient.dicom_reader.Contour_Names = list(organ_names)
            patient.dicom_reader.annotation_handle = combined_mask
            patient.all_rois = set(organ_names)
            patient.export_structures_set = set(organ_names)

            mask_array = sitk.GetArrayFromImage(combined_mask)
            max_label = len(organ_names)
            patient.dicom_reader.mask = (mask_array[..., np.newaxis] == np.arange(0, max_label + 1)).astype(np.uint8)
        else:
            patient.dicom_reader.mask = None

        patient.set_projector_geometry(reference_cbct=reference_cbct)
        patient._valid = True
        logger.info(f"Created patient from CT image (ID: {patient_id})")
        return patient

    def set_projector_geometry(
        self,
        iso_center: np.ndarray | None = None,
        reference_cbct: sitk.Image | None = None,
    ) -> None:
        image = self.ct_image
        if reference_cbct is not None:
            image = reference_cbct

        if iso_center is None:
            if self.config.image_modality.value.image_center == ImageCenter.IMAGECENTER or reference_cbct is not None:
                size = np.array(image.GetSize())
                spacing = np.array(image.GetSpacing())
                origin = np.array(image.GetOrigin())
                iso_center = origin + ((size - 1) * spacing) / 2
            else:
                iso_center = np.array(image.GetOrigin())

        center = ((np.array(self.ct_image.GetSize()) - 1) / 2 * np.array(self.ct_image.GetSpacing()))
        if reference_cbct is not None:
            self.shifted_origin = -(np.array(self.ct_image.TransformPhysicalPointToIndex(np.array(iso_center).tolist())) - 1.0) * np.array(self.ct_image.GetSpacing())
        elif self.config.image_modality.value.image_center == ImageCenter.IMAGECENTER:
            self.shifted_origin = -center
        else:
            self.shifted_origin = -(np.array(self.ct_image.TransformPhysicalPointToIndex(np.array(iso_center).tolist())) - 1.0) * np.array(self.ct_image.GetSpacing())
        self.original_origin = np.array(self.ct_image.GetOrigin())
        self.ct_image.SetOrigin(np.asarray(self.shifted_origin).tolist())
        # The DRR projector expects the isocenter in the *shifted* coordinate frame
        # (where the CT is centred near the world origin).  Storing the raw LPS value
        # places the rotation axis tens of mm away from the CT centre.
        # After the shift above the true isocenter sits at ~(0,0,0) in that frame.
        self.iso_center = np.zeros(3)

    def correct_CM(self):
        if self.config is None or self.config.cm_mask is None:
            logger.error("config or config.cm_mask is not defined.")
            return
        ct = cp.asarray(self.ct_array)
        index = self.contour_names.index(self.config.cm_mask) + 1
        mask = cp.asarray(self.mask_array[..., index])

        cm = mask.astype(bool) & (ct > 50)

        blurred = gaussian_filter(cm.astype(cp.float32), (0.0, 1.0, 1.0), mode='constant', cval=0)

        noise = cp.random.normal(loc=1.0, scale=0.02, size=cm.shape)

        overhead = blurred*ct*noise

        cm_free_ct = cp.asnumpy(ct-overhead*0.92)

        cm_free_ct = sitk.GetImageFromArray(cm_free_ct)
        spacing = self.ct_image.GetSpacing()
        cm_free_ct.SetSpacing(spacing)
        cm_free_ct.SetOrigin(self.ct_image.GetOrigin())
        self.ct_image = cm_free_ct
        self.ct_array = sitk.GetArrayFromImage(cm_free_ct).astype(np.float32)

    def shift_origin_to_iso_center(self, structure_name:str):
        if not self._valid:
            logger.warning('patient not valid')
            return

        if structure_name.lower() not in self.contour_names:
            logger.error(f'{structure_name} not in contours. Available contours: {self.contour_names}')
            return

        mask = self.mask_dictionary[structure_name.lower()]
        mask.SetOrigin(self.shifted_origin)
        return mask

    def resample_mask(self, img:sitk.Image, structure_name:str):
        mask = self.shift_origin_to_iso_center(structure_name)
        resampled_mask = sitk.Resample(mask, transform=sitk.Transform(), referenceImage=img, interpolator=sitk.sitkNearestNeighbor)
        return resampled_mask

    def resample_ct(self, img:sitk.Image, mask:sitk.Image|None=None):
        ct: sitk.Image = self.ct_image
        current_origin = np.array(ct.GetOrigin())
        ct.SetOrigin(self.shifted_origin)
        resampled_ct = sitk.Resample(ct, transform=sitk.Transform(), referenceImage=img, interpolator=sitk.sitkLinear, defaultPixelValue=-1024)
        if mask is not None:
            resampled_ct = sitk.Mask(resampled_ct, mask, -1024)
        ct.SetOrigin(current_origin)
        resampled_ct = sitk.Cast(resampled_ct, sitk.sitkInt16)
        return resampled_ct

    def combined_label_mask(self) -> sitk.Image | None:
        """Return a multi-label image built from the patient's structures.

        Uses ``config.export_structures`` as the organ list and
        ``config.priority`` as the label values, delegating the actual
        combination to
        :meth:`OrganMaskGenerator.create_combined_mask`.  The result is on
        the CT grid; callers that need it on another grid (e.g. the CBCT)
        should resample it themselves.
        """
        from simcbctgenerator.organ_mask_generator import OrganMaskGenerator

        organ_masks: dict[str, sitk.Image] = {}
        priorities: list[int] = []
        for structure, value in zip(self.config.export_structures, self.config.priority):
            key = structure.lower()
            if key in self.mask_dictionary:
                organ_masks[structure] = self.mask_dictionary[key]
                priorities.append(value)
        if not organ_masks:
            return None
        combined = OrganMaskGenerator().create_combined_mask(organ_masks, priorities)
        combined.CopyInformation(self.ct_image)
        return combined

    def save_masks(self, img:sitk.Image, output_path:Path, file_name:str, mask:sitk.Image|None=None):
        if not output_path.exists():
            output_path.mkdir(exist_ok=True, parents=True)
        combined = self.combined_label_mask()
        if combined is None:
            logger.warning(f'no structures available to export for patient {self.id}')
            return None
        resampled = sitk.Resample(combined, referenceImage=img, interpolator=sitk.sitkNearestNeighbor)
        if mask is not None:
            resampled = sitk.Multiply(sitk.Cast(resampled, sitk.sitkUInt8), sitk.Cast(mask, sitk.sitkUInt8))
        sitk.WriteImage(resampled, output_path/(file_name+'.nii.gz'))
        return resampled

    def save_resampled_ct(self, path:Path, img:sitk.Image, mask:sitk.Image|None=None, anonymize:bool=False):
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)
        resampled_ct = self.resample_ct(img, mask=mask)
        if anonymize:
            sitk.WriteImage(resampled_ct, path / ('ct.nii.gz'))
        else:
            sitk.WriteImage(resampled_ct, path / (self.id+'.nii.gz'))

    @property
    def valid(self):
        return self._valid

    @property
    def ct_image(self) -> sitk.Image:
        return self.dicom_reader.dicom_handle

    @ct_image.setter
    def ct_image(self, value: sitk.Image):
        if not isinstance(value, sitk.Image):
            raise ValueError("ct_image must be a sitk.Image")
        self.dicom_reader.dicom_handle = value

    @property
    def mask_image(self):
        return self.dicom_reader.annotation_handle

    @property
    def ct_array(self):
        return self.dicom_reader.ArrayDicom

    @ct_array.setter
    def ct_array(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("ct_array must be a np.ndarray")
        self.dicom_reader.ArrayDicom = value

    @property
    def mask_array(self):
        return self.dicom_reader.mask

    @property
    def mask_dictionary(self):
        return self.dicom_reader.mask_dictionary

    @property
    def get_label_img(self):
        return self.dicom_reader.label_img

    @property
    def contour_names(self):
        return self.dicom_reader.Contour_Names
