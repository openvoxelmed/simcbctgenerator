"""Dataset-specific loaders that construct image-centric Patient objects."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
from DicomRTTool.ReaderWriter import DicomReaderWriter

from simcbctgenerator.organ_mask_generator import (
    MOTION_SURROGATE_ABDOMEN_ORGANS,
    MOTION_SURROGATE_PELVIS_ORGANS,
    MOTION_SURROGATE_THORAX_ORGANS,
)
from simcbctgenerator.preprocessing.elekta_cbct_exporter import get_sitk_image_from_xvi
from simcbctgenerator.utils.config import DUMMY, SYNRAD, XVI, Errors, ImageCenter, PatientConfig

logger = logging.getLogger(__name__)


def _apply_export_structure_config(patient, mask: sitk.Image) -> None:
    mask_array = sitk.GetArrayFromImage(mask)
    patient.dicom_reader.annotation_handle = mask
    patient.dicom_reader.mask_dictionary = {}
    patient.dicom_reader.Contour_Names = []

    for index, structure in enumerate(patient.config.export_structures):
        if (mask_array == index + 1).sum() == 0:
            logger.warning(f"Patient error: {Errors.NOT_ALL_STRUCTURES.name}, missing structure: {structure}")
            continue
        annotation = sitk.Cast(sitk.GetImageFromArray((mask_array == index + 1).astype(np.uint8)), sitk.sitkUInt8)
        annotation.CopyInformation(mask)
        patient.dicom_reader.mask_dictionary[structure] = annotation
        patient.dicom_reader.Contour_Names.append(structure)

    mask_background = mask_array == 0
    patient.dicom_reader.mask = np.zeros((*mask_array.shape, len(patient.config.export_structures) + 1), dtype=np.uint8)
    patient.dicom_reader.mask[..., 0] = mask_background
    for index, structure in enumerate(patient.config.export_structures):
        if structure in patient.dicom_reader.Contour_Names:
            patient.dicom_reader.mask[..., index + 1] = mask_array == index + 1


def _maybe_autosegment(patient, image: sitk.Image) -> sitk.Image:
    if not patient.config.use_totalsegmentator:
        return None

    logger.info("Generating organ masks using TotalSegmentator")
    from simcbctgenerator.organ_mask_generator import OrganMaskGenerator

    generator = OrganMaskGenerator(fast_mode=True, device="gpu")
    organ_masks = generator.generate_multi_organ_masks(image, patient.config.export_structures)

    for structure in patient.config.export_structures:
        if structure in organ_masks:
            patient.dicom_reader.mask_dictionary[structure] = organ_masks[structure]
            patient.dicom_reader.Contour_Names.append(structure)
        else:
            logger.warning(f"TotalSegmentator did not generate mask for: {structure}")

    mask = generator.create_combined_mask(organ_masks, patient.config.priority)
    patient.dicom_reader.annotation_handle = mask
    return mask


def _infer_motion_surrogate(patient) -> None:
    thorax_organs = set(MOTION_SURROGATE_THORAX_ORGANS)
    pelvis_organs = set(MOTION_SURROGATE_PELVIS_ORGANS)
    abdomen_organs = set(MOTION_SURROGATE_ABDOMEN_ORGANS)
    loaded_organs = set(patient.dicom_reader.mask_dictionary.keys())

    if abdomen_organs.issubset(loaded_organs):
        patient.motion_surrogate = {organ: patient.dicom_reader.mask_dictionary[organ] for organ in abdomen_organs}
    elif thorax_organs.issubset(loaded_organs):
        patient.motion_surrogate = {organ: patient.dicom_reader.mask_dictionary[organ] for organ in thorax_organs}
    elif pelvis_organs.issubset(loaded_organs):
        patient.motion_surrogate = patient.dicom_reader.mask_dictionary["bowel"]


class PatientLoader:
    """Base dataset loader."""

    def load(self, patient, path: Path) -> None:
        raise NotImplementedError

    def load_cbct(
        self,
        patient,
        *,
        apply_correction: bool = True,
        return_transform: bool = False,
        return_projections: bool = False,
    ) -> dict[str, Any]:
        """Load a previously acquired CBCT for *patient*.

        Returns a dict with ``img`` (``sitk.Image`` or ``None``) and
        optionally ``transform`` / ``projections`` when requested.
        """
        logger.error(
            f"CBCT loading is not supported for modality: {patient.config.image_modality}"
        )
        return {}

    def save_real_cbct(self, patient, path: Path) -> None:
        """Export a previously acquired CBCT to *path* as ``<patient_id>.nii.gz``."""
        logger.warning(
            f"save_real_cbct is not implemented for modality: {patient.config.image_modality}"
        )


def _load_nifti_cbct(
    patient,
    cbct_file: Path,
    return_transform: bool,
    return_projections: bool,
) -> dict[str, Any]:
    """Shared reader for NIfTI/MHA CBCT files (SYNRAD / DUMMY)."""
    values: dict[str, Any] = {}
    if not cbct_file.exists():
        logger.warning(f"CBCT file not found: {cbct_file}")
        return values

    logger.info(f"Loaded CBCT image for patient {patient.id}")
    values["img"] = sitk.ReadImage(str(cbct_file))
    if return_transform:
        values["transform"] = None
    if return_projections:
        values["projections"] = None
    return values


class _ImageFilePatientLoader(PatientLoader):
    def _load_images(self, patient, image_path: Path, mask_path: Path | None) -> None:
        patient.dicom_reader = DicomReaderWriter(description="Examples", arg_max=True, verbose=False)
        image = sitk.ReadImage(image_path)
        patient.dicom_reader.dicom_handle = image
        patient.dicom_reader.ArrayDicom = sitk.GetArrayFromImage(image).astype(np.float32)
        patient.dicom_reader.Contour_Names = []
        patient.dicom_reader.mask_dictionary = {}

        mask = _maybe_autosegment(patient, image)
        if mask is None:
            if mask_path is not None and mask_path.exists():
                mask = sitk.ReadImage(mask_path)
            else:
                logger.warning("mask_dir was not provided. Fallback to empty image.")
                mask = sitk.GetImageFromArray(np.zeros_like(sitk.GetArrayFromImage(image)))
                mask.CopyInformation(image)
            _apply_export_structure_config(patient, mask)
        else:
            _apply_export_structure_config(patient, mask)

        _infer_motion_surrogate(patient)
        patient.set_projector_geometry()
        patient.all_rois = set(organ.lower() for organ in patient.dicom_reader.Contour_Names)
        intersect = patient.export_structures_set.intersection(patient.all_rois)
        if len(intersect) != len(patient.export_structures_set):
            logger.warning(f"Patient error: {Errors.NOT_ALL_STRUCTURES.name}, available structures: {patient.all_rois}")
            return

        patient.original_origin = np.array(patient.ct_image.GetOrigin())
        patient._valid = True


class SynthRadPatientLoader(_ImageFilePatientLoader):
    def load(self, patient, path: Path) -> None:
        modality = patient.config.image_modality.value
        image_path = path / modality.image
        mask_path = path / modality.segmentation if modality.segmentation is not None else None
        self._load_images(patient, image_path, mask_path)

    def load_cbct(
        self,
        patient,
        *,
        apply_correction: bool = True,
        return_transform: bool = False,
        return_projections: bool = False,
    ) -> dict[str, Any]:
        modality = patient.config.image_modality.value
        cbct_name = modality.cbct or "cbct.nii.gz"
        cbct_file = patient.path / cbct_name
        return _load_nifti_cbct(patient, cbct_file, return_transform, return_projections)


class DummyPatientLoader(_ImageFilePatientLoader):
    def load(self, patient, path: Path) -> None:
        modality = patient.config.image_modality.value
        image_path = modality.ct_dir / modality.image
        mask_path = modality.ct_dir / modality.segmentation if modality.segmentation is not None else None
        self._load_images(patient, image_path, mask_path)

    def load_cbct(
        self,
        patient,
        *,
        apply_correction: bool = True,
        return_transform: bool = False,
        return_projections: bool = False,
    ) -> dict[str, Any]:
        modality = patient.config.image_modality.value
        cbct_name = modality.cbct or "cbct.mhd"
        cbct_file = Path(modality.ct_dir) / cbct_name
        return _load_nifti_cbct(patient, cbct_file, return_transform, return_projections)


class XVIPatientLoader(PatientLoader):
    def load(self, patient, path: Path) -> None:
        patient.dicom_reader.walk_through_folders(patient.ct_dir)
        patient.all_rois = set(organ.lower() for organ in patient.dicom_reader.return_rois(print_rois=False))
        intersect = patient.export_structures_set.intersection(patient.all_rois)
        if len(intersect) != len(patient.export_structures_set):
            logger.warning(f"Patient error: {Errors.NOT_ALL_STRUCTURES.name}, available structures: {patient.all_rois}")
            return

        if patient.config.image_modality.value.image_center == ImageCenter.PLANISOCENTER:
            plan_files = list(patient.plan_dir.iterdir())
            if len(plan_files) > 1 and not patient.allow_multi_plan:
                logger.warning(f"Patient error: {Errors.MULTIPLE_PLANS.name}")
                return
            if len(plan_files) == 0:
                logger.warning(f"Patient error: {Errors.NO_PLAN.name}")
                return
            dcmplan = pydicom.dcmread(plan_files[0])
            iso_center = np.array(dcmplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition)
        elif patient.config.image_modality.value.image_center == ImageCenter.IMAGECENTER:
            iso_center = np.array(patient.dicom_reader.dicom_handle.GetOrigin()) + np.array(patient.dicom_reader.ArrayDicom.shape[::-1]) / 2 * np.array(patient.dicom_reader.dicom_handle.GetSpacing())
        else:
            raise ValueError(f"Image center [{patient.config.image_modality.value.image_center}] not defined.")

        contour_names_to_load = list(patient.config.export_structures)
        cm_mask = patient.config.cm_mask
        if cm_mask and cm_mask.lower() not in {s.lower() for s in contour_names_to_load}:
            if cm_mask.lower() in patient.all_rois:
                contour_names_to_load.append(cm_mask)
                logger.info(f"Loading cm_mask structure '{cm_mask}' from RTStruct for contrast media correction")
            else:
                logger.info(
                    f"cm_mask structure '{cm_mask}' not found in RTStruct "
                    f"(available: {sorted(patient.all_rois)}); will fall back to TotalSegmentator if needed"
                )

        patient.dicom_reader.set_contour_names_and_associations(contour_names=contour_names_to_load)
        patient.dicom_reader.get_images_and_mask()
        _infer_motion_surrogate(patient)
        patient.set_projector_geometry(iso_center=iso_center)
        patient._valid = True

    def load_cbct(
        self,
        patient,
        *,
        apply_correction: bool = True,
        return_transform: bool = False,
        return_projections: bool = False,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if not patient.cbct_dir.exists():
            logger.warning(f"CBCT dir not found for patient {patient.id}: {patient.cbct_dir}")
            return values

        for scan in patient.cbct_dir.iterdir():
            if not scan.is_dir():
                continue
            recon = scan / "Reconstruction"
            if not recon.exists():
                continue
            img, _, transform = get_sitk_image_from_xvi(
                recon,
                apply_correction=apply_correction,
                return_transform=return_transform,
            )
            if img is None:
                continue
            logger.info(f"Loaded CBCT image for patient {patient.id}")
            values["img"] = img
            if return_transform:
                values["transform"] = transform
            if return_projections:
                values["projections"] = self.load_projections(scan)
            return values

        logger.warning(f"No valid CBCT reconstruction found for patient {patient.id}")
        return values

    def save_real_cbct(self, patient, path: Path) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        if not patient.cbct_dir.exists():
            logger.warning(f"CBCT dir not found for patient {patient.id}")
            return
        for scan in patient.cbct_dir.iterdir():
            if not scan.is_dir():
                continue
            recon = scan / "Reconstruction"
            if not recon.exists():
                continue
            img, _ = get_sitk_image_from_xvi(recon)
            if img is not None:
                img = sitk.Cast(img, sitk.sitkInt16)
                sitk.WriteImage(img, path / (patient.id + ".nii.gz"))
                return

    @staticmethod
    def load_projections(scan: Path) -> list[dict[str, Any]]:
        """Read Elekta XVI ``.his`` projection frames from *scan*."""
        tree = ET.parse(scan / "_Frames.xml")
        root = tree.find("Frames")
        values: list[dict[str, Any]] = []
        for file in scan.iterdir():
            if file.suffix != ".his":
                continue
            index = int(file.stem.split(".")[0])
            element = {
                "uc": float(root[index - 1].find("UCentre").text),
                "vc": float(root[index - 1].find("VCentre").text),
                "angle": float(root[index - 1].find("GantryAngle").text),
                "img": np.fromfile(file, dtype=np.uint16)[50:].reshape((512, 512)),
            }
            values.append(element)
        return values


def get_patient_loader(config: PatientConfig) -> PatientLoader:
    modality = config.image_modality.value
    if isinstance(modality, XVI):
        return XVIPatientLoader()
    if isinstance(modality, SYNRAD):
        return SynthRadPatientLoader()
    if isinstance(modality, DUMMY):
        return DummyPatientLoader()
    raise ValueError(f"Unsupported patient modality: {config.image_modality}")
