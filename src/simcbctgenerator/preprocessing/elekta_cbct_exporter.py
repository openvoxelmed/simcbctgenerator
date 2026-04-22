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

import sys
from pathlib import Path
from xdrt import xdr_reader, read_as_simpleitk
from xdrt.cli.utils import write_dicom_image
from xdrt.xvi_reader import XVIReconstruction
import SimpleITK as sitk
import pydicom
from argparse import ArgumentParser
from simcbctgenerator.preprocessing.data_manager import recurse_folders
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

SCAN_TO_DICOM = np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]])

def readDICOMImage(path:Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(path))
    foruid = pydicom.dcmread(files[0])[0x0020, 0x0052].value
    sex = pydicom.dcmread(files[0])[0x0010, 0x0040].value
    reader.SetFileNames(files)
    img = reader.Execute()
    return img, foruid, sex

def eulerTransform(transMat, reference = None) -> sitk.Transform:
    # transMat (so3):
    # [ 1, -z,  y]
    # [ z,  1, -x]
    # [-y,  x,  1]

    rot = (transMat[:3,:3]@SCAN_TO_DICOM) # correct for orientation
    t = transMat[:3,3] * 10 # in mm

    rotation_center = (0, 0, 0)
    if reference is not None:
        rotation_center = reference.GetOrigin() + (np.array(reference.GetSize())/2)*reference.GetSpacing()
    theta_x, theta_y, theta_z = rot[2, 1], -rot[2, 0], rot[1, 0]#-rot[2, 0], rot[1, 0]
    rotInv = (R.from_euler('x', theta_x) *\
           R.from_euler('y', theta_y) *\
           R.from_euler('z', theta_z)).as_matrix().T
    tInv = np.round(- rotInv @ t, 2)
    return sitk.Euler3DTransform(rotation_center, -theta_x, -theta_y, -theta_z, tInv)

def resample(src, referenceImage, transform=None) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(referenceImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    if transform is not None:
        resampler.SetTransform(transform)
    return resampler.Execute(src)

def resample_to_spacing(image: sitk.Image, new_spacing: tuple = (1.0, 1.0, 4.0)) -> sitk.Image:
    """Resample image to specified spacing without reference image.

    Args:
        image: Input SimpleITK image
        new_spacing: Desired spacing in mm (x, y, z)

    Returns:
        Resampled image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate new size based on spacing ratio
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)#image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)

def get_sitk_image_from_xvi(path:Path, apply_correction:bool=False, return_transform:bool=False) -> (sitk.Image, dict):
    if not list(path.glob('*.SCAN')):
        logger.warning(f"No scan found in {path}")
        if return_transform:
            return None, None, None
        return None, None
    try:
        xvi_reconstruction = XVIReconstruction(path)#xdrt.read(str(scanFiles[0]))
    except RuntimeError as e:
        if hasattr(e, 'message'):
            logger.error(f"Runtime error: {e.message}")
        else:
            logger.error(f"Runtime error: {e}")
        logger.error(f'{path} does not contain the required information!')
        if return_transform:
            return None, None, None
        return None, None
    except UnicodeDecodeError:
        logger.error(f'{path} has an unicode error!')
        if return_transform:
            return None, None, None
        return None, None
    filename = xvi_reconstruction.scan.filename
    patient = xvi_reconstruction.patient

    if sys.platform == 'linux':
        xviCtFolder = Path(xvi_reconstruction._data_dict['XVI']['referencecachedirectory'].replace("\\", "/"))
    else:
        xviCtFolder = Path(xvi_reconstruction._data_dict['XVI']['referencecachedirectory'])
    parentFolder = xviCtFolder.parent.parent
    ctFolder = path.parent.parent.parent / xviCtFolder.relative_to(parentFolder)
    ctImage, foruid, sex = readDICOMImage(ctFolder)

    # TODO: This is not a proper encoding for all names in different character sets
    patient_name = f"{patient.last_name.strip()}^{patient.first_name.strip()}"
    modification_time = xvi_reconstruction.scan.date_time.strftime("%H%M%S")
    modification_date = xvi_reconstruction.scan.date_time.strftime("%Y%m%d")

    dicom_dict = {
        "0010|0020": patient.patient_id,
        "0010|0010": patient_name,  # patient_name,
        "0010|0030": xvi_reconstruction.patient.date_of_birth.strftime("%Y%m%d"),  # Date of birth
        "0010|0040": sex,
        "0018|5100": 'HFS',
        "0020|000d": xvi_reconstruction.scan.treatment_uid,
        "0020|000e": "1.2.826.0.1.3680043.2.1126." + modification_date + ".1" + modification_time,
        "0020|0010": xvi_reconstruction.scan.scan_uid,
        "0008|0021": modification_date,  # Series date
        "0008|0031": modification_time,  # Series time
        "0020|0052": foruid,
        "0008|0060": "CT",  # Modality
        "0008|0008": "DERIVED\\SECONDARY",
        "0008|0070":"ELEKTA",
        "0008|0080":"XVI",
        "0008|1030": "XVI Reconstruct CBCT - XDRT conversion",  # Study Description
        "0008|103e": "XVI Reconstruct CBCT - XDRT conversion",  # Series Description
        "0028|1050": str(xvi_reconstruction.scan.level - 1024),  # window center
        "0028|1051": str(xvi_reconstruction.scan.window),  # window width
        "0028|1052": "-1024",  # Intercept
        "0028|1053": "1",  # Slope
        "0028|1054": "HU"
    }

    # Read the XDR file
    xdr_image = xdr_reader.read(filename, stop_before_data=False)

    xdr_image = xdr_reader.postprocess_xdr_image(xdr_image, temporal_average=None, slope=None, intercept=None, cast='uint16', clip=True)
    sitk_image = read_as_simpleitk(xdr_image, lps_orientation=True, save_header=True)

    correction_matrix = np.array(xvi_reconstruction.scan.transform).reshape(4,4).T
    transform = eulerTransform(correction_matrix)

    if apply_correction:
        # Apply correction matrix
        resampled_image = resample(sitk_image, ctImage, transform)
    else:
        # Resample to 1x1x4 mm spacing
        resampled_image = resample_to_spacing(sitk_image, new_spacing=(1.0, 1.0, 4.0))

    corrected_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(resampled_image).astype(np.int16)-1024)
    corrected_image.CopyInformation(resampled_image)
    corrected_image.EraseMetaData('0028|1054')
    if return_transform:
        return corrected_image, dicom_dict, transform
    else:
        return corrected_image, dicom_dict

def storeXDRasDICOM(path: Path, outputdir:Path, apply_correction=False)->sitk.Image:

    sitk_image, dicom_dict = get_sitk_image_from_xvi(path, apply_correction=apply_correction)

    if sitk_image is None:
        return

    outputdir.mkdir(parents=True, exist_ok=True)
    write_dicom_image(sitk_image, outputdir, no_compression=True, metadata=dicom_dict)



def load_patient_list(path:str)-> np.ndarray:

    if (Path.cwd()/path).exists():
        with (Path.cwd()/path).open('r') as f:
            paths = [line.strip().split(',') for line in f.readlines()]
        return np.array(paths) # prevents array to be non iterable in case of one patient
    raise FileNotFoundError()

def run(patient_list, datadir, outdir, apply_correction, overwrite):
    paths = []
    patient_list = load_patient_list(patient_list)
    recurse_folders(datadir, paths)
    ids = list(map(lambda x: x.name.split('_')[1],paths))
    patient_paths_list = [(x,v) for (x,v) in zip(ids,paths)]
    patient_paths = {}
    for id, patient_path in patient_paths_list:
        if patient_paths.get(id, None) is not None:
            patient_paths[id].append(patient_path)
        else:
            patient_paths[id] = [patient_path]

    for cdl_id, patient_id in patient_list:
        if patient_id not in patient_paths.keys():
            logger.warning(f"Patient {patient_id} not found in {datadir}")
            continue

        for patient_path in patient_paths[patient_id]:
            patientdir = patient_path / 'IMAGES'
            for scan in patientdir.iterdir():
                if (outdir/cdl_id/scan.name).exists() and not overwrite:
                    logger.info(f"{outdir/cdl_id/scan.name} already exists")
                    continue
                elif (outdir/cdl_id/scan.name).exists() and overwrite:
                    logger.info(f"{outdir/cdl_id/scan.name} already exists")
                    for file in (outdir/cdl_id/scan.name).iterdir():
                       file.unlink()
                recon = scan / 'Reconstruction'
                storeXDRasDICOM(recon, outdir/cdl_id/scan.name, apply_correction=apply_correction)
            logger.info(f"successfully exported {patient_id} to {outdir/cdl_id} from {patient_path}")
    return

def getArgs():

    parser = ArgumentParser("XVI to dicom exporter")
    parser.add_argument('--patient_list', type=str, default='./lists/cdl001-032.txt')#required=True)
    parser.add_argument('--datadir', type=Path, default='/mnt/g/')#require=True)
    parser.add_argument('--outdir', type=Path, default='/mnt/d/export_cdl/')
    parser.add_argument('--apply_correction', action="store_true", help="apply clinical transform")
    parser.add_argument('--overwrite', action="store_true", help="overwrites images stored")

    args = parser.parse_args().__dict__
    return args

if __name__ == "__main__":

    args = getArgs()
    logger.info(f"Arguments: {args}")
    run(**args)
