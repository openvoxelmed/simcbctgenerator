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

from simcbctgenerator.patient import Patient
from simcbctgenerator.utils.config import (
    PhantomConfig, MotionConfig, PatientConfig,
    PhysicsConfig, GeometryConfig, ReconstructionVolumeConfig, CBCTSystemConfig
)
from pathlib import Path
from argparse import Namespace

def _download_file_from_google_drive(file_id: str, destination: Path):
    """
    Download a file from Google Drive handling large files that require confirmation.

    Args:
        file_id: Google Drive file ID
        destination: Path where to save the file
    """
    try:
        import requests
    except ImportError:
        print("Error: requests library not available.")
        print("Please install it with: uv pip install requests")
        raise

    import re

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _looks_like_binary(response) -> bool:
        # NIfTI gzip starts with 0x1f 0x8b; HTML interstitials come as text/html.
        content_type = response.headers.get("Content-Type", "")
        return "text/html" not in content_type.lower()

    def _parse_interstitial_form(html: str):
        """Extract the POST/GET target and hidden fields from Drive's virus-scan warning page."""
        action_match = re.search(r'action="([^"]+)"', html)
        if not action_match:
            return None, {}
        action = action_match.group(1).replace("&amp;", "&")
        fields = {m.group(1): m.group(2)
                  for m in re.finditer(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html)}
        return action, fields

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)

    # Path 1 (legacy): cookie-based download_warning token.
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Path 2 (modern): HTML interstitial form that posts to drive.usercontent.google.com.
    if not _looks_like_binary(response):
        html = response.text
        action, fields = _parse_interstitial_form(html)
        if action and fields:
            response = session.get(action, params=fields, stream=True)

    if not _looks_like_binary(response):
        raise RuntimeError(
            f"Google Drive did not return a binary payload for id={file_id}. "
            f"Got Content-Type={response.headers.get('Content-Type')!r}. "
            f"File may be quota-limited or require manual download."
        )

    save_response_content(response, destination)


def extract_test_data():
    """Download CT, mask and reference CBCT from Google Drive if they don't exist."""
    test_data_path = Path(__file__).parent.parent / 'test_data'
    test_data_path.mkdir(exist_ok=True)

    ct_file = test_data_path / "ct.nii.gz"
    mask_file = test_data_path / "mask.nii.gz"
    cbct_file = test_data_path / "cbct.nii.gz"

    # Download CT if it doesn't exist
    if not ct_file.exists():
        print("Downloading test CT from Google Drive...")
        gdrive_id = "1_-21jA2ayIAa1FuwW-6O1anqmF5E9_9k"
        try:
            _download_file_from_google_drive(gdrive_id, ct_file)
            print(f"Downloaded CT to {ct_file}")
        except Exception as e:
            print(f"Error downloading CT: {e}")
            print("\nManual download instructions:")
            print(f"1. Open: https://drive.google.com/file/d/{gdrive_id}/view")
            print("2. Download the file manually")
            print(f"3. Save it as: {ct_file}")
            raise
    else:
        print(f"CT file already exists: {ct_file}")

    # Download mask if it doesn't exist
    if not mask_file.exists():
        print("Downloading test mask from Google Drive...")
        mask_gdrive_id = "1TNr9E3lpNsRTkqpCd7xBnZL5ovnc9PCz"

        try:
            _download_file_from_google_drive(mask_gdrive_id, mask_file)
            print(f"Downloaded mask to {mask_file}")
        except Exception as e:
            print(f"Warning: Could not download mask file: {e}")
            print("Mask will need to be generated using autosegmentation")
            print("Run: python test/test_scripts/generate_dummy_labels.py")
    else:
        print(f"Mask file already exists: {mask_file}")

    # Download reference CBCT if it doesn't exist (used by regression examples/tests)
    if not cbct_file.exists():
        print("Downloading reference CBCT from Google Drive...")
        cbct_gdrive_id = "1AhsZ8Q93TVK4ncLlhSHpnYieMvdiTeK3"
        try:
            _download_file_from_google_drive(cbct_gdrive_id, cbct_file)
            print(f"Downloaded CBCT to {cbct_file}")
        except Exception as e:
            print(f"Warning: Could not download CBCT file: {e}")
            print("\nManual download instructions:")
            print(f"1. Open: https://drive.google.com/file/d/{cbct_gdrive_id}/view")
            print("2. Download the file manually")
            print(f"3. Save it as: {cbct_file}")
    else:
        print(f"CBCT file already exists: {cbct_file}")


def get_dummy_patient(region="PELVIS", use_autoseg=False) -> Patient:
    """Get a Patient object loaded with dummy patient data. Downloads data from Google Drive if needed.

    Args:
        region: "PELVIS" or "THORAX"
        use_autoseg: If True, use config with use_totalsegmentator=True

    Returns:
        Patient instance with loaded data
    """
    print(f"Loading dummy patient data using Patient class (region={region}, autoseg={use_autoseg})...")

    # First, ensure test data is downloaded (only needed for non-autoseg)
    if not use_autoseg:
        extract_test_data()

    # The DUMMY_CONFIG has paths relative to project root (test/test_data)
    import os
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent  # Go up to project root

    try:
        os.chdir(project_root)

        # Select config based on region and autoseg flag
        if region == "PELVIS":
            patient_config = DUMMY_CONFIG_AUTOSEG.patient_config if use_autoseg else DUMMY_CONFIG.patient_config
        else:  # THORAX
            patient_config = DUMMY_CONFIG2_AUTOSEG.patient_config if use_autoseg else DUMMY_CONFIG2.patient_config

        patient = Patient(patient_config, Path(''))

        if not patient.valid:
            raise ValueError("Dummy patient data could not be loaded - patient is not valid")

        print(f"Dummy patient loaded successfully: {patient.id}")
        return patient

    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


patient_config_dummy = PatientConfig(plan_dir= 'DICOM_PLAN',
ct_dir= 'CT_SET',
cbct_dir = '',
export_structures= ['bowel'],
priority= [1],
cm_mask='bowel',
use_totalsegmentator=False,  # Load from mask file
image_modality='dummy')

patient_config_dummy2 = PatientConfig(
    plan_dir= 'DICOM_PLAN',
    ct_dir= 'CT_SET',
    cbct_dir= 'CT_SET',
    export_structures= ['heart', 'aorta', 'lung', 'spine'],
    priority=[1, 2, 3, 4],
    cm_mask=None, #'heart',
    use_totalsegmentator=False,  # Load from mask file
    image_modality='dummy2')

# New configs with TotalSegmentator auto-segmentation
patient_config_dummy_autoseg = PatientConfig(
    plan_dir='DICOM_PLAN',
    ct_dir='CT_SET',
    cbct_dir='',
    export_structures=['bowel', 'bladder'],
    priority=[1, 2],
    cm_mask=None,
    use_totalsegmentator=True,  # Auto-generate using TotalSegmentator
    image_modality='dummy')

patient_config_dummy2_autoseg = PatientConfig(
    plan_dir='DICOM_PLAN',
    ct_dir='CT_SET',
    cbct_dir='CT_SET',
    export_structures=['heart', 'aorta', 'lung', 'spine'],
    priority=[1, 2, 3, 4],
    cm_mask=None,
    use_totalsegmentator=True,  # Auto-generate using TotalSegmentator
    image_modality='dummy2')

patient_config_xvi = PatientConfig(plan_dir= 'DICOM_PLAN',
ct_dir= 'CT_SET',
cbct_dir = 'IMAGES',
export_structures= ['bowel', 'rectum', 'bladder'],
priority= [1, 2, 3],
image_modality='xvi')

patient_config_synrad = PatientConfig(plan_dir= 'DICOM_PLAN',
ct_dir= 'CT_SET',
cbct_dir = 'IMAGES',
export_structures= ['bowel', 'rectum', 'bladder'],
priority= [1, 2, 3],
image_modality='synrad')

motion_config_pelvis = MotionConfig(motion_type=MotionConfig.MotionType.PELVIS,
                                amplitude_breathing=5,
                                # frequncy --> breathing
                                contour_name='bowel',
                                frequency_breathing=20,
                                frequency_heartbeat=0,
                                time_per_projection=0.18,  # seconds
                                uncertainty=0.02,  # seconds
                                )
motion_config_thorax = MotionConfig(
    motion_type = MotionConfig.MotionType.THORAX,
    amplitude_breathing= 20,
    amplitude_heart= 3,
    contour_name= '',
    frequency_breathing= 20, # breaths per minute (12-20)
    frequency_heartbeat= 80,
    time_per_projection= 0.18, # seconds
    uncertainty= 0.2, # seconds
    )


# Physics configurations
physics_config_elekta = PhysicsConfig(
    photon_flux=53216.9,#4.16e6,
    spr=1.6,
    mAs=1.6,
    kv=120.0,
    saturation_factor=2.0,
    bp_amplitude=1.07,
    bp_std=522.0,
    threads=16,
    max_block_index=512
)

physics_config_varian = PhysicsConfig(
    photon_flux=4.16e6,
    spr=1.3,
    mAs=1.2,
    kv=125.0,
    saturation_factor=1.0,
    bp_amplitude=1.06,
    bp_std=939.0,
    threads=16,
    max_block_index=512
)

# Geometry configurations
geometry_config_elekta = GeometryConfig(
    detector_offset=115.,
    source_origin_distance=1000.,
    source_detector_distance=1536.,
    detector_size_h=409.6,
    detector_size_w=409.6,
    detector_pixels_h=512,
    detector_pixels_w=512,
    start_angle=0.0,
    end_angle=360.0,
    angle_increments=0.545
)

geometry_config_varian = GeometryConfig(
    detector_offset=160.,
    source_origin_distance=1000.,
    source_detector_distance=1500.,
    detector_size_h=297.984,
    detector_size_w=397.312,
    detector_pixels_h=768,
    detector_pixels_w=1024,
    start_angle=0.0,
    end_angle=360.0,
    angle_increments=0.4
)

# Reconstruction volume configurations
reconstruction_volume_config_elekta = ReconstructionVolumeConfig(
    recon_size=[410, 66, 410],
    recon_origin=[-204.5, -130.0, -204.5],
    recon_spacing=[1.0, 4.0, 1.0]
)

reconstruction_volume_config_varian = ReconstructionVolumeConfig(
    recon_size=[512, 88, 512],
    recon_origin=[-231.999, -86.499, -231.999],
    recon_spacing=[0.908, 1.988, 0.908]
)

# System configurations (composite)
system_config_elekta = CBCTSystemConfig(
    physics=physics_config_elekta,
    geometry=geometry_config_elekta,
    reconstruction_volume=reconstruction_volume_config_elekta
)

system_config_varian = CBCTSystemConfig(
    physics=physics_config_varian,
    geometry=geometry_config_varian,
    reconstruction_volume=reconstruction_volume_config_varian
)

# Rectangular phantom configuration using real phantom data
phantom_config = PhantomConfig(
    phantom_path=str(Path(__file__).parent.parent / "test_data" / "phantom" / "phantom.mha"),
    intensity_factor=1.0,
    water_threshold=300.0,
    enhancement_factor=4.0,
    lower_threshold=-600.0,
    body_threshold=-200.0,
    gaussian_sigma=2.0,
    noise_range=(-35.0, 35.0)
)

DUMMY_CONFIG = Namespace(**{
    'patient_config'    : patient_config_dummy,
    'motion_config'     : motion_config_pelvis,
    'projections_path'  : Path('projections_test_dummy'),
    'reconstruction_path': Path('reconstruction_dummy'),
    'system_config'     : system_config_elekta,
    'phantom_config'    : phantom_config,
    'data_path'         : Path('../test_data/')
})

DUMMY_CONFIG_VARIAN = Namespace(**{
    'patient_config'    : patient_config_dummy,
    'motion_config'     : motion_config_pelvis,
    'projections_path'  : Path('projections_test_dummy_varian'),
    'reconstruction_path': Path('reconstruction_dummy_varian'),
    'system_config'     : system_config_varian,
    'phantom_config'    : phantom_config,
    'data_path'         : Path('../test_data/')
})

DUMMY_CONFIG2 = Namespace(**{
    'patient_config'    : patient_config_dummy2,
    'motion_config'     : motion_config_thorax,
    'projections_path'  : Path('projections_test_dummy2'),
    'reconstruction_path': Path('reconstruction_dummy2'),
    'system_config'     : system_config_elekta,
    'phantom_config': phantom_config,
    'data_path'         : Path('../test_data/')
})

DUMMY_CONFIG2_VARIAN = Namespace(**{
    'patient_config'    : patient_config_dummy2,
    'motion_config'     : motion_config_thorax,
    'projections_path'  : Path('projections_test_dummy2_varian'),
    'reconstruction_path': Path('reconstruction_dummy2_varian'),
    'system_config'     : system_config_varian,
    'phantom_config'    : phantom_config,
    'data_path'         : Path('../test_data/')
})

# Configs with TotalSegmentator auto-segmentation
DUMMY_CONFIG_AUTOSEG = Namespace(**{
    'patient_config'    : patient_config_dummy_autoseg,
    'motion_config'     : motion_config_pelvis,
    'projections_path'  : Path('projections_test_dummy_autoseg'),
    'reconstruction_path': Path('reconstruction_dummy_autoseg'),
    'system_config'     : system_config_elekta,
    'phantom_config'    : phantom_config,
    'data_path'         : Path('../test_data/')
})

DUMMY_CONFIG2_AUTOSEG = Namespace(**{
    'patient_config'    : patient_config_dummy2_autoseg,
    'motion_config'     : motion_config_thorax,
    'projections_path'  : Path('projections_test_dummy2_autoseg'),
    'reconstruction_path': Path('reconstruction_dummy2_autoseg'),
    'system_config'     : system_config_elekta,
    'phantom_config'    : phantom_config,
    'data_path'         : Path('../test_data/')
})

XVI_CONFIG = Namespace(**{
    'patient_config'    : patient_config_xvi,
    'motion_config'     : motion_config_pelvis,
    'projections_path'  : Path('projections_test_XVI'),
    'reconstruction_path': Path('reconstruction_XVI'),
    'system_config'     : system_config_elekta,
    'phantom_config'    : phantom_config,
    'data_path'         : None
})

SYNRAD_CONFIG = Namespace(**{
    'patient_config'    : patient_config_synrad,
    'motion_config'     : motion_config_pelvis,
    'projections_path'  : Path('projections_test_SYNRAD'),
    'reconstruction_path': Path('reconstruction_SYNRAD'),
    'system_config'     : system_config_elekta,
    'phantom_config'    : phantom_config,
    'data_path'         : None
})

MODES = {
    'dummy': DUMMY_CONFIG_VARIAN,  # Default (PELVIS, no autoseg)
    'pelvis': DUMMY_CONFIG,
    'thorax': DUMMY_CONFIG2,
    'pelvis_varian': DUMMY_CONFIG_VARIAN,
    'thorax_varian': DUMMY_CONFIG2_VARIAN,
    'pelvis_autoseg': DUMMY_CONFIG_AUTOSEG,
    'thorax_autoseg': DUMMY_CONFIG2_AUTOSEG,
    'XVI': XVI_CONFIG,
    'SYNRAD': SYNRAD_CONFIG
}
