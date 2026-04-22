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

from pathlib import Path
import json
import SimpleITK as sitk
import numpy as np
import argparse
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_stored_images_to_nnunet_format(path:Path, outdir:Path):
    img_path = path/"CBCT"
    label_path = path/"label"
    img_outdir = outdir/"imagesTr"
    label_outdir = outdir/"labelsTr"
    if not img_outdir.exists():
        img_outdir.mkdir(parents=True)
    if not label_outdir.exists():
        label_outdir.mkdir(parents=True)

    config = {"channel_names": {
                "0": "CBCT"},
              "labels": {
                "background": 0,
                "rectum": 1,
                "bowel": 2,
                "bladder": 3
                },
              "file_ending": ".nii.gz",
              "overwrite_image_reader_writer": "SimpleITKIO"}



    for file in tqdm(img_path.iterdir()):
        if file.suffix == ".mhd":
            img = sitk.ReadImage(str(file))
            label_imgs = []
            for i, (label, value) in enumerate(config['labels'].items()):
                if value == 0:
                    continue
                label_imgs.append(sitk.GetArrayFromImage(sitk.ReadImage(str(label_path/f"{file.stem}_{label}.mhd")))*value)

            label_background = np.zeros(label_imgs[0].shape)
            label_imgs.insert(0, label_background)

            label = np.argmax(np.stack(label_imgs), axis=0).astype(np.uint8)
            label_img = sitk.GetImageFromArray(label)
            label_img.CopyInformation(img)


            sitk.WriteImage(label_img, label_outdir/f"{file.stem}.nii.gz")
            sitk.WriteImage(img, img_outdir/f"{file.stem}_0000.nii.gz")

    config_path = outdir/"dataset.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return

def projections_to_volume(path:Path, outdir:Path):
    # Load the projections
    files = list(path.iterdir())
    filtered_files = list(filter(lambda x: x.suffix ==".mhd", files))
    filtered_files.sort(key=lambda x: float(x.name.split(".")[0]))
    projections = []
    for file in filtered_files: #filtered_files[::10]:
        projections.append(sitk.GetArrayFromImage(sitk.ReadImage(str(file))))

    volume = np.stack(projections)

    # Save the volume
    volume = sitk.GetImageFromArray(volume)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if (outdir/"projections.mhd").exists():
        (outdir/"projections.mhd").unlink()

    sitk.WriteImage(volume, outdir/"projections.mhd")


def convert_his_files(path:Path, outdir:Path):
    projections = []

    for file in path.iterdir():
        if file.is_dir():
            continue
        if file.suffix != ".his":
            continue

        data = np.fromfile(file, dtype=np.uint16)[50:]
        data = data.reshape((512, 512))/data.max()
        projections.append(data)

    volume = np.stack(projections)
    img = sitk.GetImageFromArray(volume)
    sitk.WriteImage(img, outdir/"his_projections.mhd")

def convert_mhd_to_his(projections_path: Path, outdir: Path, angles: np.ndarray = None, calibration_file: Path = None):
    """Convert .mhd projections to .his format with _Frames.xml"""
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Load projections
    files = [f for f in projections_path.iterdir() if f.suffix == ".mhd"]
    if not files:
        raise ValueError(f"No .mhd files found in {projections_path}")

    # Load calibration data if provided
    calibration_data = None
    if calibration_file and calibration_file.exists():
        calibration_data = np.load(calibration_file)
        logger.info(f"Loaded calibration file: {calibration_file}")
        logger.info(f"Calibration data shape: {calibration_data.shape}")

    # Extract angles from filenames and sort by angle
    if angles is None:
        file_angle_pairs = []
        for file in files:
            try:
                angle = float(file.stem)  # filename without extension
                # Normalize angle to [0, 360) range
                angle = angle % 360
                file_angle_pairs.append((file, angle))
            except ValueError:
                raise ValueError(f"Cannot extract angle from filename: {file.name}")

        # Sort by angle (smallest to largest)
        file_angle_pairs.sort(key=lambda x: x[1])
        files = [pair[0] for pair in file_angle_pairs]
        angles = np.array([pair[1] for pair in file_angle_pairs])

    # Create header template (50 uint16 values) based on detailed analysis
    header = np.zeros(50, dtype=np.uint16)

    # File Format Section (Pos 0-16)
    header[0] = 28672   # File Magic Number (0x7000)
    header[1] = 68      # Header Type/Size Marker
    header[2] = 0       # Reserved
    header[3] = 100     # Total Header Size (100 bytes)
    header[4] = 8       # Bit Depth
    header[5] = 32      # Data Type Info (0x0020)
    header[6] = 1       # Version Flag
    header[7] = 1       # Version Flag
    header[8] = 512     # Image Width (512 pixels)
    header[9] = 512     # Image Height (512 pixels)
    header[10] = 1      # Channels (Grayscale)
    header[16] = 4      # Data Type ID (uint16 identifier)

    # Reserved Section (Pos 17-33) - All zeros (UV offsets are in XML)

    # Timestamp Section (Pos 34-39) - From real data analysis
    header[34] = 48257  # Timestamp part 1
    header[35] = 46123  # Timestamp part 2 (constant)
    header[36] = 61864  # Timestamp part 3
    header[37] = 2870   # Timestamp part 4 (constant)
    header[38] = 57620  # Timestamp part 5 (constant)
    header[39] = 28163  # Timestamp part 6 (constant)

    # Frame ID Section (Pos 40-49) - Base values from analysis
    header[41] = 10415  # Acquisition metadata (constant)
    header[42] = 6059   # Acquisition metadata (constant)
    header[43] = 28164  # Acquisition metadata (constant)
    header[47] = 10415  # Acquisition metadata (constant)
    header[49] = 10415  # Acquisition metadata (constant)

    # First pass: find global min/max across all projections
    logger.info("Finding global min/max across all projections...")
    global_min = float('inf')
    global_max = float('-inf')

    for file in files:
        img = sitk.ReadImage(str(file))
        data = sitk.GetArrayFromImage(img).squeeze()

        # Apply calibration correction if available
        if calibration_data is not None:
            data = data / (calibration_data + 1e-6)  # Add small epsilon to avoid division by zero

        # For histogram matching, we still need to track the range but won't use it for normalization
        global_min = min(global_min, data.min())
        global_max = max(global_max, data.max())

    logger.info(f"Global range: {global_min:.6f} to {global_max:.6f}")

    # Global linear transformation to match real data distribution
    synthetic_range = global_max - global_min

    # Target real data range
    real_min = 64000#37330  # 5th percentile
    real_max = 65360  # 95th percentile
    real_range = real_max - real_min

    # Apply global linear transformation: scale and shift
    scale_factor = real_range / synthetic_range
    # Convert each projection
    for i, file in enumerate(files):
        img = sitk.ReadImage(str(file))
        data = sitk.GetArrayFromImage(img).squeeze()

        # Apply calibration correction if available
        if calibration_data is not None:
            data = data / (calibration_data + 1e-6)  # Add small epsilon to avoid division by zero
            logger.debug(f"Applied calibration correction to projection {i+1}")
        data_scaled = (data-global_min) * scale_factor + real_min

        # Clamp to uint16 range and convert
        data = np.clip(data_scaled, 0, 65535).astype(np.uint16)

        # Update projection-specific header values (positions 40, 46, 48)
        proj_id = 16504 + i * 100  # Base value + increment per projection
        header[40] = proj_id
        header[46] = proj_id
        header[48] = proj_id

        # Write .his file with correct naming pattern: {sequence:05d}.{DicomUID}.his
        his_file = outdir / f"{i+1:05d}.1.3.46.423632.synthetic.his"
        with open(his_file, 'wb') as f:
            header.tofile(f)
            data.tofile(f)

    # Create _Frames.xml
    create_frames_xml(outdir, len(files), angles)

    # Create reconstruction INI file
    create_reconstruction_ini(outdir)

def create_frames_xml(outdir: Path, num_frames: int, angles: np.ndarray):
    """Create _Frames.xml file with projection metadata matching real format"""
    if num_frames != len(angles):
        raise ValueError(
            f"num_frames ({num_frames}) does not match number of angles ({len(angles)})"
        )

    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<ProjectionSet>
  <!--Do not edit this file.-->
  <Station>
    <StationName>SyntheticCBCT</StationName>
    <LinacID>0000</LinacID>
  </Station>
  <Patient>
    <FirstName>Synthetic</FirstName>
    <MiddleName></MiddleName>
    <LastName>Patient</LastName>
    <ID>DUMMY</ID>
  </Patient>
  <Treatment>
    <ID>1:SYNTHETIC</ID>
    <Description>Synthetic CBCT Generation</Description>
    <DicomUID>1.2.3.4.5.6.7.8.9</DicomUID>
  </Treatment>
  <Field>
    <Id>***KV-IMAGES***</Id>
    <Description></Description>
  </Field>
  <Image>
    <kV>120</kV>
    <mA>40</mA>
    <ms>40</ms>
    <AcquisitionPresetName>Synthetic</AcquisitionPresetName>
    <Width>512</Width>
    <Height>512</Height>
    <Depth>16</Depth>
    <ReadoutOrientation>V</ReadoutOrientation>
    <DicomUID>1.3.46.423632.synthetic</DicomUID>
    <CTDIvol>15.5</CTDIvol>
    <CTDIPhantomType>Body Phantom (Length 15cm)</CTDIPhantomType>
    <AbsoluteTableLatPosIEC1217_MM>0.0</AbsoluteTableLatPosIEC1217_MM>
    <AbsoluteTableLongPosIEC1217_MM>0.0</AbsoluteTableLongPosIEC1217_MM>
    <AbsoluteTableVertPosIEC1217_MM>0.0</AbsoluteTableVertPosIEC1217_MM>
  </Image>
  <Frames>
"""

    for i, angle in enumerate(angles):
        xml_content += f"""    <Frame>
      <Seq>{i+1}</Seq>
      <DeltaMs>{i*182}</DeltaMs>
      <HasPixelFactor>False</HasPixelFactor>
      <PixelFactor>0</PixelFactor>
      <GantryAngle>{angle:.9f}</GantryAngle>
      <Exposed>True</Exposed>
      <MVOn>False</MVOn>
      <UCentre>115.0</UCentre>
      <VCentre>0.0</VCentre>
      <Inactive>False</Inactive>
    </Frame>
"""

    xml_content += """  </Frames>
</ProjectionSet>"""

    with open(outdir / "_Frames.xml", 'w') as f:
        f.write(xml_content)

def create_reconstruction_ini(outdir: Path):
    """Create reconstruction INI file for XVI reconstruction tool"""
    # Create Reconstruction subdirectory
    recon_dir = outdir / "Reconstruction"
    if not recon_dir.exists():
        recon_dir.mkdir(parents=True)

    # INI content based on real 1.3.46.423632.3384062024110114634188.32.INI
    ini_content = """[XVI]
AVLState=2
ImageSelectionSheetVisible=0
ReconstructionGroupVisible=0
DisplayGroupVisible=0
ReferencePresetGroupVisible=0
ImageGroupVisible=1
FilterPresetGroupVisible=1
AlignmentGroupVisible=0
CouchShiftGroupVisible=0
CalibrationGroupVisible=0
AdminCount=15
LogLevel=1
ElektaDatabaseSheetVisible=0
StayOnTop=0
RunMaximized=1
RunMinimized=0
Visible=0
Multithreaded=0
ZoomFix=1.2200
GUIRefreshInterval=40
InterpolatedZoom=1
SelectLatestScan=0
ReadOnly=0
IECAngleConvention=1
IECLinearConvention=2
HintHidePauseLength=5000
ShowMatchResultsDetailsBone=0
ShowMatchResultsDetailsGreyValue=0
NameOf6DOFSystem=HexaPOD
NameOf3DOFSystem=Precise Table
AdministrativeFilesDirectory=./Reconstruction/
ReferenceCacheDirectory=
ProjectionDirectory=./
ReconstructedScansDirectory=./Reconstruction/
StatusLineText=  Synthetic CBCT Data
[IDENTIFICATION]
PatientID=DUMMY
TreatmentID=1:SYNTHETIC
TreatmentUID=1.3.46.423632.synthetic
ReferenceUID=1.2.3.4.5.6.7.8.9
ScanUID=1.3.46.423632.synthetic.scan
FirstName=Synthetic
LastName=Patient
DOB=01.01.2000
TitleBarString=VolumeView-Rekonstruktion: Synthetic CBCT Data
[RECONSTRUCTION]
UseOnlineReconstruction=1
ProjectionsCalibrated=3
UseFlexMap=0
ProjectionAngleFile=Angle.%04d
ReconstructionOutputFile=1.3.46.423632.synthetic.scan.SCAN
AcquisitionDate=01.01.2024
AcquisitionTime=12:00:00.000
TubeMA=40.0000
TubeKV=120.0000
TubeKVLength=40.0000
FOV=medium
CollimatorU1=0.0000
CollimatorU2=0.0000
CollimatorV1=0.0000
CollimatorV2=0.0000
ReconstructionDataType=Float
ReconstructionFilter=Wiener
ReconstructionFilterParameters= 0.05,  90.00
Interpolate=Partial2
FPS=5.4900
ProjectionImageDimension=256
CameraWidth=512
GantryStartAngle=-180.0000
GantryStopAngle=180.0000
ShortScan=0
ReconstructionVoxelSize=0.1000
ReconstructionDimensionX=264
ReconstructionDimensionY=410
ReconstructionDimensionZ=410
ReconstructionOffsetX=-0.0000
ReconstructionOffsetY=-0.0000
ReconstructionOffsetZ=0.0000
ProjectionTimeout=240
ScatterCorrectionAlg=Uniform
ScatterCorrectionParameters= 0.24000
DisableMVScatterCorrection=0
FirstFrameImageLag=0.0250
Intrafraction=0
AcquisitionTrigger=Continuous
BeamHardeningExponent=1.0000
BlockSize=64
PreFilter=None
PanelReadoutOrientation=V
SkipColumnsLeft=4
SkipColumnsRight=4
SkipRowsTop=6
SkipRowsBottom=6
DisplacedDetectorTaperLength=9
ProjectionImageScaling=1
SaturationHandling=None
SaturationScalingFactor=0.367064074704880
VolumeRenormaliseTo=40000
ScaleOut=420.0000
OffsetOut=0.0000
KVFilter=F1
CollimatorName=M20
FloodImageOpenNorm=43302.0000
FloodImageOpenMA=20
FloodImageOpenMS=20
FloodImageOpenFile=
FloodImageFilterNorm=34054.0000
FloodImageFilterMA=25
FloodImageFilterMS=20
FloodImageFilterFile=
BlankCaps=1
ReconstructionProtocolName=M20 - Med_Res
ScanUID=1.3.46.423632.synthetic.scan
RespirationCorrelatedPhases=1
FourD_SortingMode=2
AmsShrd_PanelFix=0
ReconstructionStatus=1
[REFERENCE]
Online1.Level=1000
Online1.Window=500
Reference1.Level=1000
Reference1.Window=500
[1.3.46.423632.synthetic.scan.ALIGN]
Reconstruction=1.3.46.423632.synthetic.scan.SCAN
"""

    # Write INI file
    with open(recon_dir / "1.3.46.423632.synthetic.INI", 'w') as f:
        f.write(ini_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../projections_noise_test/', help='Path to the projections')
    parser.add_argument('--outpath', type=str, default='../output/', help='Path to the projections')
    args = parser.parse_args()

    projections_to_volume(Path(args.path), Path(args.outpath))
