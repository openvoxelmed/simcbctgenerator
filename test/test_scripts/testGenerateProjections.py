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
import argparse
sys.path.append(str(Path(__file__).parent))
from testConfigPatient import MODES
from simcbctgenerator.generate_4d_ct import FourDCTGenerator
from simcbctgenerator.patient import Patient
from simcbctgenerator.generate_projections import DRRGenerator

import numpy as np
np.random.seed(42)  # for reproducibility

if sys.platform == 'linux':
    import os
    from PyQt5.QtCore import QLibraryInfo
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )

def main(mode):
    patient = Patient(mode.patient_config, mode.data_path)

    generator = FourDCTGenerator(mode.motion_config)
    generator.initialize(patient)

    drr_generator = DRRGenerator(mode.projections_path,
                                system_config=mode.system_config)

    #drr_generator.generate_DRRs()
    drr_generator.generate_all_projections(generator.patient, generator)

    projs = sorted(filter(lambda x: x.suffix == ".mhd", mode.projections_path.iterdir()), key=lambda x: int(x.stem))
    import SimpleITK as sitk
    proj_array = []
    for proj in projs:

        img = sitk.GetArrayFromImage(sitk.ReadImage(str(proj)))
        proj_array.append((img*100).astype(np.uint16))

    sitk.WriteImage(sitk.GetImageFromArray(np.stack(proj_array, 0)), 'img_16.mha')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Test projection generation")
    argparser.add_argument('--dummy', action='store_true', help='Use dummy data')
    argparser.add_argument('--thorax', action='store_true', default=False, help='Use thorax data')
    argparser.add_argument('--XVI', action='store_true', help='Use XVI data')
    argparser.add_argument('--SYNRAD', action='store_true', help='Use SYNRAD data')
    argparser.add_argument('--data_path', type=str, help='path to the data')
    args = argparser.parse_args()

    if args.data_path is None:
        if args.dummy:
            if not Path('./test/test_data').exists():
                import tarfile
                with tarfile.open('test_data.tar.gz', 'r:gz') as tar:
                    tar.extractall(path='test/test_data')
            else:
                print('cached test data used.')

        elif args.thorax:
            pass


        else:
            raise ValueError("data_path was not provided.")

    if args.dummy:
        print("Testing dummy dataset...")
        main(MODES['dummy'])

    elif args.thorax:
        print("Testing thorax dataset...")
        main(MODES['thorax'])
    elif args.XVI:

        print("Testing XVI dataset...")
        path = Path(args.data_path)
        if not path.exists():
            raise FileNotFoundError(f'{args.data_path} does not exist.')
        mode = MODES['XVI']
        mode.data_path = path
        main(mode)
    elif args.SYNRAD:
        print("Testing SYNRAD dataset...")
        path = Path(args.data_path)
        if not path.exists():
            raise FileNotFoundError(f'{args.data_path} does not exist.')
        mode = MODES['SYNRAD']
        mode.data_path = path
        main(mode)
