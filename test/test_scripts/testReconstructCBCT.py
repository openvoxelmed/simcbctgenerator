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

from simcbctgenerator.patient import Patient
from simcbctgenerator.cbct_reconstruction import SyntheticCBCTReconstruction
from testConfigPatient import MODES
import matplotlib.pyplot as plt
import SimpleITK as sitk

if sys.platform == 'linux':
    import os
    from PyQt5.QtCore import QLibraryInfo
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )

def main(mode):

    patient = Patient(mode.patient_config, mode.data_path)
    cbct_reconstructor = SyntheticCBCTReconstruction(mode.system_config, gpu=True)
    recon = cbct_reconstructor.reconstruct(mode.projections_path)
    patient.save_masks(recon, output_path=mode.reconstruction_path, file_name='labels', mask=cbct_reconstructor.fov_img)
    volume = sitk.GetArrayFromImage(recon)

    slice_num = int(volume.shape[0]//2)

    plt.imshow(volume[slice_num], cmap='gray')
    plt.show()

    cbct_reconstructor.save(recon, mode.reconstruction_path, 'recon')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Test projection generation")
    argparser.add_argument('--dummy', action='store_true', help='Use dummy data')
    argparser.add_argument('--thorax', action='store_true', default=False, help='Use thorax data')
    argparser.add_argument('--XVI', action='store_true', help='Use XVI data')
    argparser.add_argument('--SYNRAD', action='store_true', help='Use SYNRAD data')
    argparser.add_argument('--data_path', type=str, default="thorax_recon", help='path to the data')
    args = argparser.parse_args()

    if args.data_path is None:
        if args.dummy:
            if not Path('./test/test_data').exists():
                import tarfile
                with tarfile.open('test_data.tar.gz', 'r:gz') as tar:
                    tar.extractall(path='test/test_data')
            else:
                print('cached test data used.')


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
