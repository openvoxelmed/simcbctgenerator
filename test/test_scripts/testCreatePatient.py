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

sys.path.append(str(Path(__file__).parent))

from testConfigPatient import DUMMY_CONFIG

from simcbctgenerator.patient import Patient


if __name__ == "__main__":
    patient = Patient(DUMMY_CONFIG.patient_config, Path(''))
    if not patient.valid:
        raise ValueError('Patient not valid')
    ct = patient.ct_array
    mask = patient.mask_array
    # Do something with ct and mask
