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

import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)

def recurse_folders(path, paths):
    if not path.is_dir():
        return
    for folder in path.iterdir():
        try:
            if folder.is_dir():
                if 'patient_' in folder.name:
                    paths.append(folder)
                else:
                    recurse_folders(folder, paths)

        except PermissionError:
            pass

def main(path, results, db, selection):
    df = pd.read_excel(db)

    path = Path(path)
    results = Path(results)

    paths = []
    recurse_folders(path, paths)

    ids = list(map(lambda x: x.name.split("_")[-1], paths))
    df_id = pd.DataFrame(data=np.array([ids, paths]).T, columns=['IDB', 'Path'])
    df_merged = pd.merge(left=df, right=df_id, on='IDB')
    df_merged = df_merged[~df_merged['Gruppe'].isna()]

    df_merged['Gruppe'] = df_merged['Gruppe'].apply(lambda x: x.strip())

    df_filtered = df_merged[df_merged['Gruppe'] == 'Gruppe GYN']
    logger.debug(f"Unique IDB count: {df_filtered['IDB'].unique().shape}")
    logger.debug(f"Filtered data:\n{df_filtered}")

    df_filtered.to_csv(results, mode='a', header=not results.exists(), index=False)
    df_filtered['Path'].to_csv(selection, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data manager for Linac export")
    parser.add_argument('--drive', type=str, help='which drive to use')
    parser.add_argument('--results', type=str, help='stores the full data frame at the end.')
    parser.add_argument('--patient_list_mosaiq', type=str, help='a full list of patients from mosaiq. This should include at least IDB, Gruppe')
    parser.add_argument('--output', type=str, help='output file')

    args = parser.parse_args()

    main(args.drive, args.results, args.patient_list_mosaiq, args.output)
