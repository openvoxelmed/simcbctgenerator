"""Download the dummy patient data and write the patient list file."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "test" / "test_scripts"))

from testConfigPatient import extract_test_data

extract_test_data()

list_dir = _project_root / "list"
list_dir.mkdir(exist_ok=True)
(list_dir / "dummy_patient.txt").write_text("test/test_data\n")
print(f"Wrote {list_dir / 'dummy_patient.txt'}")
