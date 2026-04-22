# Patient

Image-centric patient model used by the simulation backends and pipelines.

## Responsibilities

- `Patient` stores CT/CBCT/masks plus image-space helpers such as resampling and isocenter/origin bookkeeping.
- Dataset-specific loading is handled by loader adapters in `simcbctgenerator.patient_setup`. Each loader knows how to populate a `Patient` from its input layout:
    - `XVIPatientLoader` — DICOM `CT_SET`/`DICOM_PLAN` folders, reads CBCT from `IMAGES/*/Reconstruction`, owns the `.his` projection reader (`load_projections`) and `save_real_cbct`.
    - `SynthRadPatientLoader` — NIfTI `ct.nii.gz`/`masks.nii.gz` and optional `cbct.nii.gz`.
    - `DummyPatientLoader` — same layout as SynthRAD but resolves paths against a fixed `ct_dir`.

## `from_folder` vs `from_images`

`Patient` has two constructors:

- `Patient(config, path)` / `Patient.from_folder(config, path)` — routes through `get_patient_loader(config)` and reads the patient from disk using the modality-specific loader.
- `Patient.from_images(ct_image, config, mask_image=..., reference_cbct=...)` — builds a `Patient` directly from in-memory `sitk.Image` objects, skipping the filesystem entirely. Use this when you already have images loaded (for example, challenge submissions or unit tests) and only need the simulation components.

When `from_images` is used, TotalSegmentator can generate masks on the fly by setting `use_totalsegmentator=True` on the `PatientConfig`.

::: simcbctgenerator.patient
    options:
      show_root_heading: true
      show_source: false
