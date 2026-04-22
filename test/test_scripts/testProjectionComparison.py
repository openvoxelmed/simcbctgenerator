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

"""Physics noise comparison using matched reconstruction.

Reconstructs both the real clinical projections and the simulated projections
with the same RTK FDK algorithm, removing the scanner's proprietary corrections
from the comparison. This gives a clean signal on how well the projection
physics match, independent of clinical post-processing.

Usage
-----
uv run python test/test_scripts/testProjectionComparison.py \\
    --patient_dir /mnt/f/graz/data/output_challenge_Varian/stage-1/G000 \\
    --vendor varian

# With motion simulation:
uv run python test/test_scripts/testProjectionComparison.py \\
    --patient_dir /mnt/f/graz/data/output_challenge_Varian/stage-1/G000 \\
    --vendor varian --motion PELVIS
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from simcbctgenerator import ProjectionPipeline
from simcbctgenerator.generate_projections import gaussian as _gaussian, double_sigmoid_profile as _ds_profile
from simcbctgenerator.registration.visualization import save_cbct_comparison
from simcbctgenerator.simulation.standard import StandardCBCTSimulator
from simcbctgenerator.utils.config import MotionConfig


def split_stacked_projections(stacked_path: Path, out_dir: Path, pixel_spacing: tuple):
    """Split a stacked projections.mha into individual numbered .mhd files.

    Handles two storage formats:
      - float32: already log-attenuation domain (simulated or pre-processed) → used as-is
      - uint16:  raw detector counts → converted via -log(counts / 65535)

    Each slice along axis 0 becomes one projection (0000.mhd, 0001.mhd, ...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = sitk.ReadImage(str(stacked_path))
    arr   = sitk.GetArrayFromImage(stack)  # (N, H, W)

    if arr.dtype == np.uint16:
        print("  Raw uint16 counts detected — applying -log(counts / 65535)")
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 1.0, None)          # avoid log(0)
        arr = -np.log(arr / np.iinfo(np.uint16).max)
    else:
        arr = arr.astype(np.float32)

    for i, proj in enumerate(arr):
        img = sitk.GetImageFromArray(proj)
        img.SetSpacing(pixel_spacing)
        sitk.WriteImage(img, str(out_dir / f"{i:04d}.mhd"))

    print(f"  Split {len(arr)} projections → {out_dir}")
    return len(arr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare real vs simulated CBCT via matched RTK reconstruction"
    )
    parser.add_argument("--patient_dir", type=Path, required=True)
    parser.add_argument("--vendor", choices=["elekta", "varian"], default="elekta")
    parser.add_argument(
        "--ct_filename",          default="ct_def.mha",
        help="CT filename (default: ct_def.mha)"
    )
    parser.add_argument(
        "--cbct_filename", default="cbct_clinical.mha",
        help="Real CBCT filename for isocenter (default: cbct_clinical.mha)"
    )
    parser.add_argument(
        "--projections_filename", default="projections.mha",
        help="Stacked real projections filename (default: projections.mha)"
    )
    parser.add_argument(
        "--geometry_filename",    default="geometry.xml",
        help="RTK geometry XML (default: geometry.xml)"
    )
    parser.add_argument(
        "--metadata_filename",    default="metadata.yaml",
        help="Metadata YAML (default: metadata.yaml)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory (default: <patient_dir>/projection_comparison)"
    )
    parser.add_argument(
        "--motion", choices=["PELVIS", "THORAX"], default=None,
        help="Enable random motion simulation"
    )
    parser.add_argument(
        "--correct_cm", action="store_true",
        help="Apply contrast-media correction"
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="Use CPU reconstruction"
    )
    parser.add_argument(
        "--bowtie", action="store_true",
        help="Include beam profile (bowtie) in simulated projections and reconstruction"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    patient_dir       = args.patient_dir
    output_dir        = args.output_dir or (patient_dir / "projection_comparison")
    ct_path           = patient_dir / args.ct_filename
    projections_path  = patient_dir / args.projections_filename
    geometry_xml      = patient_dir / args.geometry_filename
    metadata_yaml     = patient_dir / args.metadata_filename

    for path, name in [
        (ct_path,          "CT"),
        (projections_path, "Real projections"),
        (geometry_xml,     "Geometry"),
        (metadata_yaml,    "Metadata"),
    ]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Patient : {patient_dir.name}")
    print(f"Output  : {output_dir}")
    print(f"Vendor  : {args.vendor}  |  Motion: {args.motion or 'none'}  |  GPU: {not args.no_gpu}")

    api = ProjectionPipeline(
        vendor=args.vendor,
        gpu=not args.no_gpu,
        correct_contrast_media=args.correct_cm,
    )

    # ------------------------------------------------------------------
    # 1. Build system config (geometry + physics from metadata)
    # ------------------------------------------------------------------
    print("\n[1/4] Building system config from geometry + metadata...")
    system_config = StandardCBCTSimulator(
        vendor=args.vendor,
        correct_contrast_media=args.correct_cm,
        gpu=not args.no_gpu,
    ).build_system_config(
        geometry_xml=geometry_xml,
        metadata_yaml=metadata_yaml,
    )
    pixel_spacing = system_config.pixel_size
    print(f"  Pixel spacing : {pixel_spacing} mm")
    print(f"  Angles        : {len(system_config.effective_angles)}")

    # ------------------------------------------------------------------
    # 2. Reconstruct real projections with RTK FDK
    # ------------------------------------------------------------------
    real_recon_path = output_dir / "cbct_real_rtk.mha"
    print("\n[2/4] Reconstructing real projections with RTK FDK...")
    with tempfile.TemporaryDirectory(prefix="real_proj_") as tmp_real:
        n = split_stacked_projections(projections_path, Path(tmp_real), pixel_spacing)
        print(f"  Reconstructing {n} real projections...")
        real_recon = api.reconstruct(
            proj_dir=tmp_real,
            system_config=system_config,
            output_dir=None,
        )

    sitk.WriteImage(real_recon, str(real_recon_path))
    print(f"  Saved → {real_recon_path}")

    # ------------------------------------------------------------------
    # 3. Generate simulated projections + reconstruct
    # ------------------------------------------------------------------
    print("\n[3/4] Generating simulated projections and reconstructing...")
    ct_image = sitk.ReadImage(str(ct_path))

    # Use real CBCT for isocenter so simulated projections align with real ones
    cbct_path = patient_dir / args.cbct_filename
    cbct_image = sitk.ReadImage(str(cbct_path)) if cbct_path.exists() else None
    if cbct_image is not None:
        print(f"  Using real CBCT for isocenter: {cbct_path.name}")
    else:
        print("  No real CBCT found — isocenter from CT center")

    random_motion_type = MotionConfig.MotionType[args.motion] if args.motion else None

    bowtie_label = "with_bowtie" if args.bowtie else "no_bowtie"
    print(f"  Bowtie mode   : {bowtie_label}")

    sim_proj_dir = output_dir / "sim_proj_temp"
    _, _ = api.generate_projections(
        ct_image=ct_image,
        cbct_image=cbct_image,
        geometry_xml=geometry_xml,
        metadata_yaml=metadata_yaml,
        output_dir=sim_proj_dir,
        random_motion_type=random_motion_type,
        include_beam_profile=args.bowtie,
    )

    # Compute normalised beam profile from system config (needed for recon + proj comparison)
    _off  = system_config.geometry.detector_offset
    _pw   = system_config.geometry.detector_pixels_w
    _ps   = system_config.pixel_size[0]
    _x    = np.linspace(-_pw * _ps / 2, _pw * _ps / 2, _pw) + _off
    _phys = system_config.physics
    if _phys.bp_ds_slope1 != 0.0:
        bp_norm = _ds_profile(
            _x, _phys.bp_ds_Afloor, _phys.bp_ds_edge1,
            _phys.bp_ds_slope1, _phys.bp_ds_edge2, _phys.bp_ds_slope2,
        ).astype(np.float32)
    else:
        _bp = np.maximum(_gaussian(_x, system_config.bp_amplitude, 0, system_config.bp_std),
                         system_config.bp_floor)
        bp_norm = (_bp / _bp.max()).astype(np.float32)

    # For reconstruction: if bowtie mode, apply bp_norm in log space to individual files
    drr_temp = sim_proj_dir / "drr_temp"
    if args.bowtie:
        bt_log = -np.log(np.clip(bp_norm, 1e-6, None))  # (W,) ≥ 0, add to log-att
        for mhd_file in sorted(drr_temp.glob("*.mhd")):
            proj    = sitk.GetArrayFromImage(sitk.ReadImage(str(mhd_file))).astype(np.float32)
            out_img = sitk.GetImageFromArray(proj + bt_log)
            out_img.SetSpacing(sitk.ReadImage(str(mhd_file)).GetSpacing())
            sitk.WriteImage(out_img, str(mhd_file))

    sim_recon = api.reconstruct(
        proj_dir=drr_temp,
        system_config=system_config,
        output_dir=None,
    )

    # ------------------------------------------------------------------
    # 3b. Compare projections directly in projection space
    # ------------------------------------------------------------------
    print("\n[3b] Saving projection-space comparison...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_log_attenuation(arr):
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 1.0, None)
        return -np.log(arr / np.iinfo(np.uint16).max)

    real_stack = to_log_attenuation(sitk.GetArrayFromImage(sitk.ReadImage(str(projections_path))))
    sim_raw    = sitk.GetArrayFromImage(sitk.ReadImage(str(sim_proj_dir / "projections_simulated.mha"))).astype(np.float32)
    sim_stack  = to_log_attenuation(sim_raw)

    n_proj = min(real_stack.shape[0], sim_raw.shape[0])

    indices      = [n_proj // 4, n_proj // 2, 3 * n_proj // 4]

    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    fig.suptitle(f"Projection Comparison — {patient_dir.name} ({bowtie_label})", fontsize=14)
    for row, idx in enumerate(indices):
        real_proj = real_stack[idx]
        sim_proj  = sim_stack[idx]
        vmin = min(real_proj.min(), sim_proj.min())
        vmax = max(np.percentile(real_proj, 99), np.percentile(sim_proj, 99))
        axes[row, 0].imshow(real_proj, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Real (frame {idx})")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(sim_proj, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Sim {bowtie_label} (frame {idx})")
        axes[row, 1].axis("off")
        diff = sim_proj - real_proj
        lim  = np.percentile(np.abs(diff), 99)
        im   = axes[row, 2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[row, 2].set_title("Difference (Sim - Real)")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], label="Attenuation diff")
    plt.tight_layout()
    proj_comparison_path = output_dir / f"comparison_projections_{bowtie_label}.png"
    plt.savefig(str(proj_comparison_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {proj_comparison_path}")

    # clean up simulated projection files
    shutil.rmtree(sim_proj_dir, ignore_errors=True)

    sim_recon_path = output_dir / f"cbct_sim_rtk_{bowtie_label}.mha"
    sitk.WriteImage(sim_recon, str(sim_recon_path))
    print(f"  Saved → {sim_recon_path}")

    # ------------------------------------------------------------------
    # 4. Compare
    # ------------------------------------------------------------------
    print("\n[4/4] Saving comparison image...")
    comparison_path = output_dir / f"comparison_rtk_{bowtie_label}.png"
    save_cbct_comparison(sim_recon, real_recon, comparison_path, f"{patient_dir.name} ({bowtie_label})")
    print(f"  Saved → {comparison_path}")

    diff = sitk.GetArrayFromImage(sim_recon).astype(float) - sitk.GetArrayFromImage(real_recon).astype(float)
    print(f"\nSim-RTK size : {sim_recon.GetSize()}")
    print(f"Real-RTK size: {real_recon.GetSize()}")
    print(f"MAE          : {np.abs(diff).mean():.1f} HU")
    print(f"RMSE         : {np.sqrt((diff**2).mean()):.1f} HU")
    print("\nDone.")


if __name__ == "__main__":
    main()
