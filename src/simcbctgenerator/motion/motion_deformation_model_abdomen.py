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

"""Motion deformation model for abdomen region.

Blends thorax and pelvis motion models with a smooth spatial transition
along the cranio-caudal (z / axis-0) direction. The transition boundaries
are derived automatically from the extent of the lung mask (thorax region)
and the bowel mask (pelvis region).

The axis-0 orientation is detected automatically by comparing the centroids
of the lung and bowel masks: the lung is always cranial to the bowel, so
whichever has the higher array index is the cranial end.
"""

import numpy as np
import SimpleITK as sitk
import cupy as cp
import logging

from simcbctgenerator.utils.config import MotionConfig
from simcbctgenerator.motion.motion_deformation_model_thorax import MotionDeformationModelThorax
from simcbctgenerator.motion.motion_deformation_model_pelvis import MotionDeformationModelPelvis

logger = logging.getLogger(__name__)


class MotionDeformationModelAbdomen:
    """Combined thorax/pelvis motion model with smooth abdominal blending.

    Instead of building a dedicated motion model for the abdomen, this class
    runs the existing thorax and pelvis models on the same volume and blends
    their displacement fields with a smooth weight ramp along axis 0 (z).

    The transition zone is determined from organ masks by comparing the
    centroids of the lung and bowel masks to determine which end of axis-0
    corresponds to the thorax and which to the pelvis.
    """

    # Expose a minimal params container so callers can toggle debug output
    # the same way they do for the thorax / pelvis models.
    class ParamsMotionDeformation:
        createDebugOutput: bool = False

    def __init__(self, params: "MotionDeformationModelAbdomen.ParamsMotionDeformation",
                 motion_config: MotionConfig):
        self.params = params
        self.motion_config = motion_config

        # Sub-models – created during computeMotionDeformation
        self.thorax_model: MotionDeformationModelThorax | None = None
        self.pelvis_model: MotionDeformationModelPelvis | None = None

        # Blending weight on GPU – shape (D, 1, 1, 1), values in [0, 1]
        # 1.0 = pure thorax, 0.0 = pure pelvis
        self._w_thorax_cp: cp.ndarray | None = None

        self.debugOutput = None

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_mask_extent_along_z(mask_sitk: sitk.Image):
        """Return (z_min, z_max) slice indices that contain non-zero voxels."""
        arr = sitk.GetArrayFromImage(mask_sitk)  # shape (D, H, W)
        nonzero_slices = np.any(arr > 0, axis=(1, 2))
        indices = np.where(nonzero_slices)[0]
        if len(indices) == 0:
            return None, None
        return int(indices[0]), int(indices[-1])

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def computeMotionDeformation(
        self,
        volume_sitk: sitk.Image,
        mask_dict: dict,
        max_amplitude_breathing: float,
        max_amplitude_heart: float,
    ):
        """Compute both sub-model displacement fields and the blending weights.

        Args:
            volume_sitk: CT volume (SimpleITK Image).
            mask_dict: Dictionary with keys
                ``'heart'``, ``'aorta'``, ``'lung'``, ``'spine'``, ``'bowel'``.
                Values are ``sitk.Image`` masks.
            max_amplitude_breathing: Maximum breathing amplitude in mm.
            max_amplitude_heart: Maximum heart amplitude in mm.
        """
        required_keys = {'heart', 'aorta', 'lung', 'spine', 'bowel'}
        missing = required_keys - set(mask_dict.keys())
        if missing:
            raise KeyError(
                f"mask_dict is missing required keys for ABDOMEN model: {missing}"
            )

        # -- 1. Run thorax model ----------------------------------------
        thorax_mask_dict = {
            k: mask_dict[k] for k in ('heart', 'aorta', 'lung', 'spine')
        }

        thorax_params = MotionDeformationModelThorax.ParamsMotionDeformation()
        thorax_params.createDebugOutput = self.params.createDebugOutput
        self.thorax_model = MotionDeformationModelThorax(thorax_params, self.motion_config)
        self.thorax_model.computeMotionDeformation(
            volume_sitk, thorax_mask_dict,
            max_amplitude_breathing, max_amplitude_heart,
        )
        logger.info("Thorax sub-model computed for abdomen blend.")

        # -- 2. Run pelvis model ----------------------------------------
        bowel_mask_sitk = sitk.Cast(mask_dict['bowel'], sitk.sitkUInt8)

        pelvis_params = MotionDeformationModelPelvis.ParamsMotionDeformation()
        pelvis_params.createDebugOutput = self.params.createDebugOutput
        self.pelvis_model = MotionDeformationModelPelvis(pelvis_params, self.motion_config)
        self.pelvis_model.computeMotionDeformation(
            volume_sitk, bowel_mask_sitk, max_amplitude_breathing,
        )
        logger.info("Pelvis sub-model computed for abdomen blend.")

        # -- 3. Derive transition boundaries from masks -----------------
        lung_z_min, lung_z_max = self._find_mask_extent_along_z(mask_dict['lung'])
        bowel_z_min, bowel_z_max = self._find_mask_extent_along_z(mask_dict['bowel'])

        depth = sitk.GetArrayFromImage(volume_sitk).shape[0]

        # -- 4. Build blending weight array on GPU ----------------------
        #
        # w = 1.0 in the thorax (lung) region, 0.0 in the pelvis (bowel)
        # region.  The axis-0 direction is determined by comparing the
        # centroids of the two masks: the lung is always cranial to the
        # bowel anatomically.
        #
        w = cp.zeros(depth, dtype=cp.float32)

        if lung_z_min is None and bowel_z_min is None:
            logger.warning("Both lung and bowel masks are empty – "
                           "defaulting to equal blend (0.5).")
            w[:] = 0.5

        elif lung_z_min is None:
            # No lung → pure pelvis everywhere
            logger.warning("Lung mask is empty – falling back to pure pelvis model.")
            # w stays 0.0

        elif bowel_z_min is None:
            # No bowel → pure thorax everywhere
            logger.warning("Bowel mask is empty – falling back to pure thorax model.")
            w[:] = 1.0

        else:
            lung_center = (lung_z_min + lung_z_max) / 2.0
            bowel_center = (bowel_z_min + bowel_z_max) / 2.0

            if lung_center > bowel_center:
                # Axis-0 goes inferior → superior  (standard HFS DICOM).
                # Lung sits at high indices, bowel at low indices.
                # Transition runs from bowel_z_max up to lung_z_min.
                z_transition_start = bowel_z_max   # end of pure-pelvis zone
                z_transition_end   = lung_z_min    # start of pure-thorax zone
                if z_transition_end <= z_transition_start:
                    z_transition_end = z_transition_start + 1

                # w[:z_transition_start] stays 0.0  (pure pelvis)
                n = z_transition_end - z_transition_start
                w[z_transition_start:z_transition_end] = cp.linspace(
                    0.0, 1.0, n, dtype=cp.float32,
                )
                w[z_transition_end:] = 1.0           # pure thorax
            else:
                # Axis-0 goes superior → inferior.
                # Lung sits at low indices, bowel at high indices.
                # Transition runs from lung_z_max down to bowel_z_min.
                z_transition_start = lung_z_max    # end of pure-thorax zone
                z_transition_end   = bowel_z_min   # start of pure-pelvis zone
                if z_transition_end <= z_transition_start:
                    z_transition_end = z_transition_start + 1

                w[:z_transition_start] = 1.0          # pure thorax
                n = z_transition_end - z_transition_start
                w[z_transition_start:z_transition_end] = cp.linspace(
                    1.0, 0.0, n, dtype=cp.float32,
                )
                # w[z_transition_end:] stays 0.0  (pure pelvis)

            logger.info(
                f"Abdomen blend: lung_center={lung_center:.0f}, "
                f"bowel_center={bowel_center:.0f}, transition="
                f"[{z_transition_start}, {z_transition_end}] (depth={depth})"
            )

        self._w_thorax_cp = w[:, cp.newaxis, cp.newaxis, cp.newaxis]

        # -- 5. Collect debug output ------------------------------------
        if self.params.createDebugOutput:
            class DebugOutput:
                pass

            debug = DebugOutput()
            debug.thorax_debug = self.thorax_model.debugOutput
            debug.pelvis_debug = self.pelvis_model.debugOutput
            debug.blend_weight_thorax = cp.asnumpy(w)
            self.debugOutput = debug
        else:
            self.debugOutput = None

    def getMotionField_cp(self, t: float) -> cp.ndarray:
        """Return the blended displacement field at time *t*.

        Both sub-models are queried at the same time point and their
        outputs are combined via the pre-computed spatial weight:

            ``combined = w * thorax + (1 - w) * pelvis``

        Args:
            t: Simulation time (seconds).

        Returns:
            CuPy array of shape ``(D, H, W, 3)`` – displacement vectors.
        """
        thorax_field = self.thorax_model.getMotionField_cp(t)
        pelvis_field = self.pelvis_model.getMotionField_cp(t)

        combined = (
            self._w_thorax_cp * thorax_field
            + (1.0 - self._w_thorax_cp) * pelvis_field
        )
        return combined