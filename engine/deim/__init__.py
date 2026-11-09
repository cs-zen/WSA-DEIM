# Modified for the WSA-DEIM full-model code release (2026).

"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

from .deim import DEIM
from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder, HaarFPN, HaarPAN, MultiScaleDWEnhancement
from .fsff import GLCAM, FSFF
from .dfine_decoder import DFINETransformer
from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
