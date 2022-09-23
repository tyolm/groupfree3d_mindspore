"""General Sampling Module"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pointnet2_util

ms.set_context(max_call_depth=10000)
ms.set_context(mode=ms.PYNATIVE_MODE)

class GeneralSamplingModule(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        transpose = ops.Transpose()
        # xyz_flipped = transpose(xyz, (0, 2, 1))
        new_xyz = pointnet2_util.index_points(xyz, sample_inds)
        new_features = pointnet2_util.index_points(transpose(features, (0, 2, 1)), sample_inds)

        return new_xyz, transpose(new_features, (0, 2, 1)), sample_inds
