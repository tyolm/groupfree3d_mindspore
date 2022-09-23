"""FPS Module"""

import mindspore.nn as nn
import mindspore.ops as ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pointnet2_util

class FPSModule(nn.Cell):
    def __init__(self, num_proposal):
        super().__init__()
        self.num_proposal = num_proposal

    def construct(self, xyz, features):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        # Farthest point sampling (FPS)
        transpose = ops.Transpose()
        sample_inds = pointnet2_util.farthest_point_sample(xyz, self.num_proposal)
        xyz_flipped = xyz.transpose(1, 2)
        new_xyz = pointnet2_util.index_points(transpose(xyz_flipped, (0, 2, 1)), sample_inds)
        new_features = transpose(pointnet2_util.index_points(transpose(features, (0, 2, 1)), sample_inds), (0, 2, 1))

        return new_xyz, new_features, sample_inds
