# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
