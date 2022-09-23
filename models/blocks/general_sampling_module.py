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
