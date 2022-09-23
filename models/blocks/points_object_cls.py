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
"""Points Object Classification"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore import context

ms.set_context(max_call_depth=10000)
ms.set_context(mode=ms.PYNATIVE_MODE)


class PointsObjClsModule(nn.Cell):
    def __init__(self, seed_feature_dim):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_dim)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=1, has_bias=True)
        self.bn2 = nn.BatchNorm2d(self.in_dim)
        self.conv3 = nn.Conv1d(self.in_dim, 1, kernel_size=1, has_bias=True)
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(-1)

    def construct(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed)
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        relu = nn.ReLU()
        net = relu(self.squeeze(self.bn1(self.expand_dims(self.conv1(seed_features), -1))))
        net = relu(self.squeeze(self.bn2(self.expand_dims(self.conv2(net), -1))))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits