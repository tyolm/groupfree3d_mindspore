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
"""KPS module"""

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindvision.ms3d.utils.pointnet2_util import index_points


class PointsObjClsModule(nn.Cell):
    """
    Object candidate point prediction from seed point features.

    Args:
        seed_feature_dim(int): number of channels of seed point features.
    """

    def __init__(self, seed_feature_dim):
        super(PointsObjClsModule, self).__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_dim)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=1, has_bias=True)
        self.bn2 = nn.BatchNorm2d(self.in_dim)
        self.conv3 = nn.Conv1d(self.in_dim, 1, kernel_size=1, has_bias=True)
        self.relu = ops.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(-1)

    def construct(self, seed_features):
        """
        Args:
            seed_features: (batch_size, feature_dim, num_seed) tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = self.relu(self.squeeze(
            self.bn1(self.expand_dims(self.conv1(seed_features), -1))))
        net = self.relu(self.squeeze(
            self.bn2(self.expand_dims(self.conv2(net), -1))))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class KPS(nn.Cell):
    """
    Args:
        num_proposal: int (default: 128)
                Number of proposals/detections generated from the network.
                Each proposal is a 3D OBB with a semantic class.
        seed_feature_dim: int (default: 288)
                number of channels of seed point features
    """

    def __init__(self, seed_feature_dim, num_proposal):
        super(KPS, self).__init__()
        self.seed_feature_dim = seed_feature_dim
        self.num_proposal = num_proposal
        self.points_obj_cls = PointsObjClsModule(self.seed_feature_dim)

    def construct(self, xyz, features):
        points_obj_cls_logits = self.points_obj_cls(features)
        points_obj_cls_scores = ops.Sigmoid(points_obj_cls_logits).squueze(1)
        sample_inds = ops.TopK(points_obj_cls_scores, self.num_proposal)[1].int()
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = index_points(
            xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = index_points(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds
