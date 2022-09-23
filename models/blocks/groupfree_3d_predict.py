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
"""Group-Free 3D Predict"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context

context.set_context(max_call_depth=10000)
context.set_context(mode=context.PYNATIVE_MODE)


class PredictHead(nn.Cell):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = nn.Conv1d(seed_feat_dim, seed_feat_dim, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(seed_feat_dim)
        self.conv2 = nn.Conv1d(seed_feat_dim, seed_feat_dim, kernel_size=1, has_bias=True)
        self.bn2 = nn.BatchNorm2d(seed_feat_dim)
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(-1)
        self.transpose = ops.Transpose()

        self.objectness_scores_head = nn.Conv1d(seed_feat_dim, 1, kernel_size=1, has_bias=True)
        self.center_residual_head = nn.Conv1d(seed_feat_dim, 3, kernel_size=1, has_bias=True)
        self.heading_class_head = nn.Conv1d(seed_feat_dim, num_heading_bin, kernel_size=1, has_bias=True)
        self.heading_residual_head = nn.Conv1d(seed_feat_dim, num_heading_bin, kernel_size=1, has_bias=True)
        self.size_class_head = nn.Conv1d(seed_feat_dim, num_size_cluster, kernel_size=1, has_bias=True)
        self.size_residual_head = nn.Conv1d(seed_feat_dim, num_size_cluster * 3, kernel_size=1, has_bias=True)
        self.sem_cls_scores_head = nn.Conv1d(seed_feat_dim, self.num_class, kernel_size=1, has_bias=True)

    def construct(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        relu = nn.ReLU()
        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = relu(self.squeeze(self.bn1(self.expand_dims(self.conv1(features), -1))))
        net = relu(self.squeeze(self.bn2(self.expand_dims(self.conv2(net), -1))))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, 1)
        # print('objectness_scores:', objectness_scores.shape)
        # center
        center_residual = self.center_residual_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(0, 2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        # mean_size_arr = Tensor.from_numpy(self.mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster, 3)
        mean_size_arr = Tensor(self.mean_size_arr, ms.float32)
        mean_size_arr = self.expand_dims(self.expand_dims(mean_size_arr, 0), 0) # (1, 1, num_size_cluster, 3)
        size_scores = self.size_class_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, num_size_cluster)
        size_residuals_normalized = self.size_residual_head(net).transpose(0, 2, 1).view(
            (batch_size, num_proposal, self.num_size_cluster, 3))  # (batch_size, num_proposal, num_size_cluster, 3)
        size_residuals = size_residuals_normalized * mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        size_recover = size_residuals + mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        pred_size_class = ops.Argmax()(size_scores)  # batch_size, num_proposal
        pred_size_class = self.expand_dims(self.expand_dims(pred_size_class, -1), -1)
        # pred_size_class = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_size_class = mnp.tile(pred_size_class, (1, 1, 1, 3))
        pred_size = ops.GatherD()(size_recover, 2, pred_size_class)  # batch_size, num_proposal, 1, 3
        pred_size = pred_size.squeeze(2)  # batch_size, num_proposal, 3

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}size_scores'] = size_scores
        end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
        end_points[f'{prefix}size_residuals'] = size_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return center, pred_size