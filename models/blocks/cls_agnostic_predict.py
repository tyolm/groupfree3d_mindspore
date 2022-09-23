"""Class Agnostic Predict"""

import numpy as np
import mindspore.nn as nn
from mindspore import context

context.set_context(max_call_depth=10000)
context.set_context(mode=context.PYNATIVE_MODE)

class ClsAgnosticPredictHead(nn.Cell):
    def __init__(self, num_class, num_heading_bin, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = nn.Conv1d(seed_feat_dim, seed_feat_dim, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = nn.Conv1d(seed_feat_dim, seed_feat_dim, kernel_size=1, has_bias=True)
        self.bn2 = nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = nn.Conv1d(seed_feat_dim, 1, kernel_size=1, has_bias=True)
        self.center_residual_head = nn.Conv1d(seed_feat_dim, 3, kernel_size=1, has_bias=True)
        self.heading_class_head = nn.Conv1d(seed_feat_dim, num_heading_bin, kernel_size=1, has_bias=True)
        self.heading_residual_head = nn.Conv1d(seed_feat_dim, num_heading_bin, kernel_size=1, has_bias=True)
        self.size_pred_head = nn.Conv1d(seed_feat_dim, 3, kernel_size=1, has_bias=True)
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
        net = relu(self.bn1(self.conv1(features)))
        net = relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(0, 2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        pred_size = self.size_pred_head(net).transpose(0, 2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(0, 2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return center, pred_size
