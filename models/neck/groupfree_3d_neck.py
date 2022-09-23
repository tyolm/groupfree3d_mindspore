"""Groupfree 3D Neck"""

import os
import sys
import mindspore.nn as nn
import mindspore.ops as ops
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from blocks import FPSModule, PointsObjClsModule, GeneralSamplingModule

class GroupFree3DNeck(nn.Cell):
    def __init__(self, sampling, num_proposal=128):
        super(GroupFree3DNeck, self).__init__()
        self.sampling = sampling
        self.num_proposal = num_proposal
        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()

    def construct(self, end_points):
        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            sigmoid = ops.Sigmoid()
            squeeze = ops.Squeeze(1)
            topk = ops.TopK
            points_obj_cls_scores = squeeze(sigmoid(points_obj_cls_logits))
            sample_inds = topk(points_obj_cls_scores, self.num_proposal)[1].asnumpy().astype(int)
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal

        return points_xyz, points_features, cluster_xyz, cluster_feature, end_points