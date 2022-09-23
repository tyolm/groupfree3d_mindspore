"""Groupfree 3D Backbone"""

import mindspore.nn as nn
import mindspore.ops as ops
import sys
import os
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from blocks import PointNet2SetAbstraction, PointNetFeaturePropagation


class Pointnet2Backbone(nn.Cell):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, width=1, depth=2):
        super().__init__()
        self.depth = depth
        self.width = width

        self.sa1 = PointNet2SetAbstraction(
            npoint=2048,
            radius=0.2,
            nsample=64,
            in_channel=input_feature_dim + 3,
            mlp=[64 * width for i in range(depth)] + [128 * width],
            group_all=False
        )

        self.sa2 = PointNet2SetAbstraction(
            npoint=1024,
            radius=0.4,
            nsample=32,
            in_channel=128 * width + 3,
            mlp=[128 * width for i in range(depth)] + [256 * width],
            group_all=False
        )

        self.sa3 = PointNet2SetAbstraction(
            npoint=512,
            radius=0.8,
            nsample=16,
            in_channel=256 * width + 3,
            mlp=[128 * width for i in range(depth)] + [256 * width],
            group_all=False
        )

        self.sa4 = PointNet2SetAbstraction(
            npoint=256,
            radius=1.2,
            nsample=16,
            in_channel=256 * width + 3,
            mlp=[128 * width for i in range(depth)] + [256 * width],
            group_all=False
        )

        self.fp1 = PointNetFeaturePropagation(in_channel=256 * width + 256 * width, mlp=[256 * width, 256 * width])

        self.fp2 = PointNetFeaturePropagation(in_channel=256 * width + 256 * width, mlp=[256 * width, 288])



    def construct(self, data, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        # end_points = {}
        xyz = data[:, :, :3] # [B, N, 3]
        xyz = xyz.transpose(0, 2, 1)
        features = None

        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz.transpose(0, 2, 1)
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz.transpose(0, 2, 1)
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz.transpose(0, 2, 1)
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz.transpose(0, 2, 1)
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        # features = self.fp1(fp_transpose(end_points['sa3_xyz'], (0, 2, 1)), 
        #                     fp_transpose(end_points['sa4_xyz'], (0, 2, 1)), 
        #                     end_points['sa3_features'],
        #                     end_points['sa4_features'])
        # features = self.fp2(fp_transpose(end_points['sa2_xyz'], (0, 2, 1)), 
        #                     fp_transpose(end_points['sa3_xyz'], (0, 2, 1)), 
        #                     end_points['sa2_features'], 
        #                     features)

        features = self.fp1(end_points['sa3_xyz'], 
                            end_points['sa4_xyz'], 
                            end_points['sa3_features'],
                            end_points['sa4_features'])

        features = self.fp2(end_points['sa2_xyz'], 
                            end_points['sa3_xyz'], 
                            end_points['sa2_features'],
                            features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds

        return end_points