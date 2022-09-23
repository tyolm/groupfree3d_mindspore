"""PointNet2SetAbstraction"""

import os
import sys
from mindspore import nn
from mindspore import ops
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pointnet2_util import sample_and_group_all, sample_and_group


class PointNet2SetAbstraction(nn.Cell):
    """
    SA_ssg  module.
    Input:
        npoint(int):points sampled in farthest point sampling.
        radius(float):search radius in local region,0~1.
        nsample(int): how many points in each local region.
        in_channel(int): Input characters of points.
        mlp(array):output size for MLP on each point.
        group_all(bool): if True choose  pointnet2_group.SampleGroup.sample_and_group_all
                    if False  choose  pointnet2_group.SampleGroup.sample_and_group

        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, C, S]
        new_points_concat: sample points feature data, [B, D', S]

    Examples:
        >> l1_xyz= Tensor(np.ones((24, 3, 512)),ms.float32)
        >> l1_points= Tensor(np.ones((24,128, 512)),ms.float32)
        >> sa2 = PointNet2SetAbstraction(npoint=128, radius=0.4, nsample=64,
                                        in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        >> l2_xyz, l2_points = sa2.construct(l1_xyz, l1_points)
        >> print(l2_xyz.shape, l2_points.shape)

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNet2SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.relu = ops.ReLU()
        self.transpose = ops.Transpose()
        self.reduce_max = ops.ReduceMax()

    def construct(self, xyz, points):
        """SA construct"""
        xyz = self.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = self.transpose(points, (0, 2, 1))
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, inds = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = self.transpose(new_points, (0, 3, 2, 1))

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))

        new_points = self.reduce_max(new_points, 2)
        new_xyz = self.transpose(new_xyz, (0, 2, 1))
        return new_xyz, new_points, inds
