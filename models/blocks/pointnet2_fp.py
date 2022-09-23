"""PointNet2FeaturePropagation"""

import os
import sys
from mindspore import nn
import mindspore.numpy as mnp
from mindspore import ops
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pointnet2_util import square_distance, index_points


class PointNetFeaturePropagation(nn.Cell):
    """
    FP module.
    Input:
        in_channel(int): Input characters of points.
        mlp(array):output size for MLP on each point.
        xyz1: input points position data, [B, N, C]
        points1: input points data, [B, D, N]
        xyz2: input points position data, [B, S, C]
        points2: input points data, [B, D, S]
    Return:
        new_points: upsampled points data, [B, D', N], D'=mlp[-1]

    Examples:
        >> fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 64])
        >> l1_xyz = Tensor(np.ones((24, 512, 3)),ms.float32)
        >> l2_xyz = Tensor(np.ones((24, 128, 3)),ms.float32)
        >> l1_points = Tensor(np.ones((24, 128, 512)),ms.float32)
        >> l2_points = Tensor(np.ones((24, 256, 128)),ms.float32)

        >> l1_points = fp2.construct(l1_xyz, l2_xyz, l1_points, l2_points)
        >> print(l1_points.shape)
    """

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        self.transpose = ops.Transpose()
        self.relu = ops.ReLU()

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def construct(self, xyz1, xyz2, points1, points2):
        """FP construct"""
        points2 = self.transpose(points2, (0, 2, 1))
        _, n, _ = xyz1.shape
        _, s, _ = xyz2.shape
        if s == 1:
            interpolated_points = mnp.tile(points2, (1, n, 1))
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = ops.Sort(axis=-1)(dists)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = ops.ReduceSum(keep_dims=True)(dist_recip, 2)
            weight = ops.ExpandDims()((dist_recip / norm), 3)
            interpolated_points = ops.ReduceSum()((index_points(points2, idx) * weight), 2)
        if points1 is not None:
            points1 = self.transpose(points1, (0, 2, 1))
            new_points = ops.Concat(-1)((points1, interpolated_points))
        else:
            new_points = interpolated_points

        new_points = self.transpose(new_points, (0, 2, 1))
        new_points = ops.ExpandDims()(new_points, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        new_points = ops.Squeeze(2)(new_points)
        return new_points
