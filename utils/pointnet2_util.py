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
"""pointnet2 utils"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr


@constexpr
def generate_tensor_fps(b, n):
    """generate tensor"""
    farthest = Tensor(np.random.randint(n, size=(b,)), ms.int32)
    return farthest


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [b, n, c]
        dst: target points, [b, m, c]
    Output:
        dist: per-point square distance, [b, n, m]
    """
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * ops.BatchMatMul()(src, ops.Transpose()(dst, (0, 2, 1)))
    dist += ops.Reshape()(ops.ReduceSum()(src ** 2, -1), (b, n, 1))
    dist += ops.Reshape()(ops.ReduceSum()(dst ** 2, -1), (b, 1, m))
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [b, n, c]
        idx: sample index data, [b, s] or [b, s, nsample]
    Return:
        new_points:, indexed points data, [b, S, c] or [b, s, nsample, c]
    """
    shape = idx.shape
    batch_indices = mnp.arange(shape[0], dtype=ms.int32)
    if len(shape) == 2:
        batch_indices = batch_indices.view(shape[0], 1)
    else:
        batch_indices = batch_indices.view(shape[0], 1, 1)
    batch_indices = batch_indices.expand_as(idx)
    index = ops.Concat(-1)((batch_indices.reshape(idx.shape + (1,)), idx.reshape(idx.shape + (1,))))
    new_points = ops.GatherNd()(points, index)
    return new_points

if __name__ == '__main__':
    x = Tensor(np.random.rand(4, 1024, 3), ms.float32)
    indx = Tensor(np.random.rand(4, 128), ms.int32)
    res = index_points(x, indx)
    print(res.shape)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [b, n, 3] or[b, n, 6]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [b, npoint]
    """
    poinds = xyz[:, :, :3]
    b, n, _ = poinds.shape
    centroids = mnp.zeros((npoint, b), ms.int32)
    distance = mnp.ones((b, n), ms.int32) * 1e9
    farthest = generate_tensor_fps(b, n)
    batch_indices = mnp.arange(b, dtype=ms.int32)
    for i in range(npoint):
        centroids = ops.Cast()(centroids, ms.float32)
        farthest = ops.Cast()(farthest, ms.float32)
        centroids[i] = farthest
        centroids = ops.Cast()(centroids, ms.int32)
        farthest = ops.Cast()(farthest, ms.int32)
        index = ops.Concat(-1)((batch_indices.reshape(batch_indices.shape + (1,)),
                                farthest.reshape(farthest.shape + (1,))))
        centroid = ops.GatherNd()(poinds, index).reshape((b, 1, 3))
        dist = ops.ReduceSum()((poinds - centroid) ** 2, -1)
        distance = ops.Minimum()(distance, dist)
        farthest = ops.Argmax()(distance)
    return ops.Transpose()(centroids, (1, 0))


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [b, n, 3]
        new_xyz: query points, [b, s, 3]
    Return:
        group_idx: grouped points index, [b, s, nsample]
    """
    b, n, _ = xyz.shape
    _, s, _ = new_xyz.shape
    group_idx = mnp.arange(0, n, 1, ms.int32).view(1, 1, n)
    group_idx = ops.Tile()(group_idx, (b, s, 1))
    sqrdists = square_distance(new_xyz, xyz)

    idx = sqrdists > radius ** 2
    group_idx = ops.Select()(idx, -1 * ops.OnesLike()(group_idx), group_idx)
    group_idx = ops.Cast()(group_idx, ms.float32)
    group_idx, _ = ops.TopK()(group_idx, nsample)
    group_idx = ops.Cast()(group_idx, ms.int32)

    group_first = group_idx[:, :, 0].view(b, s, 1)
    group_first = ops.Tile()(group_first, (1, 1, nsample))

    index = group_idx != -1
    group_first = ops.Select()(index, -1 * ops.OnesLike()(group_first), group_first)
    group_idx = ops.Maximum()(group_idx, group_first)

    return group_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [b, n, 3]
        points: input points data, [b, n, d]
    Return:
        new_xyz: sampled points position data, [b, 1, 3]
        new_points: sampled points data, [b, 1, n, 3+d]
    """
    b, n, c = xyz.shape
    new_xyz = ops.Zeros()((b, 1, c), ms.float32)
    grouped_xyz = ops.Reshape()(xyz, (b, 1, n, c))
    if points is not None:
        new_points = ops.Concat(-1)((grouped_xyz, ops.Reshape()(points, (b, 1, n, -1))))
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        xyz: input points position data, [b, n, 3]
        points: input points data, [b, n, d]
    Return:
        new_xyz: sampled points position data, [b, npoint, nsample, 3]
        new_points: sampled points data, [b, npoint, nsample, 3+d]
    """
    b, _, c = xyz.shape
    s = npoint
    fps_idx = farthest_point_sample(xyz, s)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(b, s, 1, c)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = ops.Concat(-1)((grouped_xyz_norm, grouped_points))
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points, fps_idx
