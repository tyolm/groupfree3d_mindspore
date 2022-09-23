"""Groupfree 3D Losses"""

import mindspore as ms
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore.common.initializer import Zero
import numpy


def smoothl1_loss(error, delta=1.0):
    """Smooth L1 loss.
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    |x| - 0.5 * d               if |x|>d
    """
    abs = ms.ops.Abs()
    diff = abs(error)
    loss = np.where(diff < delta, 0.5 * diff *
                    diff / delta, diff - 0.5 * delta)
    return loss


def l1_loss(error):
    abs = ms.ops.Abs()
    loss = abs(error)
    return loss


def scatter_one_along_dim2(x, index):
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                x[i][j][int(index[i][j][k])] = 1.0
    return x


class SigmoidFocalClassificationLoss(nn.Cell):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """

        # log1p = ms.ops.Log1p()
        min_value = Tensor(0, mindspore.float32)
        max_value = Tensor(510, mindspore.float32)
        loss = ms.ops.clip_by_value(input, min_value, max_value) - input * target + \
            np.log1p((ms.ops.Exp()(-ms.ops.Abs()(input))))
        return loss

    def construct(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = ms.ops.Sigmoid()(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        # print(pred_sigmoid.shape, target.shape)
        # pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        pt = ops.Mul()(target, (1.0 - pred_sigmoid)) + ops.Mul()((1.0 - target), pred_sigmoid)
        focal_weight = alpha_weight * ms.ops.Pow()(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = ms.ops.ExpandDims()(weights, -1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    transpose = ops.Transpose()
    box_label_mask = end_points['box_label_mask']
    seed_inds = Tensor(end_points['seed_inds'], dtype=ms.int64)  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = ops.GatherD()(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - \
        1  # set background points to the last gt bbox
    object_assignment_one_hot = ops.Zeros()((B, K, K2), ms.int64)
    scatter_one_along_dim2(object_assignment_one_hot, ops.ExpandDims()(object_assignment, -1))
    # object_assignment_one_hot.scatter_(
    #     2, ops.ExpandDims()(object_assignment, -1), 1)  # (B, K, K2)
    delta_xyz = ops.ExpandDims()(seed_xyz, 2) - \
        ops.ExpandDims()(gt_center, 1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / \
        (ops.ExpandDims()(gt_size, 1) + 1e-6)  # (B, K, K2, 3)
    new_dist = ops.ReduceSum(keep_dims=False)(delta_xyz ** 2, axis=-1)
    euclidean_dist1 = ops.Sqrt()(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + \
        100 * (1 - object_assignment_one_hot)  # BxKxK2
    # euclidean_dist1 = euclidean_dist1.transpose(1, 2)  # BxK2xK
    euclidean_dist1 = transpose(euclidean_dist1, (0, 2, 1))
    topk_inds = ops.TopK()(-euclidean_dist1, topk)[1] * box_label_mask[:, :, None] + \
        (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = Tensor(topk_inds, dtype=ms.int64)  # BxK2xtopk
    topk_inds = topk_inds.view((B, -1))  # B, K2xtopk
    batch_inds = np.tile(ops.ExpandDims()(np.arange(B), 1), (1, K2 * topk))
    batch_inds = Tensor(batch_inds, dtype=ms.int64)
    batch_topk_inds = ops.Stack(-1)([batch_inds, topk_inds]).view(-1, 2)

    objectness_label = ops.Zeros()((B, K + 1), ms.int64)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = ops.GatherD()(
        point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        ops.ReduceSum(keep_dims=False)(
            Tensor(objectness_label, ms.float32)) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - \
        end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = Tensor((objectness_label >= 0), ms.float32)
    cls_normalizer = Tensor(cls_weights.sum(axis=1, keepdims=True), ms.float32)
    min_value = Tensor(1.0, ms.float32)
    max_value = Tensor(510.0, ms.float32)
    cls_weights /= ops.clip_by_value(cls_normalizer, min_value, max_value)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(
        B, K, 1), ops.ExpandDims()(objectness_label, -1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    padding_array = np.arange(0, B) * 10000
    padding_array = ops.ExpandDims()(padding_array, 1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = ops.Unique()(point_instance_label.squeeze())[0].shape[0] - 1
    seed_instance_label = ops.GatherD()(
        point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * \
        objectness_label + (objectness_label - 1)
    num_query_bboxes = ops.Unique()(pos_points_instance_label.squeeze())[0].shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss


def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + \
            [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Associate proposal and GT objects
        # B,num_seed in [0,num_points-1]
        seed_inds = Tensor(end_points['seed_inds'], ms.int64)
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        query_points_sample_inds = Tensor(end_points['query_points_sample_inds'], ms.int64)

        B = seed_inds.shape[0]
        K = query_points_sample_inds.shape[1]
        K2 = gt_center.shape[1]

        seed_obj_gt = ops.GatherD()(
            end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
        query_points_obj_gt = ops.GatherD()(
            seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

        # B, num_points
        point_instance_label = end_points['point_instance_label']
        seed_instance_label = ops.GatherD()(
            point_instance_label, 1, seed_inds)  # B,num_seed
        query_points_instance_label = ops.GatherD()(
            seed_instance_label, 1, query_points_sample_inds)  # B,query_points

        objectness_mask = ops.Ones()((B, K), ms.float32)

        # Set assignment
        # (B,K) with values in 0,1,...,K2-1
        object_assignment = query_points_instance_label
        object_assignment[object_assignment < 0] = K2 - \
            1  # set background points to the last gt bbox

        end_points[f'{prefix}objectness_label'] = query_points_obj_gt
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment
        total_num_proposal = query_points_obj_gt.shape[0] * \
            query_points_obj_gt.shape[1]
        end_points[f'{prefix}pos_ratio'] = \
            ops.ReduceSum(keep_dims=False)(
                ops.Cast()(query_points_obj_gt, ms.float32)) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = \
            ops.ReduceSum(keep_dims=False)(ops.Cast()(objectness_mask, ms.float32)) / \
            float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = Tensor(objectness_mask, ms.float32)
        cls_normalizer = Tensor(cls_weights.sum(axis=1, keepdims=True), ms.float32)
        min_value = Tensor(1.0, ms.float32)
        max_value = Tensor(510.0, ms.float32)
        cls_weights /= ops.clip_by_value(cls_normalizer, min_value, max_value)
        cls_loss_src = criterion(ops.Transpose()(objectness_scores, (0, 2, 1)).view(B, K, 1),
                                 ops.ExpandDims()(query_points_obj_gt, -1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + \
            [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = ops.Cast()(end_points[f'{prefix}objectness_label'], ms.float32)
            object_assignment_expand = np.tile(
                ops.ExpandDims()(object_assignment, 2), (1, 1, 3))
            assigned_gt_center = ops.GatherD()(
                gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(
                assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = ops.ReduceSum(keep_dims=False)(center_loss * ops.ExpandDims()(
                objectness_label, 2)) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = Tensor(end_points[f'{prefix}objectness_label'], ms.float32)
            object_assignment_expand = np.tile(
                ops.ExpandDims()(object_assignment, 2), (1, 1, 3))
            assigned_gt_center = ops.GatherD()(
                gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = ops.ReduceSum(keep_dims=False)(center_loss * ops.ExpandDims()(
                objectness_label, 2)) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = ops.GatherD()(end_points['heading_class_label'], 1,
                                            object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss()
        heading_class_label = ops.Cast()(heading_class_label, ms.int32)
        heading_class_loss = criterion_heading_class(ops.Transpose()(end_points[f'{prefix}heading_scores'], (0, 2, 1)), heading_class_label)  # (B,K)
        heading_class_loss = ops.ReduceSum(keep_dims=False)(
            heading_class_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

        heading_residual_label = ops.GatherD()(end_points['heading_residual_label'], 1,
                                               object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / \
            (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = Tensor(shape=(batch_size, heading_class_label.shape[1],
                                              num_heading_bin), dtype=ms.float32, init=Zero())
        # heading_label_one_hot.scatter_(2, ops.ExpandDims()(heading_class_label, -1), 1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        scatter_one_along_dim2(heading_label_one_hot, ops.ExpandDims()(heading_class_label, -1))
        heading_residual_normalized_error = ops.ReduceSum(keep_dims=False)(
            end_points[f'{prefix}heading_residuals_normalized'] *
            heading_label_one_hot,
            -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                heading_residual_normalized_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(
                heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                heading_residual_normalized_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = end_points[f'{prefix}pred_size']
            size_label = ops.GatherD()(
                end_points['size_gts'], 1,
                np.tile(ops.ExpandDims()(object_assignment, -1), (1, 1, 3)))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error,
                                                       delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = ops.ReduceSum(keep_dims=False)(size_loss * ops.ExpandDims()(objectness_label, 2)) / (
                    ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = ops.ReduceSum(keep_dims=False)(size_loss * ops.ExpandDims()(objectness_label, 2)) / (
                    ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = ops.GatherD()(end_points['size_class_label'], 1,
                                             object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss()
            size_class_label = ops.Cast()(size_class_label, ms.int32)
            size_class_loss = criterion_size_class(ops.Transpose()(end_points[f'{prefix}size_scores'], (0, 2, 1)),
                                                   size_class_label)  # (B,K)
            size_class_loss = ops.ReduceSum(keep_dims=False)(
                size_class_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

            size_residual_label = ops.GatherD()(
                end_points['size_residual_label'], 1,
                np.tile(ops.ExpandDims()(object_assignment, -1), (1, 1, 3)))  # select (B,K,3) from (B,K2,3)

            size_label_one_hot = Tensor(shape=(
                batch_size, size_class_label.shape[1], num_size_cluster), dtype=ms.float32, init=Zero())
            # size_label_one_hot.scatter_(2, ops.ExpandDims()(size_class_label, -1),
            #                             1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            scatter_one_along_dim2(size_label_one_hot, ops.ExpandDims()(size_class_label, -1))
            size_label_one_hot_tiled = np.tile(ops.ExpandDims()(
                size_label_one_hot, -1), (1, 1, 1, 3))  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = ops.ReduceSum(keep_dims=False)(
                end_points[f'{prefix}size_residuals_normalized'] *
                size_label_one_hot_tiled,
                2)  # (B,K,3)

            mean_size_arr_expanded = ops.ExpandDims()(ops.ExpandDims()(Tensor.from_numpy(mean_size_arr.astype(numpy.float32)), 0),
                                                      0)  # (1,1,num_size_cluster,3)
            mean_size_label = ops.ReduceSum(keep_dims=False)(
                size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / \
                mean_size_label  # (B,K,3)

            size_residual_normalized_error = predicted_size_residual_normalized - \
                size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                           delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                    size_residual_normalized_loss * ops.ExpandDims()(objectness_label, 2)) / (
                    ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(
                    size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                    size_residual_normalized_loss * ops.ExpandDims()(objectness_label, 2)) / (
                    ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = ops.GatherD()(
            end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss()
        sem_cls_label = ops.Cast()(sem_cls_label, ms.int32)
        sem_cls_loss = criterion_sem_cls(
            ops.Transpose()(end_points[f'{prefix}sem_cls_scores'], (0, 2, 1)), sem_cls_label)  # (B,K)
        sem_cls_loss = ops.ReduceSum(keep_dims=False)(
            sem_cls_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            end_points[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + \
                heading_residual_normalized_loss + size_loss
        else:
            end_points[f'{prefix}size_cls_loss'] = size_class_loss
            end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + \
                0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points


def get_loss(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0,
             size_cls_agnostic=False):
    """ Loss functions
    """
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk)

        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points(
            end_points, num_decoder_layers)

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta,
        size_cls_agnostic=size_cls_agnostic)
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
        1.0 / (num_decoder_layers + 1) * (
            obj_loss_coef * objectness_loss_sum + box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    return loss, end_points
