import sys
import mindspore as ms
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, context
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.common.initializer import Zero
import numpy
import os
from .groupfree_3d_losses import SigmoidFocalClassificationLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class GroupFree3DLoss(nn.Cell):
    def __init__(self, num_heading_bin, num_size_cluster, num_class, num_decoder_layers, query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef, query_points_obj_topk=5, center_loss_type='smoothl1', center_delta=1.0,size_loss_type='smoothl1', size_delta=1.0, heading_loss_type='smoothl1', heading_delta=1.0, size_cls_agnostic=False, mean_size_arr=np.array([[0.76966726, 0.81160211, 0.92573741],
    [1.876858,   1.84255952, 1.19315654],
    [0.61327999, 0.61486087, 0.71827014],
    [1.39550063, 1.51215451, 0.83443565],
    [0.97949596, 1.06751485, 0.63296875],
    [0.53166301, 0.59555772, 1.75001483],
    [0.96247056, 0.72462326, 1.14818682],
    [0.83221924, 1.04909355, 1.68756634],
    [0.21132214, 0.4206159,  0.53728459],
    [1.44400728, 1.89708334, 0.26985747],
    [1.02942616, 1.40407966, 0.87554322],
    [1.37664116, 0.65521793, 1.68131292],
    [0.66508189, 0.71111926, 1.29885307],
    [0.41999174, 0.37906947, 1.75139715],
    [0.59359559, 0.59124924, 0.73919014],
    [0.50867595, 0.50656087, 0.30136236],
    [1.15115265, 1.0546296,  0.49706794],
    [0.47535286, 0.49249493, 0.58021168]]
    )):
        super(GroupFree3DLoss, self).__init__()
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_class = num_class
        self.mean_size_arr = mean_size_arr
        self.num_decoder_layers = num_decoder_layers
        self.query_points_generator_loss_coef = query_points_generator_loss_coef
        self.obj_loss_coef = obj_loss_coef
        self.box_loss_coef = box_loss_coef
        self.sem_cls_loss_coef = sem_cls_loss_coef
        self.query_points_obj_topk = query_points_obj_topk
        self.center_loss_type = center_loss_type
        self.center_delta = center_delta
        self.size_loss_type = size_loss_type
        self.size_delta = size_delta
        self.heading_loss_type = heading_loss_type
        self.heading_delta = heading_delta
        self.size_cls_agnostic = size_cls_agnostic
        # context.set_context(mode=context.PYNATIVE_MODE)

    def smoothl1_loss(self, error, delta=1.0):
        """Smooth L1 loss.
        x = error = pred - gt or dist(pred,gt)
        0.5 * |x|^2                 if |x|<=d
        |x| - 0.5 * d               if |x|>d
        """
        abs = ms.ops.Abs()
        diff = abs(error)
        loss = mnp.where(diff < delta, 0.5 * diff *
                        diff / delta, diff - 0.5 * delta)
        return loss

    def l1_loss(self, error):
        abs = ms.ops.Abs()
        loss = abs(error)
        return loss

    def scatter_one_along_dim2(self, x, index):
        print(x)
        print(x.dtype, index.dtype, "----")
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                for k in range(index.shape[2]):
                    print(x)
                    x[i][j][int(index[i][j][k])] = 1.0
                    print("fdassfga")
        return x

    def compute_points_obj_cls_loss_hard_topk(self, end_points, topk):
        box_label_mask = end_points['box_label_mask']
        # seed_inds = Tensor(end_points['seed_inds'], ms.int64)  # B, K
        seed_inds = end_points['seed_inds']  # B, K
        seed_xyz = end_points['seed_xyz']  # B, K, 3
        seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
        B = gt_center.shape[0]
        K = seed_xyz.shape[1]
        K2 = gt_center.shape[1]

        # B, num_points
        point_instance_label = end_points['point_instance_label']
        object_assignment = ops.GatherD()(point_instance_label, 1, seed_inds)  # B, num_seed
        object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
        object_assignment_one_hot = ops.Zeros()((B, K, K2), ms.float64)
        object_assignment_index = Tensor(ops.ExpandDims()(object_assignment, -1), ms.float64)
        print(object_assignment_one_hot)
        print(object_assignment_index.asnumpy())
        object_assignment_one_hot = self.scatter_one_along_dim2(object_assignment_one_hot.asnumpy(), object_assignment_index.asnumpy())  # (B, K, K2)
        delta_xyz = ops.ExpandDims()(seed_xyz, 2) - \
            ops.ExpandDims()(gt_center, 1)  # (B, K, K2, 3)
        delta_xyz = delta_xyz / \
            (ops.ExpandDims()(gt_size, 1) + 1e-6)  # (B, K, K2, 3)
        new_dist = ops.ReduceSum(keep_dims=False)(delta_xyz ** 2, dim=-1)
        euclidean_dist1 = ops.Sqrt()(new_dist + 1e-6)  # BxKxK2
        euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + \
            100 * (1 - object_assignment_one_hot)  # BxKxK2
        euclidean_dist1 = euclidean_dist1.transpose(1, 2)  # BxK2xK
        topk_inds = ops.TopK()(-euclidean_dist1, topk)[1] * box_label_mask[:, :, None] + \
            (box_label_mask[:, :, None] - 1)  # BxK2xtopk
        topk_inds = Tensor(topk_inds, ms.int64)  # BxK2xtopk
        topk_inds = topk_inds.view(B, -1)  # B, K2xtopk
        batch_inds = mnp.tile(ops.ExpandDims()(mnp.arange(B), 1), (1, K2 * topk))
        batch_topk_inds = ops.Stack(-1)([batch_inds, topk_inds]).view(-1, 2)

        objectness_label = ops.Zeros()((B, K + 1), dtype=ms.int64)
        objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
        objectness_label = objectness_label[:, :K]
        objectness_label_mask = ops.GatherD()(
            point_instance_label, 1, seed_inds)  # B, num_seed
        objectness_label[objectness_label_mask < 0] = 0

        total_num_points = B * K
        str_part1 = 'points_hard_topk'
        str_pos_ratio_part = '_pos_ratio'
        str_pos_ratio = str_part1 + str(topk) + str_pos_ratio_part
        end_points[str_pos_ratio] = \
            ops.ReduceSum(keep_dims=False)(
                objectness_label.float()) / float(total_num_points)
        str_neg_ratio_part = '_neg_ratio'
        str_neg_ratio = str_part1 + str(topk) + str_neg_ratio_part
        end_points[str_neg_ratio] = 1 - \
            end_points[str_pos_ratio]

        # Compute objectness loss
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = (objectness_label >= 0).float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        min_value = Tensor(1.0, ms.float32)
        max_value = Tensor(510.0, ms.float32)
        cls_weights /= ops.clip_by_value(cls_normalizer, min_value, max_value)
        cls_loss_src = criterion(seeds_obj_cls_logits.view(
            B, K, 1), ops.ExpandDims()(objectness_label, -1), weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        # Compute recall upper bound
        padding_array = mnp.arange(0, B) * 10000
        padding_array = ops.ExpandDims()(padding_array, 1)  # B,1
        point_instance_label_mask = (point_instance_label < 0)  # B,num_points
        point_instance_label = point_instance_label + padding_array  # B,num_points
        point_instance_label[point_instance_label_mask] = -1
        num_gt_bboxes = ops.Unique()(point_instance_label)[0].shape[0] - 1
        seed_instance_label = ops.GatherD()(
            point_instance_label, 1, seed_inds)  # B,num_seed
        pos_points_instance_label = seed_instance_label * \
            objectness_label + (objectness_label - 1)
        num_query_bboxes = ops.Unique()(
            pos_points_instance_label)[0].shape[0] - 1
        if num_gt_bboxes > 0:
            str_upper_part = '_upper_recall_ratio'
            str_upper = str_part1 + str(topk) + str_upper_part
            end_points[str_upper] = num_query_bboxes / num_gt_bboxes

        return objectness_loss

    def compute_objectness_loss_based_on_query_points(self, end_points, num_decoder_layers):
        """ 
        Compute objectness loss for the proposals.
        """

        if num_decoder_layers > 0:
            prefixes = ['proposal_'] + ['last_'] + \
                [str(i) + 'head_' for i in range(num_decoder_layers - 1)]
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
            print(object_assignment.dtype, "-+_+_+_+_+")
            print(query_points_obj_gt.dtype)
            object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

            end_points[prefix + 'objectness_label'] = query_points_obj_gt
            end_points[prefix + 'objectness_mask'] = objectness_mask
            end_points[prefix + 'object_assignment'] = object_assignment
            total_num_proposal = query_points_obj_gt.shape[0] * \
                query_points_obj_gt.shape[1]
            print(objectness_mask.dtype, "_+_+_+_+_")
            end_points[prefix + 'pos_ratio'] = \
                ops.ReduceSum(keep_dims=False)(
                    query_points_obj_gt) / float(total_num_proposal)
            end_points[prefix + 'neg_ratio'] = \
                ops.ReduceSum(keep_dims=False)(objectness_mask) / \
                float(total_num_proposal) - end_points[prefix + 'pos_ratio']

            # Compute objectness loss
            objectness_scores = end_points[prefix + 'objectness_scores']
            criterion = SigmoidFocalClassificationLoss()
            cls_weights = objectness_mask
            cls_normalizer = cls_weights.sum(axis=1, keepdims=True)
            min_value = Tensor(1.0, ms.float32)
            max_value = Tensor(510.0, ms.float32)
            cls_weights /= ops.clip_by_value(cls_normalizer,
                                             min_value, max_value)
            print(objectness_scores.shape, "++++++++")
            transpose = ops.Transpose()
            cls_loss_src = criterion(transpose(objectness_scores, (0, 2, 1)).view(B, K, 1),
                                     ops.ExpandDims()(query_points_obj_gt, -1),
                                     weights=cls_weights)
            objectness_loss = cls_loss_src.sum() / B

            end_points[prefix + 'objectness_loss'] = objectness_loss
            objectness_loss_sum += objectness_loss

        return objectness_loss_sum, end_points

    def compute_box_and_sem_cls_loss(self, end_points, num_decoder_layers,
                                     center_loss_type='smoothl1', center_delta=1.0,
                                     size_loss_type='smoothl1', size_delta=1.0,
                                     heading_loss_type='smoothl1', heading_delta=1.0,
                                     size_cls_agnostic=False):
        """ Compute 3D bounding box and semantic classification loss.
        """

        num_heading_bin = self.num_heading_bin
        num_size_cluster = self.num_size_cluster
        num_class = self.num_class
        mean_size_arr = self.mean_size_arr

        if num_decoder_layers > 0:
            prefixes = ['proposal_'] + ['last_'] + \
                [str(i) + 'head_' for i in range(num_decoder_layers - 1)]
        else:
            prefixes = ['proposal_']  # only proposal
        box_loss_sum = 0.0
        sem_cls_loss_sum = 0.0
        for prefix in prefixes:
            object_assignment = end_points[prefix + 'object_assignment']
            batch_size = object_assignment.shape[0]
            # Compute center loss
            pred_center = end_points[prefix + 'center']
            gt_center = end_points['center_label'][:, :, 0:3]

            if center_loss_type == 'smoothl1':
                objectness_label = end_points[prefix + 'objectness_label'].float(
                )
                object_assignment_expand = mnp.tile(
                    ops.ExpandDims()(object_assignment, 2), (1, 1, 3))
                assigned_gt_center = ops.GatherD()(
                    gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
                center_loss = self.smoothl1_loss(
                    assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
                center_loss = ops.ReduceSum(keep_dims=False)(center_loss * ops.ExpandDims()(
                    objectness_label, 2)) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            elif center_loss_type == 'l1':
                objectness_label = end_points[prefix + 'objectness_label'].float(
                )
                object_assignment_expand = mnp.tile(
                    ops.ExpandDims()(object_assignment, 2), (1, 1, 3))
                assigned_gt_center = ops.GatherD()(
                    gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
                center_loss = self.l1_loss(
                    assigned_gt_center - pred_center)  # (B,K)
                center_loss = ops.ReduceSum(keep_dims=False)(center_loss * ops.ExpandDims()(
                    objectness_label, 2)) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            else:
                print('NotImplementedError')

            # Compute heading loss
            heading_class_label = ops.GatherD()(end_points['heading_class_label'], 1,
                                                object_assignment)  # select (B,K) from (B,K2)
            criterion_heading_class = nn.SoftmaxCrossEntropyWithLogits()
            heading_class_loss = criterion_heading_class(end_points[prefix + 'heading_scores'].transpose(2, 1),
                                                         heading_class_label)  # (B,K)
            heading_class_loss = ops.ReduceSum(keep_dims=False)(
                heading_class_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

            heading_residual_label = ops.GatherD()(end_points['heading_residual_label'], 1,
                                                   object_assignment)  # select (B,K) from (B,K2)
            heading_residual_normalized_label = heading_residual_label / \
                (mnp.pi / num_heading_bin)

            heading_label_one_hot = Tensor(shape=(batch_size, heading_class_label.shape[1],
                                                  num_heading_bin), dtype=ms.float32, init=Zero())
            heading_label_one_hot = self.scatter_one_along_dim2(heading_label_one_hot, ops.ExpandDims()(heading_class_label, -1))  # src==1 so it's *one-hot* (B,K,num_heading_bin)
            heading_residual_normalized_error = ops.ReduceSum(keep_dims=False)(
                end_points[prefix + 'heading_residuals_normalized'] *
                heading_label_one_hot,
                -1) - heading_residual_normalized_label

            if heading_loss_type == 'smoothl1':
                heading_residual_normalized_loss = heading_delta * self.smoothl1_loss(heading_residual_normalized_error, delta=heading_delta)  # (B,K)
                heading_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                    heading_residual_normalized_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            elif heading_loss_type == 'l1':
                heading_residual_normalized_loss = self.l1_loss(
                    heading_residual_normalized_error)  # (B,K)
                heading_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                    heading_residual_normalized_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
            else:
                sys.exit(0)

            # Compute size loss
            if size_cls_agnostic:
                pred_size = end_points[prefix + 'pred_size']
                size_label = ops.GatherD()(
                    end_points['size_gts'], 1,
                    mnp.tile(ops.ExpandDims()(object_assignment, -1), (1, 1, 3)))  # select (B,K,3) from (B,K2,3)
                size_error = pred_size - size_label
                if size_loss_type == 'smoothl1':
                    size_loss = size_delta * self.smoothl1_loss(size_error,
                                                                delta=size_delta)  # (B,K,3) -> (B,K)
                    size_loss = ops.ReduceSum(keep_dims=False)(size_loss * ops.ExpandDims()(objectness_label, 2)) / (
                        ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
                elif size_loss_type == 'l1':
                    size_loss = self.l1_loss(size_error)  # (B,K,3) -> (B,K)
                    size_loss = ops.ReduceSum(keep_dims=False)(size_loss * ops.ExpandDims()(objectness_label, 2)) / (
                        ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
                else:
                    sys.exit(0)
            else:
                size_class_label = ops.GatherD()(end_points['size_class_label'], 1,
                                                 object_assignment)  # select (B,K) from (B,K2)
                criterion_size_class = nn.SoftmaxCrossEntropyWithLogits()
                size_class_loss = criterion_size_class(end_points[prefix + 'size_scores'].transpose(2, 1),
                                                       size_class_label)  # (B,K)
                size_class_loss = ops.ReduceSum(keep_dims=False)(
                    size_class_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

                size_residual_label = ops.GatherD()(
                    end_points['size_residual_label'], 1,
                    mnp.tile(ops.ExpandDims()(object_assignment, -1), (1, 1, 3)))  # select (B,K,3) from (B,K2,3)

                size_label_one_hot = Tensor(shape=(
                    batch_size, size_class_label.shape[1], num_size_cluster), dtype=ms.float32, init=Zero())
                size_label_one_hot = self.scatter_one_along_dim2(size_label_one_hot, ops.ExpandDims()(size_class_label, -1)) # src==1 so it's *one-hot* (B,K,num_size_cluster)
                size_label_one_hot_tiled = mnp.tile(ops.ExpandDims()(
                    size_label_one_hot, -1), (1, 1, 1, 3))  # (B,K,num_size_cluster,3)
                predicted_size_residual_normalized = ops.ReduceSum(keep_dims=False)(
                    end_points[prefix + 'size_residuals_normalized'] *
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
                    size_residual_normalized_loss = size_delta * self.smoothl1_loss(size_residual_normalized_error,
                                                                                    delta=size_delta)  # (B,K,3) -> (B,K)
                    size_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                        size_residual_normalized_loss * ops.ExpandDims()(objectness_label, 2)) / (
                        ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
                elif size_loss_type == 'l1':
                    size_residual_normalized_loss = self.l1_loss(
                        size_residual_normalized_error)  # (B,K,3) -> (B,K)
                    size_residual_normalized_loss = ops.ReduceSum(keep_dims=False)(
                        size_residual_normalized_loss * ops.ExpandDims()(objectness_label, 2)) / (
                        ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)
                else:
                    sys.exit(0)

            # 3.4 Semantic cls loss
            sem_cls_label = ops.GatherD()(
                end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
            criterion_sem_cls = nn.SoftmaxCrossEntropyWithLogits()
            sem_cls_loss = criterion_sem_cls(
                end_points[prefix + 'sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
            sem_cls_loss = ops.ReduceSum(keep_dims=False)(
                sem_cls_loss * objectness_label) / (ops.ReduceSum(keep_dims=False)(objectness_label) + 1e-6)

            end_points[prefix + 'center_loss'] = center_loss
            end_points[prefix + 'heading_cls_loss'] = heading_class_loss
            end_points[prefix + 'heading_reg_loss'] = heading_residual_normalized_loss
            if size_cls_agnostic:
                end_points[prefix + 'size_reg_loss'] = size_loss
                box_loss = center_loss + 0.1 * heading_class_loss + \
                    heading_residual_normalized_loss + size_loss
            else:
                end_points[prefix + 'size_cls_loss'] = size_class_loss
                end_points[prefix + 'size_reg_loss'] = size_residual_normalized_loss
                box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + \
                    0.1 * size_class_loss + size_residual_normalized_loss
            end_points[prefix + 'box_loss'] = box_loss
            end_points[prefix + 'sem_cls_loss'] = sem_cls_loss

            box_loss_sum += box_loss
            sem_cls_loss_sum += sem_cls_loss
        return box_loss_sum, sem_cls_loss_sum, end_points

    def construct(self, logits, labels):
        """ Loss functions
        """
        # unpack labels
        print("label", labels.shape)
        squeeze = ops.Squeeze()
        unsqueeze = ops.ExpandDims()
        if labels.shape[0] != 1:
            logits['center_label'] = labels[:, :64, :3]
            logits['heading_class_label'] = squeeze(labels[:, :64, 3: 4])
            logits['heading_residual_label'] = squeeze(labels[:, :64, 4: 5])
            logits['size_class_label'] = squeeze(labels[:, :64, 5: 6])
            logits['size_residual_label'] = labels[:, :64, 6: 9]
            logits['size_gts'] = labels[:, :64, 9: 12]
            logits['sem_cls_label'] = squeeze(labels[:, :64, 12: 13])
            logits['box_label_mask'] = squeeze(labels[:, :64, 13: 14])
            logits['point_obj_mask'] = squeeze(labels[:, :, 14: 15])
            logits['point_instance_label'] = squeeze(labels[:, :, 15: 16])
        else:
            logits['center_label'] = labels[:, :64, :3]
            logits['heading_class_label'] = unsqueeze(squeeze(labels[:, :64, 3: 4]), 0)
            logits['heading_residual_label'] = unsqueeze(squeeze(labels[:, :64, 4: 5]), 0)
            logits['size_class_label'] = unsqueeze(squeeze(labels[:, :64, 5: 6]), 0)
            logits['size_residual_label'] = labels[:, :64, 6: 9]
            logits['size_gts'] = labels[:, :64, 9: 12]
            logits['sem_cls_label'] = unsqueeze(squeeze(labels[:, :64, 12: 13]), 0)
            logits['box_label_mask'] = unsqueeze(squeeze(labels[:, :64, 13: 14]), 0)
            logits['point_obj_mask'] = unsqueeze(squeeze(labels[:, :, 14: 15]), 0)
            logits['point_instance_label'] = unsqueeze(squeeze(labels[:, :, 15: 16]), 0)
        # print("------")
        # print(logits['center_label'].shape)
        # print(logits['heading_class_label'].shape)
        # print(logits['heading_residual_label'].shape)
        # print(logits['size_class_label'].shape)
        # print(logits['size_residual_label'].shape)
        # print(logits['size_gts'].shape)
        # print(logits['sem_cls_label'].shape)
        # print(logits['box_label_mask'].shape)
        # print(logits['point_obj_mask'].shape)
        # print(logits['point_instance_label'].shape)
        # print("------")

        # if 'seeds_obj_cls_logits' in logits.keys():
        #     query_points_generation_loss = self.compute_points_obj_cls_loss_hard_topk(
        #         logits, self.query_points_obj_topk)

        #     labels['query_points_generation_loss'] = query_points_generation_loss
        # else:
        #     query_points_generation_loss = 0.0
        query_points_generation_loss = 0.0

        # Obj loss
        objectness_loss_sum, logits = \
            self.compute_objectness_loss_based_on_query_points(
                logits, self.num_decoder_layers)

        labels['sum_heads_objectness_loss'] = objectness_loss_sum

        # Box loss and sem cls loss
        box_loss_sum, sem_cls_loss_sum, logits = self.compute_box_and_sem_cls_loss(
            logits, self.num_decoder_layers,
            self.center_loss_type, center_delta=self.center_delta,
            size_loss_type=self.size_loss_type, size_delta=self.size_delta,
            heading_loss_type=self.heading_loss_type, heading_delta=self.heading_delta,
            size_cls_agnostic=self.size_cls_agnostic)
        labels['sum_heads_box_loss'] = box_loss_sum
        labels['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

        # means average proposal with prediction loss
        loss = self.query_points_generator_loss_coef * query_points_generation_loss + \
            1.0 / (self.num_decoder_layers + 1) * (
                self.obj_loss_coef * objectness_loss_sum + self.box_loss_coef * box_loss_sum + self.sem_cls_loss_coef * sem_cls_loss_sum)
        loss *= 10

        labels['loss'] = loss
        # return loss, end_points
        return loss
