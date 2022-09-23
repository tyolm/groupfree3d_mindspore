"""Groupfree 3D Head Module"""

import sys
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from blocks import FPSModule, PointsObjClsModule, GeneralSamplingModule
from blocks import PositionEmbeddingLearned, TransformerDecoderLayer, ClsAgnosticPredictHead, PredictHead


class GroupFree3DHead(nn.Cell):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 size_cls_agnostic=False,
                 mean_size_arr=np.array([[0.76966726, 0.81160211, 0.92573741],
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
        super(GroupFree3DHead, self).__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert mean_size_arr.shape[0] == self.num_size_cluster
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        if self.size_cls_agnostic:
            self.proposal_head = ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288)
        else:
            self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                             mean_size_arr, num_proposal, 288)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1, has_bias=True)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1, has_bias=True)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.CellList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.CellList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.CellList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.CellList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = nn.CellList()
        for i in range(self.num_decoder_layers):
            if self.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288))
            else:
                self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                         mean_size_arr, num_proposal, 288))

        # Init
        self.init_weights()
        # self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm(self)

    def construct(self, end_points):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

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
            topk = ops.TopK()
            points_obj_cls_scores = squeeze(sigmoid(points_obj_cls_logits))
            sample_inds = topk(points_obj_cls_scores, self.num_proposal)[1]
            sample_inds = Tensor(sample_inds, ms.int32)
            print('xyz', xyz.shape)
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            print('cluster_xyz', cluster_xyz.shape)
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            # raise NotImplementedError
            print("NotImplementedError")

        # Proposal
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.copy()
        base_size = proposal_size.copy()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                cat = ops.Concat(-1)
                query_pos = cat((base_xyz, base_size))
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.copy()
            base_size = base_size.copy()

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.get_parameters():
            if m.dim() > 1:
                m = initializer(XavierUniform(), m.shape)

    # def init_bn_momentum(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             m.momentum = self.bn_momentum
