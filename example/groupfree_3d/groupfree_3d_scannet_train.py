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
""" Group-free 3D ScanNet train script. """

import os
import sys
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train import Model
from numpy import save
import mindspore.dataset as ds


base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(base_dir))
sys.path.append(base_dir)
sys.path.append(os.path.join(root_dir, 'dataset'))
sys.path.append(os.path.join(root_dir, 'models'))
sys.path.append(os.path.join(root_dir, 'engine'))
sys.path.append(os.path.join(root_dir, 'utils'))

from mindvision.engine.callback import LossMonitor
from unpack_dataset import unpack_scannet, unpack_sunrgbd
from groupfree_3d_detection import groupfree_3d_model
from ops.groupfree_3d_loss_helper import GroupFree3DLoss
from ops.test_loss import TestLoss
from backbones.groupfree_3d_backbone import Pointnet2Backbone
from head import groupfree_3d_head
from dataset.scannet_v2 import ScannetDetectionDataset
from models.groupfree_3d_detection import Groupfree3DModel

def parse_option():
    parser = argparse.ArgumentParser(description='Groupfree_3D train.')
    # model

    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # transformer
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='number of decoder layers')

    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')

    # data
    parser.add_argument('--num_point', type=int, default=20000, help='Number of points.')
    parser.add_argument('--data_url', type=str, default='/home/tyolm/repos/Group-Free-3D', help='Location of data.')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size per GPU during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=50, help='val frequency')

    # others
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    parser.add_argument('--device_target', type=str, default="CPU", choices=["Ascend", "GPU", "CPU"])

    parser.add_argument('--num_class', type=int, default=18, help='Number of classes.')
    parser.add_argument('--num_heading_bin', type=int, default=1, help='Number of heading bin.')
    parser.add_argument('--num_size_cluster', type=int, default=18, help='Number of cluster size.')
    parser.add_argument('--min_lr', type=float, default=0.00001, help="The min learning rate.")
    parser.add_argument('--max_lr', type=float, default=0.001, help="The max learning rate.")
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004, help='Initial learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay [default: 0.0005]')


    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./groupfree_3d", help='Location of training outputs.')
    parser.add_argument('--epoch_size', type=int, default=250, help='Train epoch size.')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Initial learning rate for all except decoder.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')

    args = parser.parse_known_args()[0]

def get_dataset(args):
    if args.dataset == 'scannet':
        from utils.model_util_scannet import ScannetDatasetConfig
        dataset_config = ScannetDatasetConfig()
        train_dataset = ScannetDetectionDataset(path=args.data_url,
                      split='train',
                      num_points=args.num_point)
        test_dataset = ScannetDetectionDataset(path=args.data_url,
                      split='val',
                      num_points=args.num_point)
        train_dataset_generator = ds.GeneratorDataset(train_dataset, ["point_clouds", "center_label", "heading_class_label", "heading_residual_label", "size_class_label", "size_residual_label", "size_gts", "sem_cls_label", "box_label_mask", "point_obj_mask", "point_instance_label", "scan_idx", "pcl_color"], shuffle=True)
        test_dataset_generator = ds.GeneratorDataset(test_dataset, ["point_clouds", "center_label", "heading_class_label", "heading_residual_label", "size_class_label", "size_residual_label", "size_gts", "sem_cls_label", "box_label_mask", "point_obj_mask", "point_instance_label", "scan_idx", "pcl_color"], shuffle=True)
        train_dataset_generator_batch = train_dataset_generator.batch(batch_size=args.batch_size)
        test_dataset_generator_batch = test_dataset_generator.batch(batch_size=args.batch_size)

    elif args.dataset == 'sunrgbd':
        from utils.model_util_sunrgbd import SunrgbdDatasetConfig
        dataset_config = SunrgbdDatasetConfig()
        train_dataset = ScannetDetectionDataset(path=args.data_url,
                      split='train',
                      num_points=args.num_point)
        test_dataset = ScannetDetectionDataset(path=args.data_url,
                      split='val',
                      num_points=args.num_point)

    print('train_len:', train_dataset_generator_batch.get_dataset_size(), \
        'test_len:', test_dataset_generator_batch.get_dataset_size())

    return train_dataset_generator_batch, test_dataset_generator_batch, dataset_config


def get_model(args, dataset_config):
    if args.use_height:
        num_input_channel = int(args.use_color) * 3 + 1
    else:
        num_input_channel = int(args.use_color) * 3
    model = Groupfree3DModel(num_class=dataset_config.num_class,
                            num_heading_bin=dataset_config.num_heading_bin,
                            num_size_cluster=dataset_config.num_size_cluster,
                            mean_size_arr=dataset_config.mean_size_arr,
                            input_feature_dim=num_input_channel,
                            width=args.width,
                            bn_momentum=args.bn_momentum,
                            sync_bn=True if args.syncbn else False,
                            num_proposal=args.num_target,
                            sampling=args.sampling,
                            dropout=args.transformer_dropout,
                            activation=args.transformer_activation,
                            nhead=args.nhead,
                            num_decoder_layers=args.num_decoder_layers,
                            dim_feedforward=args.dim_feedforward,
                            self_position_embedding=args.self_position_embedding,
                            cross_position_embedding=args.cross_position_embedding,
                            size_cls_agnostic=True if args.size_cls_agnostic else False)
    criterion = get_loss
    return model, criterion



def groupfree_3d_scannet_train(args_opt):
    """Group-free 3d train"""
    # context.set_context(mode=context.GRAPH_MODE,
    #                     device_target=args_opt.device_target)

    # Data Pipeline
    dataset = ScannetDetectionDataset(path=args_opt.data_url,
                      split='train',
                      num_points=args_opt.num_point
                      )
    dataset_train = dataset.run()
    # step_size = dataset_train.get_dataset_size()
    step_size = 1

    # Create model
    network = groupfree_3d_model(num_class=args_opt.num_class,
                                 num_heading_bin=args_opt.num_heading_bin,
                                 num_size_cluster=args_opt.num_size_cluster)

    # Set learning rate scheduler
    min_lr = args_opt.min_lr
    max_lr = args_opt.max_lr
    decay_steps = step_size
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)

    # Define optimizer
    # param_dicts = [
    #     {"params": [p for n, p in network.named_parameters(
    #     ) if "decoder" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in network.named_parameters() if "decoder" in n and p.requires_grad],
    #         "lr": args_opt.decoder_learning_rate,
    #     },
    # ]
    network_opt = nn.AdamWeightDecay(network.trainable_params(),
                                     learning_rate=cosine_decay_lr,
                                     weight_decay=args_opt.weight_decay)

    # Define loss function
    # network_loss = GroupFree3DLoss(num_heading_bin=args_opt.num_heading_bin,
    #                                num_size_cluster=args_opt.num_size_cluster,
    #                                num_class=args_opt.num_class,
    #                                num_decoder_layers=args_opt.num_decoder_layers,
    #                                query_points_generator_loss_coef=args_opt.query_points_generator_loss_coef,
    #                                obj_loss_coef=args_opt.obj_loss_coef,
    #                                box_loss_coef=args_opt.box_loss_coef,
    #                                sem_cls_loss_coef=args_opt.sem_cls_loss_coef
    #                                )
    network_loss = TestLoss()

    # Init the model
    model = Model(network, loss_fn=network_loss, optimizer=network_opt,
                  metrics=None)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size,
                                   keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='groupfree_3d_scannet',
                                    directory=args_opt.ckpt_save_dir,
                                    config=ckpt_config)




    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(args_opt.learning_rate)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    from mindspore import context
    context.set_context(max_call_depth=10000)
    context.set_context(mode=context.PYNATIVE_MODE)
    parser = argparse.ArgumentParser(description='Groupfree_3D train.')
    parser.add_argument('--device_target', type=str,
                        default="CPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', type=str,
                        default='/home/tyolm/repos/Group-Free-3D',
                        help='Location of data.')
    parser.add_argument('--num_point', type=int,
                        default=20000, help='Number of points.')
    parser.add_argument('--num_class', type=int,
                        default=18, help='Number of classes.')
    parser.add_argument('--num_heading_bin', type=int,
                        default=1, help='Number of heading bin.')
    parser.add_argument('--num_size_cluster', type=int,
                        default=18, help='Number of cluster size.')
    parser.add_argument('--min_lr', type=float,
                        default=0.00001, help="The min learning rate.")
    parser.add_argument('--max_lr', type=float, default=0.001,
                        help="The max learning rate.")
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004,
                        help='Initial learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='number of decoder layers')
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float,
                        help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float,
                        help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float,
                        help='Loss weight for classification loss')
    parser.add_argument('--keep_checkpoint_max', type=int,
                        default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str,
                        default="./groupfree_3d", help='Location of training outputs.')
    parser.add_argument('--epoch_size', type=int,
                        default=250, help='Train epoch size.')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder.')
    parser.add_argument('--dataset_sink_mode', type=bool,
                        default=False, help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    # print(args)
    groupfree_3d_scannet_train(args)

