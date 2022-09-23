""" Group-free 3D ScanNet eval script. """


import argparse
import os
import sys
import time
from turtle import back
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.train import Model
from mindspore.ops import stop_gradient
import mindspore.communication as comm
import mindspore.dataset as ds
import mindspore.ops as ops
from datetime import datetime


base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(base_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'dataset'))
sys.path.append(os.path.join(root_dir, 'models'))
sys.path.append(os.path.join(root_dir, 'engine'))
sys.path.append(os.path.join(root_dir, 'utils'))
sys.path.append(os.path.join(root_dir, 'models/ops'))

from models.groupfree_3d_detection import Groupfree3DModel
from dataset.scannet_v2 import ScannetDetectionDataset
# from ops.test_loss import TestLoss
from utils.model_util_scannet import ScannetDatasetConfig
from engine.ops.groupfree_3d_ap_helper import APCalculator, parse_predictions, parse_groundtruths
from utils.groupfree_3d_logger import setup_logger
from engine.ops.groupfree_3d_loss_helper import GroupFree3DLoss
from engine.ops.groupfree_3d_losses import get_loss

def parse_option():
    parser = argparse.ArgumentParser(description='Groupfree_3D Finetune')
    # eval
    parser.add_argument('--checkpoint_path', default='/home/tyolm/codes/torch_to_ms/model/scannet_l6o256.ckpt', help='Model checkpoint path [default: None]')
    parser.add_argument('--avg_times', default=5, type=int, help='Average times')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument('--dump_dir', default='dump', help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                        help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--faster_eval', action='store_true',
                        help='Faster evaluation by skippling empty bounding box removal.')
    parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')

    # model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')
    parser.add_argument('--data_root', default='/home/tyolm/datasets/groupfree3d', help='data root path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')

    args = parser.parse_known_args()[0]

    return args


def get_dataset(args):
    if args.dataset == 'scannet':
        from utils.model_util_scannet import ScannetDatasetConfig
        dataset_config = ScannetDatasetConfig()
        test_dataset = ScannetDetectionDataset(data_root=args.data_root,
                      split_set='val',
                      num_points=args.num_point)
        test_dataset_generator = ds.GeneratorDataset(test_dataset, ["point_clouds", "center_label", "heading_class_label", "heading_residual_label", "size_class_label", "size_residual_label", "size_gts", "sem_cls_label", "box_label_mask", "point_obj_mask", "point_instance_label", "scan_idx", "pcl_color"], shuffle=True)
        test_dataset_generator_batch = test_dataset_generator.batch(batch_size=args.batch_size)

    elif args.dataset == 'sunrgbd':
        from utils.model_util_sunrgbd import SunrgbdDatasetConfig
        dataset_config = SunrgbdDatasetConfig()
        test_dataset = ScannetDetectionDataset(path=args.data_url,
                      split='val',
                      num_points=args.num_point)

    print('test_len:', test_dataset_generator_batch.get_dataset_size())

    return test_dataset_generator_batch, dataset_config

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

def load_checkpoint(args, model):
    ckpt_dir = args.checkpoint_path
    param_dict = ms.load_checkpoint(ckpt_dir)
    ms.load_param_into_net(model, parameter_dict=param_dict)
    # miss = ms.load_param_into_net(model, parameter_dict=param_dict)
    # print('miss', miss)
    logger.info('checkpoint loaded successfully!')
    return model


def evaluate_one_time(test_dataset, dataset_config, config_dict, AP_IOU_THRESHOLDS, model, criterion, args, time=0):
    # context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    stat_dict = {}
    if args.num_decoder_layers > 0:
        if args.dataset == 'sunrgbd':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['all_layers_']
        elif args.dataset == 'scannet':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['last_three_'] + ['all_layers_']
    else:
        prefixes = ['proposal_']  # only proposal
        _prefixes = prefixes

    if args.num_decoder_layers >= 3:
        last_three_prefixes = ['last_', f'{args.num_decoder_layers - 2}head_', f'{args.num_decoder_layers - 3}head_']
    elif args.num_decoder_layers == 2:
        last_three_prefixes = ['last_', '0head_']
    elif args.num_decoder_layers == 1:
        last_three_prefixes = ['last_']
    else:
        last_three_prefixes = []

    ap_calculator_list = [APCalculator(iou_thresh, dataset_config.class2type) \
                        for iou_thresh in AP_IOU_THRESHOLDS]

    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    test_iterator = test_dataset.create_dict_iterator()

    for batch_idx, batch_data_label in enumerate(test_iterator):
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = model(inputs)

        for key in batch_data_label:
            assert key not in end_points
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, dataset_config,
                                     num_decoder_layers=args.num_decoder_layers,
                                     query_points_generator_loss_coef=args.query_points_generator_loss_coef,
                                     obj_loss_coef=args.obj_loss_coef,
                                     box_loss_coef=args.box_loss_coef,
                                     sem_cls_loss_coef=args.sem_cls_loss_coef,
                                     query_points_obj_topk=args.query_points_obj_topk,
                                     center_loss_type=args.center_loss_type,
                                     center_delta=args.center_delta,
                                     size_loss_type=args.size_loss_type,
                                     size_delta=args.size_delta,
                                     heading_loss_type=args.heading_loss_type,
                                     heading_delta=args.heading_delta,
                                     size_cls_agnostic=args.size_cls_agnostic)

        # accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].asnumpy().item()

        for prefix in prefixes:
            if prefix == 'last_three_':
                end_points[f'{prefix}center'] = ops.concat(([end_points[f'{ppx}center']
                                                           for ppx in last_three_prefixes]), 1)
                end_points[f'{prefix}heading_scores'] = ops.concat(([end_points[f'{ppx}heading_scores']
                                                                   for ppx in last_three_prefixes]), 1)
                end_points[f'{prefix}heading_residuals'] = ops.concat(([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in last_three_prefixes]), 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = ops.concat(([end_points[f'{ppx}pred_size']
                                                                  for ppx in last_three_prefixes]), 1)
                else:
                    end_points[f'{prefix}size_scores'] = ops.concat(([end_points[f'{ppx}size_scores']
                                                                    for ppx in last_three_prefixes]), 1)
                    end_points[f'{prefix}size_residuals'] = ops.concat(([end_points[f'{ppx}size_residuals']
                                                                       for ppx in last_three_prefixes]), 1)
                end_points[f'{prefix}sem_cls_scores'] = ops.concat(([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in last_three_prefixes]), 1)
                end_points[f'{prefix}objectness_scores'] = ops.concat(([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in last_three_prefixes]), 1)

            elif prefix == 'all_layers_':
                end_points[f'{prefix}center'] = ops.concat(([end_points[f'{ppx}center']
                                                           for ppx in _prefixes]), 1)
                end_points[f'{prefix}heading_scores'] = ops.concat(([end_points[f'{ppx}heading_scores']
                                                                   for ppx in _prefixes]), 1)
                end_points[f'{prefix}heading_residuals'] = ops.concat(([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in _prefixes]), 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = ops.concat(([end_points[f'{ppx}pred_size']
                                                                  for ppx in _prefixes]), 1)
                else:
                    end_points[f'{prefix}size_scores'] = ops.concat(([end_points[f'{ppx}size_scores']
                                                                    for ppx in _prefixes]), 1)
                    end_points[f'{prefix}size_residuals'] = ops.concat(([end_points[f'{ppx}size_residuals']
                                                                       for ppx in _prefixes]), 1)
                end_points[f'{prefix}sem_cls_scores'] = ops.concat(([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in _prefixes]), 1)
                end_points[f'{prefix}objectness_scores'] = ops.concat(([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in _prefixes]), 1)

            batch_pred_map_cls = parse_predictions(end_points, config_dict, prefix,
                                                  size_cls_agnostic=args.size_cls_agnostic)
            batch_gt_map_cls = parse_groundtruths(end_points, config_dict,
                                                  size_cls_agnostic=args.size_cls_agnostic)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f'T[{time}] Eval: [{batch_idx + 1}/{test_dataset.get_dataset_size()}]  ' + ''.join(
                [f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                 for key in sorted(stat_dict.keys()) if 'loss' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if
                                 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'last_' in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
            for ihead in range(args.num_decoder_layers - 2, -1, -1):
                logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                     for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            logger.info(f'=====================>{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}<=====================')
            for key in metrics_dict:
                logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

    for mAP in mAPs:
        logger.info(f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

    return mAP, mAPs


def groupfree_3d_scannet_finetune(args, avg_times=1):
    test_dataset, dataset_config = get_dataset(args)
    n_data = test_dataset.get_dataset_size()
    logger.info("length of testing dataset:", n_data)

    model, criterion = get_model(args, dataset_config)
    model = load_checkpoint(args, model)
    # logger.info(str(model))
    config_dict = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': dataset_config}

    logger.info(str(datetime.now()))
    mAPs_times = [None for i in range(avg_times)]
    for i in range(avg_times):
        np.random.seed(i + args.rng_seed)
        mAPs = evaluate_one_time(test_dataset, dataset_config, config_dict, args.ap_iou_thresholds, model, criterion, args, i)
        mAPs_times[i] = mAPs

    mAPs_avg = mAPs

    for i, mAP in enumerate(mAPs_avg):
        for key in mAP[1].keys():
            avg = 0
            for t in range(avg_times):
                cur = mAPs_times[t][i][1][key]
                avg += cur
            avg /= avg_times
            mAP[1][key] = avg

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \n' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \n' for key in sorted(mAP[1].keys())]))

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \t' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    args = parse_option()
    dataset_config = ScannetDatasetConfig()
    input_file = open('./input.txt', 'w')
    output_file = open('./output.txt', 'w')
    if args.use_height:
        num_input_channel = int(args.use_color) * 3 + 1
    else:
        num_input_channel = int(args.use_color) * 3
    # mindspore model
    ckpt = '/home/tyolm/codes/torch_to_ms/model/scannet_l6o256.ckpt'
    ms_param_dict = ms.load_checkpoint(ckpt)
    ms_model = Groupfree3DModel(num_class=dataset_config.num_class,
                              num_heading_bin=dataset_config.num_heading_bin,
                              num_size_cluster=dataset_config.num_size_cluster,
                              mean_size_arr=dataset_config.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=args.width,
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
    ms.load_param_into_net(ms_model, ms_param_dict)
    param_not_load = ms.load_param_into_net(ms_model, ms_param_dict)

    np.random.seed(41)
    point_clouds = ms.Tensor(np.random.randn(4, 20000, 3), ms.float32)
    print(point_clouds, file=input_file)
    inputs = {'point_clouds': point_clouds}
    backbone_output, end_points_ms = ms_model(inputs)
    for key in backbone_output:
        print(key, backbone_output[key].asnumpy().tolist(), '\n', file=output_file)




    # opt.dump_dir = os.path.join(opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    # logger = setup_logger(output=opt.dump_dir, name="eval")

    # groupfree_3d_scannet_finetune(opt, opt.avg_times)





#     # Data Pipeline
#     dataset_config = ScannetDatasetConfig()
#     dataset = ScannetDetectionDataset(split_set='val',
#                       num_points=args_opt.num_point,
#                       data_root=args_opt.data_url)
#     dataset_generator = ds.GeneratorDataset(dataset, ["point_clouds", "center_label", 
#                                                       "heading_class_label", "heading_residual_label", "size_class_label", "size_residual_label", "size_gts", "sem_cls_label", "box_label_mask", "point_obj_mask", "point_instance_label", "scan_idx", "pcl_color"], 
#                                                       shuffle=True)
#     dataset_generator_batch = dataset_generator.batch(args_opt.batch_size, drop_remainder=True)
#     iterator = dataset_generator_batch.create_dict_iterator()

#     ckpt_dir = args_opt.ckpt_dir
#     param_dict = ms.load_checkpoint(ckpt_dir)
#     network = Groupfree3DModel(num_class=dataset_config.num_class,
#                                 num_heading_bin=dataset_config.num_heading_bin,
#                                 num_size_cluster=dataset_config.num_size_cluster,
#                                 mean_size_arr=dataset_config.mean_size_arr)

#     param_dict = network.parameters_dict()
#     ms.load_param_into_net(network, param_dict)

#     dataset_config = ScannetDatasetConfig()
#     stat_dict = {}
#     if args_opt.num_decoder_layers > 0:
#         prefixes = ['last_', 'proposal_'] + [str(i) + 'head_' for i in range(args_opt.num_decoder_layers - 1)]
#     else:
#         prefixes = ['proposal_'] # only proposal
#     ap_calculator_list = [APCalculator(iou_thresh, dataset_config.class2type) \
#                          for iou_thresh in args_opt.ap_iou_thresholds]
#     mAPs = [[iou_thresh, {k: 0 for k in prefixes}] \
#             for iou_thresh in args_opt.ap_iou_thresholds]

#     # Define loss function
#     network_loss = GroupFree3DLoss(num_heading_bin=args_opt.num_heading_bin,
#                                    num_size_cluster=args_opt.num_size_cluster,
#                                    num_class=args_opt.num_class,
#                                    num_decoder_layers=args_opt.num_decoder_layers,
#                                    query_points_generator_loss_coef=args_opt.query_points_generator_loss_coef,
#                                    obj_loss_coef=args_opt.obj_loss_coef,
#                                    box_loss_coef=args_opt.box_loss_coef,
#                                    sem_cls_loss_coef=args_opt.sem_cls_loss_coef
#                                    )
#     # network_loss = TestLoss()

#     # Init the model
#     # model = Model(network, loss_fn=network_loss, optimizer=None,
#     #               metrics=None)

#     logger = setup_logger(output=args_opt.log_dir, name="group-free")
#     # model.eval(dataset_val, dataset_sink_mode=False)
#     config_dict= {'remove_empty_box': False, 'use_3d_nms': True,
#                    'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
#                    'per_class_proposal': True, 'conf_thresh': 0.0,
#                    'dataset_config': dataset_config}
#     batch_pred_map_cls_dict = {k: [] for k in prefixes}
#     batch_gt_map_cls_dict = {k: [] for k in prefixes}

#     for batch_idx, batch_data in enumerate(iterator):
#         # inputs = {'point_clouds': batch_data['point_clouds']}
#         inputs = batch_data['point_clouds'] # [B, N, 3]
#         end_points = network(inputs)
#         # end_points = stop_gradient(end_points)

#         # computer loss
#         for key in batch_data:
#             assert key not in end_points
#             end_points[key] = batch_data[key]
#         _, end_points = get_loss(end_points, dataset_config,
#                                     num_decoder_layers=args_opt.num_decoder_layers,
#                                     query_points_generator_loss_coef=args_opt.query_points_generator_loss_coef,
#                                     obj_loss_coef=args_opt.obj_loss_coef,
#                                     box_loss_coef=args_opt.box_loss_coef,
#                                     sem_cls_loss_coef=args_opt.sem_cls_loss_coef,
#                                     query_points_obj_topk=args_opt.query_points_obj_topk,
#                                     center_loss_type=args_opt.center_loss_type,
#                                     center_delta=args_opt.center_delta,
#                                     size_loss_type=args_opt.size_loss_type,
#                                     size_delta=args_opt.size_delta,
#                                     heading_loss_type=args_opt.heading_loss_type,
#                                     heading_delta=args_opt.heading_delta,
#                                     size_cls_agnostic=args_opt.size_cls_agnostic)

#         # Accumulate statistics and print out
#         for key in end_points:
#             if 'loss' in key or 'acc' in key or 'ratio' in key:
#                 if key not in stat_dict:stat_dict[key] = 0
#                 if isinstance(end_points[key], float):
#                     stat_dict[key] += end_points[key]
#                 else:
#                     print('end_points[key]', end_points[key])
#                     stat_dict[key] += end_points[key].asnumpy().item()

#         for prefix in prefixes:
#             batch_pred_map_cls = parse_predictions(end_points, config_dict, prefix,
#                                                   size_cls_agnostic=args.size_cls_agnostic)
#             batch_gt_map_cls = parse_groundtruths(end_points, config_dict,
#                                                   size_cls_agnostic=args.size_cls_agnostic)
#             batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
#             batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

#         if (batch_idx + 1) % args.print_freq == 0:
#             logger.info(f'Eval: [{batch_idx + 1}/{dataset_generator_batch.get_dataset_size()}]  ' + ''.join(
#                 [f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
#                  for key in sorted(stat_dict.keys()) if 'loss' not in key]))
#             logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
#                                  for key in sorted(stat_dict.keys()) if
#                                  'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
#             logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
#                                  for key in sorted(stat_dict.keys()) if 'last_' in key]))
#             logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
#                                  for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
#             for ihead in range(args.num_decoder_layers - 2, -1, -1):
#                 logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
#                                      for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

#     mAP = 0.0
#     for prefix in prefixes:
#         for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
#                                                           batch_gt_map_cls_dict[prefix]):
#             for ap_calculator in ap_calculator_list:
#                 ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
#         # Evaluate average precision
#         for i, ap_calculator in enumerate(ap_calculator_list):
#             metrics_dict = ap_calculator.compute_metrics()
#             logger.info(f'=====================>{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}<=====================')
#             for key in metrics_dict:
#                 logger.info(f'{key} {metrics_dict[key]}')
#             if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
#                 mAP = metrics_dict['mAP']
#             mAPs[i][1][prefix] = metrics_dict['mAP']
#             ap_calculator.reset()

#     for mAP in mAPs:
#         logger.info(f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

#     return mAP, mAPs


# if __name__ == '__main__':
#     opt = parse_option()
#     opt.dump_dir = os.path.join(opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
#     logger = setup_logger(output=opt.dump_dir, name='eval')
#     groupfree_3d_scannet_finetune(opt, opt.avg_times)
