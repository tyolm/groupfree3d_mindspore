# groupfree3d_mindspore

This repo is the MindSpore implementation of "Group-Free 3D Object Detection via Transformers" (https://arxiv.org/abs/2104.00678)

## Install
### Requirements
- `Ubuntu 20.04`
- `Anaconda` with `python=3.8`
- `mindspore=1.8`
- `cuda=11.1`
- `plyfile=0.7.4`
- `trimesh=3.14.1`
- `termcolor=2.0.1`

### Data preparation

For SUN RGB-D, follow the [README](./sunrgbd/README.md) under the `sunrgbd` folder.
For ScanNet, follow the [README](./scannet/README.md) under the `scannet` folder.


## Main Results
### ScanNet V2

|Method | backbone | mAP@0.25 | mAP@0.5 | Model |
|:---:|:---:|:---:|:---:|:---:|
|[HGNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Hierarchical_Graph_Network_for_3D_Object_Detection_on_Point_CVPR_2020_paper.pdf)| GU-net| 61.3 | 34.4 | - |
|[GSDN](https://arxiv.org/pdf/2006.12356.pdf)| MinkNet | 62.8 | 34.8 | [waiting for release](https://github.com/jgwak/GSDN) |
|[3D-MPA](https://arxiv.org/abs/2003.13867)| MinkNet | 64.2 | 49.2 |  [waiting for release](https://github.com/francisengelmann/3D-MPA) |
|[VoteNet](https://arxiv.org/abs/1904.09664) | PointNet++ | 62.9 | 39.9 | [official repo](https://github.com/facebookresearch/votenet) |
|[MLCVNet](https://arxiv.org/abs/2004.05679) | PointNet++ | 64.5 | 41.4 | [official repo](https://github.com/NUAAXQ/MLCVNet) |
|[H3DNet](https://arxiv.org/abs/2006.05682) | PointNet++ | 64.4 | 43.4 | [official repo](https://github.com/zaiweizhang/H3DNet) |
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ | 67.2| 48.1 | [official repo](https://github.com/zaiweizhang/H3DNet) |
| Ours | PointNet++ |  |  |  |

### SUN RGB-D

|Method | backbone | inputs | mAP@0.25 | mAP@0.5 | Model |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)| PointNet++ |point | 59.1 | 35.8 |[official repo](https://github.com/facebookresearch/votenet)|
|[MLCVNet](https://arxiv.org/abs/2004.05679)|PointNet++ | point | 59.8 | - | [official repo](https://github.com/NUAAXQ/MLCVNet) |
|[HGNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Hierarchical_Graph_Network_for_3D_Object_Detection_on_Point_CVPR_2020_paper.pdf)| GU-net |point | 61.6 |-|-|
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ |point | 60.1 | 39.0 | [official repo](https://github.com/zaiweizhang/H3DNet) |
|[imVoteNet](https://arxiv.org/abs/2001.10692)|PointNet++|point+RGB| 63.4 | - |  [official repo](https://github.com/facebookresearch/imvotenet)|
| Ours| PointNet++ | point |  |  | |

## Usage

### ScanNet
For training:
```
python groupfree3d_scannet_train.py
```
For evaluation:
```
python groupfree3d_scannet_eval.py
```
### SUN RGB-D
For training:
```
python groupfree3d_sunrgbd_train.py
```
For evaluation:
```
python groupfree3d_sunrgbd_eval.py
```
## Acknowledgements
We thank a lot for the official repo of [group-free-3d](https://github.com/zeliu98/Group-Free-3D).

## License
The code is released under MIT License (see LICENSE file for details).











