"""Position Embedding"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(max_call_depth=10000)
ms.set_context(mode=ms.PYNATIVE_MODE)

class PositionEmbeddingLearned(nn.Cell):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.unsqueeze = ops.ExpandDims()
        self.conv1 = nn.Conv1d(input_channel, num_pos_feats, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(num_pos_feats)
        self.conv2 = nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1, has_bias=True)
        self.relu = nn.ReLU()
        # self.position_embedding_head = nn.SequentialCell(
        #     nn.Conv1d(input_channel, num_pos_feats, kernel_size=1, has_bias=True),
        #     nn.BatchNorm2d(self.unsqueeze(Tensor(num_pos_feats), -1)).squeeze(-1),
        #     # nn.BatchNorm1d(num_pos_feats),
        #     nn.ReLU(),
        #     nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1, has_bias=True))

    def construct(self, xyz):
        if xyz.shape[1] != 3:
            xyz = xyz.transpose(0, 2, 1)
        position_embedding = self.conv1(xyz)
        position_embedding = self.bn1(self.unsqueeze(position_embedding, -1)).squeeze(-1)
        position_embedding = self.relu(position_embedding)
        position_embedding = self.conv2(position_embedding)
        # print(xyz.shape)
        # position_embedding = self.position_embedding_head(xyz)
        return position_embedding