import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    定义残差块
    """
    def __init__(self, input_dim, hidden_dim):
        """
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        """
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        inputs = x
        inputs = self.relu1(self.linear1(inputs))
        # print(self.linear1.weight.data)
        inputs = self.relu2(self.linear2(inputs))
        inputs = self.relu3(inputs + x)
        return inputs


def get_residual_blocks(input_dim, output_dim, block_num):
    layers = []
    for i in range(block_num):
        layers.append(ResidualBlock(input_dim, output_dim))
    return nn.Sequential(*layers)


class DeepCrossing(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cat_tuples_list, device, emb_len=4):
        """

        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param hidden_dim: 残差块的隐藏层维度
        :param cat_tuples_list:
        :param device:
        :param emb_len: embedding的表示长度
        """
        # input_dim = len(num_cols) + emb_len*len(cat_cols)
        super(DeepCrossing, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_len = emb_len

        self.cat_col_num = len(cat_tuples_list)
        self.embeddings = nn.ModuleList()
        for fc in cat_tuples_list:
            self.embeddings.append(nn.Embedding(fc.vocab_size, self.emb_len))
        self.residual_blocks = get_residual_blocks(self.input_dim, self.hidden_dim, 3)
        self.dense = nn.Linear(input_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):

        cat_x = x[:, 0:self.cat_col_num]
        num_x = x[:, self.cat_col_num:]
        # print(self.dense.weight)
        # print(self.device)
        for i in range(self.cat_col_num):
            # print(cat_x[:, i].long())
            # print(self.embeddings[i])
            # print(cat_x[:, i].long())
            # print(self.embeddings[i].weight)
            temp_tensor = self.embeddings[i](cat_x[:, i].long())
            # print(num_x.data.shape)
            num_x = torch.cat((num_x, temp_tensor), dim=1)

        num_x = self.residual_blocks(num_x)
        num_x = self.dense(num_x)
        num_x = self.act(num_x)
        return num_x

