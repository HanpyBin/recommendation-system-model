import torch
import torch.nn as nn

class WideDeep(nn.Module):
    def __init__(self, num_cols, cat_cols, cat_tuple_list, emb_len=4):
        super(WideDeep, self).__init__()
        # TODO: 对啊，为什么tf的代码里面，embedding不是共享的？艹，那我这里怎么写？

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_col_len = len(self.cat_cols)
        self.num_col_len = len(self.num_cols)
        self.cat_tuple_list = cat_tuple_list
        self.emb_len = emb_len
        self.deep_input_dim = self.emb_len * self.cat_col_len + self.num_col_len

        self.final_linear = nn.Linear(2, 2)
        # 分为wide part和deep part
        # deep部分
        self.deep_embeddings = nn.ModuleList() # 这里的embedding应该是wide和deep共享的
                                               # 先令deep和wide各自占有一个embedding层
        for cat_tuple in self.cat_tuple_list:
            self.deep_embeddings.append(nn.Embedding(cat_tuple.vocab_size, self.emb_len))
        self.deep_linear1 = nn.Linear(self.deep_input_dim, 1024)
        self.deep_linear2 = nn.Linear(1024, 512)
        self.deep_linear3 = nn.Linear(512, 256)
        self.deep_final_linear = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.1)

        # Wide部分
        self.wide_embeddings = nn.ModuleList()
        for cat_tuple in cat_tuple_list:
            self.wide_embeddings.append(nn.Embedding(cat_tuple.vocab_size, 1))
        self.wide_num_linear = nn.Linear(self.num_col_len, 1)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        cat_x = x[:, 0:self.cat_col_len]
        num_x = x[:, self.cat_col_len:]

        # 分别针对wide层和deep层进行处理
        # wide层
        wide_num_output = self.wide_num_linear(num_x)
        wide_cat_input = []

        for i in range(self.cat_col_len):
            wide_cat_input.append(self.wide_embeddings[i](cat_x[:, i].long()))
        wide_cat_output = wide_cat_input[0]
        for i in range(1, len(wide_cat_input)):
            wide_cat_output += wide_cat_input[i]
        wide_output = wide_cat_output + wide_cat_output

        ## deep层
        for i in range(self.cat_col_len):
            num_x = torch.cat([num_x, self.deep_embeddings[i](cat_x[:, i].long())], dim=1)
        deep_output = self.dropout1(self.relu(self.deep_linear1(num_x)))
        deep_output = self.dropout2(self.relu(self.deep_linear2(deep_output)))
        deep_output = self.dropout3(self.relu(self.deep_linear3(deep_output)))
        deep_output = self.deep_final_linear(deep_output)

        # 结合wide&deep
        output = self.sigmoid(self.final_linear(torch.cat([wide_output, deep_output], dim=1)))
        return output
