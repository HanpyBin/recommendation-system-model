import torch
import torch.nn as nn


def BInteractionPolling(x): #  (batch_size, feature_num, emb_len)
    left = torch.square(torch.sum(x, dim=1)) # (batch_size, emb_len)
    right = torch.sum(torch.square(x), dim=1) # (batch_size, emb_len)
    return 0.5 * (left - right) # (batch_size, emb_len)


class NFM(nn.Module):
    def __init__(self, num_cols, cat_cols, cat_tuple_list, emb_len=4):
        super(NFM, self).__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_col_len = len(self.num_cols)
        self.cat_col_len = len(self.cat_cols)
        self.emb_len = emb_len
        self.cat_tuple_list = cat_tuple_list

        # 一阶特征交叉
        self.degree1_linear_num = nn.Linear(self.num_col_len, 1)
        self.degree1_linear_cat = nn.ModuleList()
        for fc in cat_tuple_list:
            self.degree1_linear_cat.append(nn.Embedding(fc.vocab_size, 1))

        # Pooling层
        self.degree2_cat = nn.ModuleList()
        for fc in cat_tuple_list:
            self.degree2_cat.append(nn.Embedding(fc.vocab_size, self.emb_len))

        # DNN
        self.linear1 = nn.Linear(emb_len, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(num_features=self.emb_len)

    def forward(self, x):
        cat_x = x[:, :self.cat_col_len]
        num_x = x[:, self.cat_col_len:]

        # 一阶
        degree1_num_output = self.degree1_linear_num(num_x)
        degree1_cat_output_list = []
        for i in range(self.cat_col_len):
            degree1_cat_output_list.append(self.degree1_linear_cat[i](cat_x[:, i].long()))
        degree1_cat_output = degree1_cat_output_list[0]
        for i in range(1, self.cat_col_len):
            degree1_cat_output += degree1_cat_output_list[i]

        degree1_output = degree1_cat_output + degree1_num_output

        # 高阶
        pooling_input = self.degree2_cat[0](cat_x[:, 0].long()).unsqueeze(1)
        for i in range(1, self.cat_col_len):
            pooling_input = torch.cat([pooling_input, self.degree2_cat[i](cat_x[:, i].long()).unsqueeze(1)], dim=1)
        pooling_output = BInteractionPolling(pooling_input) # (batch_size, emb_len)

        # DNN
        pooling_output = self.batchnorm(pooling_output)
        highrank_output = self.dropout1(self.relu(self.linear1(pooling_output)))
        highrank_output = self.dropout2(self.relu(self.linear2(highrank_output)))
        highrank_output = self.dropout3(self.relu(self.linear3(highrank_output)))
        highrank_output = self.linear4(highrank_output)

        ouput = self.sigmoid(highrank_output+degree1_output)
        output = torch.cat([ouput, 1-ouput], dim=1)
        return output

