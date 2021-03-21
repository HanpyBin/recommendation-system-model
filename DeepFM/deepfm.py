import torch
import torch.nn as nn


def FM(x): # x:(batch, feature_num, emb_len)
    left = torch.square(torch.sum(x, dim=1, keepdim=True)) # (batch, 1, emb_len)
    right = torch.sum(torch.square(x), dim=1, keepdim=True) # (batch, 1, emb_len)
    res = 0.5 * torch.sum((left - right), dim=2) # (batch, 1)
    return res




class DeepFM(nn.Module):
    def __init__(self, num_cols, cat_cols, cat_tuple_list, emb_len=4):
        super(DeepFM, self).__init__()
        self.cat_col_len = len(cat_cols)
        self.num_col_len = len(num_cols)
        self.cat_tuple_list = cat_tuple_list
        self.emb_len = emb_len
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.deep_input_dim = self.num_col_len + self.emb_len * self.cat_col_len
        # three part: linear, fm, dnn

        # linear
        self.fm_linear_embeddings = nn.ModuleList()
        for fc in self.cat_tuple_list:
            self.fm_linear_embeddings.append(nn.Embedding(fc.vocab_size, 1))

        self.linear_dense = nn.Linear(self.num_col_len, 1)

        # FM
        self.fm_embeddings = nn.ModuleList()
        for fc in self.cat_tuple_list:
            self.fm_embeddings.append(nn.Embedding(fc.vocab_size, emb_len))

        # DNN
        self.deep_linear1 = nn.Linear(self.deep_input_dim, 1024)
        self.deep_linear2 = nn.Linear(1024, 512)
        self.deep_linear3 = nn.Linear(512, 256)
        self.deep_final_linear = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.1)

        # total dense
        self.final_linear = nn.Linear(3, 2)

    def forward(self, x):
        cat_x = x[:, :self.cat_col_len]
        num_x = x[:, self.cat_col_len:]

        # linear
        linear_num_output = self.linear_dense(num_x)
        linear_output_list = []
        for i in range(self.cat_col_len):
            linear_output_list.append(self.fm_linear_embeddings[i](cat_x[:, i].long()))
        linear_cat_output = linear_output_list[0]
        for i in range(1, self.cat_col_len):
            linear_cat_output += linear_output_list[i]
        linear_output = linear_cat_output + linear_num_output

        # FM
        fm_input = self.fm_embeddings[0](cat_x[:, 0].long()).unsqueeze(dim=1)
        for i in range(1, self.cat_col_len):
            fm_input = torch.cat([fm_input, self.fm_embeddings[i](cat_x[:, i].long()).unsqueeze(dim=1)], dim=1)
        fm_output = FM(fm_input)

        # dnn

        for i in range(self.cat_col_len):
            num_x = torch.cat([num_x, self.fm_embeddings[i](cat_x[:, i].long())], dim=1)
        # print(num_x.shape)
        deep_output = self.dropout1(self.relu(self.deep_linear1(num_x)))
        deep_output = self.dropout2(self.relu(self.deep_linear2(deep_output)))
        deep_output = self.dropout3(self.relu(self.deep_linear3(deep_output)))
        deep_output = self.deep_final_linear(deep_output)

        # 总和
        output = self.sigmoid(linear_output + fm_output + deep_output)
        return torch.cat([output, 1-output], dim=1)
        # return self.final_linear(torch.cat([linear_output, fm_output, deep_output], dim=1))


