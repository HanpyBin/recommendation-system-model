import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, feature_num):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.feature_num = feature_num
        self.bn = nn.BatchNorm1d(self.feature_num, affine=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # print(x.shape)
        x_norm = self.bn(x)
        x_p = self.sigmoid(x_norm)

        return self.alpha * (1.0-x_p) * x + x_p * x



class LocalActivationUnit(nn.Module):
    def __init__(self, len_num, emb_len=8):
        super(LocalActivationUnit, self).__init__()
        self.linear1 = nn.Linear(4*emb_len, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)
        self.dice1 = Dice(len_num)
        self.dice2 = Dice(len_num)
        self.dice3 = Dice(len_num)
        # self.relu = nn.PReLU()


    def forward(self, keys, query): # query(B, len, emb_len), key(B, 1, emb_len)
        queries = query.repeat(1, keys.shape[1], 1)
        # print('queries', queries.shape)
        # print('keys', keys.shape)
        output = torch.cat([queries, keys, queries-keys, queries*keys], dim=-1) # (B, len, emb_len*4)
        # print(output.shape)
        output = self.dice1(self.linear1(output))
        output = self.dice2(self.linear2(output))
        output = self.dice3(self.linear3(output))
        # output = self.relu(self.linear1(output))
        # output = self.relu(self.linear2(output))
        # output = self.relu(self.linear3(output))
        output = self.linear4(output) # (B, len ,1)

        return output.squeeze(-1)


class AttentionPoolingLayer(nn.Module):
    def __init__(self, len_num, emb_len):
        super(AttentionPoolingLayer, self).__init__()
        self.local_activation_unit = LocalActivationUnit(len_num, emb_len)

    def forward(self, keys, query): # query(B, 1, emb_len), keys(B, len, emb_len)
        # print(keys.shape)
        # print(query.shape)
        query = query.unsqueeze(1)
        key_mask = torch.not_equal(keys[:, :, 0], 0) # (B, len)
        # print(key_mask[0, :].data) # mask通过验证，没问题
        attention_score = self.local_activation_unit(keys, query) # (B, len)
        paddings = torch.zeros_like(attention_score)
        outputs = torch.where(key_mask, attention_score, paddings) # (B, len)
        outputs = outputs.unsqueeze(dim=1) # (B, 1, len)

        outputs = torch.matmul(outputs, keys) # (B, 1, emb_len)
        outputs = outputs.squeeze(dim=1) # (B, emb_len)
        # print(outputs)
        return outputs




class DIN(nn.Module):
    def __init__(self, sparse_cols, dense_cols, varlen_cols, behavior_list, behavior_hist_list, tuple_list, emb_len=8):
        super(DIN, self).__init__()
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        self.varlen_cols = varlen_cols
        self.sparse_col_len = len(self.sparse_cols)
        self.dense_col_len = len(self.dense_cols)
        self.varlen_col_len = len(self.varlen_cols)
        self.sparse_tuple_list = tuple_list[0]
        self.varlen_tuple_list = tuple_list[1]
        self.behavior_list = behavior_list
        self.behavior_hist_list = behavior_hist_list
        self.emb_len = emb_len

        self.attention_pool_layer = AttentionPoolingLayer(self.varlen_col_len, self.emb_len)

        self.linear1 = nn.Linear(self.dense_col_len+self.emb_len*self.sparse_col_len+self.emb_len, 200)
        self.linear2 = nn.Linear(200, 80)
        self.linear3 = nn.Linear(80, 1)
        self.sigmoid = nn.Sigmoid()
        self.dice1 = Dice(200)
        self.dice2 = Dice(80)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        # self.relu = nn.PReLU()
        # sparse features embedding
        self.sparse_embeddings = nn.ModuleList()
        self.varlen_embeddings = nn.ModuleList()
        for fc in self.sparse_tuple_list:
            self.sparse_embeddings.append(nn.Embedding(fc.vocab_size, fc.emb_len))
        for fc in self.varlen_tuple_list:
            self.varlen_embeddings.append(nn.Embedding(fc.vocab_size+1, fc.emb_len, padding_idx=0))


    def forward(self, x):
        # print(self.sparse_col_len)
        sparse_x = x[:, :self.sparse_col_len]
        varlen_x = x[:, self.sparse_col_len:self.sparse_col_len+self.varlen_col_len]
        # print(varlen_x.shape)
        dense_x = x[:, self.sparse_col_len+self.varlen_col_len:] # (B, dense_col_len)

        # sparse embedding
        sparse_embedding_input = self.sparse_embeddings[0](sparse_x[:, 0].long())
        for i in range(1, self.sparse_col_len):
            sparse_embedding_input = torch.cat([sparse_embedding_input, self.sparse_embeddings[i](sparse_x[:, i].long())], dim=1) # (B, emb_len*sparse_col_len)

        # varlen embedding
        varlen_embedding_input = self.varlen_embeddings[0](varlen_x[:, :].long()) # (B, varlen_feature[0]_num, emb_len)
        # print(varlen_x[:, self.sparse_col_len:self.sparse_col_len+self.varlen_col_len].long().shape)
        varlen_input = self.attention_pool_layer(varlen_embedding_input, self.sparse_embeddings[3](sparse_x[:, 3].long())) # movie_id是第3列, varlen_intput:(B, emb_len)

        # print(dense_x.shape)
        # print(sparse_embedding_input.shape)
        # print(varlen_input.shape)
        final_input = torch.cat([dense_x, sparse_embedding_input, varlen_input], dim=1)

        final_output = self.dropout1(self.dice1(self.linear1(final_input)))
        final_output = self.dropout2(self.dice2(self.linear2(final_output)))
        # final_output = self.relu(self.linear1(final_input))
        # final_output = self.relu(self.linear2(final_output))
        final_output = self.sigmoid(self.linear3(final_output))

        final_output = torch.cat([1-final_output, final_output], dim=1)
        return final_output



