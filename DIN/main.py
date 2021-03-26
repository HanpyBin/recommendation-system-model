import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import warnings
from collections import namedtuple
from din import DIN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torchsummary import summary
warnings.filterwarnings('ignore')


def load_data():
    file_path = 'movie_sample.txt'
    raw_data = pd.read_csv(file_path, sep='\t', header=None, names=["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id", "label"])
    return raw_data

def preprocess_data(df):
    df_cp = df.copy()
    hist_movie_len = 50
    df_cp['hist_movie_id'] = df_cp['hist_movie_id'].apply(lambda x: x.split(','))
    for i in range(hist_movie_len):
        df_cp['hist_movie_id'+str(i+1)] = df_cp['hist_movie_id'].apply(lambda x: int(x[i]))
    df_cp.drop(['hist_movie_id'], axis=1, inplace=True)
    return df_cp

def get_tuple_list(df, sparse_cols, varlen_cols):
    #  definition
    sparse_feat = namedtuple('sparse_feat', ('vocab_size', 'emb_len'))
    varlen_feat = namedtuple('varlen_feat', ('vocab_size', 'emb_len', 'maxlen'))
    # dense_feat = namedtuple('dense_feat', ('emb_len'))

    sparse_tuple_list = [sparse_feat(vocab_size=max(df[col])+1, emb_len=8) for col in sparse_cols]
    # print(max(df['movie_id'])+1)
    varlen_tuple_list = [varlen_feat(vocab_size=max(df['movie_id'])+1, emb_len=8, maxlen=50)]

    return [sparse_tuple_list, varlen_tuple_list]

def plot_loss_accu(loss_his, accu_his, epochs):
    plt.plot(list(range(1, epochs+1)), loss_his)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss iteration')
    plt.savefig('loss iteration.png', dpi=300)
    plt.cla()
    plt.plot(list(range(1, epochs+1)), accu_his)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('acc iteration')
    plt.savefig('acc iteration.png', dpi=300)


def train(model, data_iter, device, optimizer, loss, epochs=20):
    model = model.to(device)
    loss_his, accu_his = [], []
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            l = loss(outputs, y).sum()

            optimizer.zero_grad()

            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (outputs.argmax(dim=1) == y).sum().item()
            # try:
            #     train_auc_sum += roc_auc_score(y.data.cpu().numpy(), outputs.data.cpu().numpy()[:, 1])
            # except ValueError:
            #     pass
            # n += 1
            n += y.shape[0]
        accu_his.append(train_acc_sum*100/n)
        loss_his.append(train_l_sum/n)
        print('epoch %d loss %f train acc %f' % (epoch + 1, train_l_sum / n, train_acc_sum / n))
    plot_loss_accu(loss_his, accu_his, epochs)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = load_data()
    data = preprocess_data(raw_data)
    sparse_cols = ['user_id', 'gender', 'age', 'movie_id', 'movie_type_id']
    varlen_cols = [col for col in data.columns if 'hist_movie_id' in col]
    varlen_raw_cols = ['hist_movie_id']
    num_cols = ['hist_len']
    behavior_list = ['movie_id']
    behavior_his_list = ['hist_movie_id']

    tuple_list = get_tuple_list(data, sparse_cols, varlen_raw_cols)

    # data.to_csv('temp_csv.csv', index=False)
    #
    # print(data.info())


    X = torch.tensor(data[sparse_cols+varlen_cols+num_cols].values, dtype=torch.float)
    y = torch.tensor(data['label'].values, dtype=torch.long)


    dataset = Data.TensorDataset(X, y)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    model = DIN(sparse_cols, num_cols, varlen_cols, behavior_list, behavior_his_list, tuple_list)

    # print(summary(model, input_size=(X.shape[1], )))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # loss = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # # print(model)
    # train(model, data_iter, device, optimizer, loss, epochs=400)




