import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
from nfm import NFM
import matplotlib.pyplot as plt


def load_data():
    # file_path = '../../DeepRecommendationModel/code/data/criteo_sample.txt'
    file_path = 'train.txt'
    raw_data = pd.read_csv(file_path)
    raw_data.drop(['Id'], axis=1, inplace=True)
    cat_cols = [col for col in raw_data.columns.values if 'C' in col]
    num_cols = [col for col in raw_data.columns.values if 'I' in col]

    return raw_data, cat_cols, num_cols


def preprocess_data(df, cat_cols, num_cols):
    df_cp = df.copy()
    df_cp[num_cols] = df_cp[num_cols].fillna(0.0)
    for col in num_cols:
        df_cp[col] = df_cp[col].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    df_cp[cat_cols] = df_cp[cat_cols].fillna("-1")
    for col in cat_cols:
        encoder = LabelEncoder()
        df_cp[col] = encoder.fit_transform(df_cp[col])

    return df_cp[cat_cols + num_cols]

def get_cat_tuple_list(df, cat_cols):
    cat_tuple = namedtuple('cat_tuple', ('name', 'vocab_size'))
    cat_tuple_list = [cat_tuple(name=col, vocab_size=df[col].nunique()) for col in cat_cols]
    return cat_tuple_list

def plot_loss_accu(loss_his, accu_his, epochs):
    plt.plot(list(range(1, epochs+1)), loss_his)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss iteration')
    plt.savefig('loss iteration.png', dpi=300)
    plt.cla()
    plt.plot(list(range(1, epochs+1)), accu_his)
    plt.xlabel('epoch')
    plt.ylabel('accuracy/%')
    plt.title('accuracy iteration')
    plt.savefig('accuracy iteration.png', dpi=300)

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

            n += y.shape[0]
        accu_his.append(train_acc_sum*100/n)
        loss_his.append(train_l_sum/n)
        print('epoch %d loss %f train acc %f' % (epoch + 1, train_l_sum / n, train_acc_sum / n))
    plot_loss_accu(loss_his, accu_his, epochs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data, cat_cols, num_cols = load_data()
    train_data = preprocess_data(raw_data, cat_cols, num_cols)
    # train_data['label'] = raw_data['label']
    train_data['label'] = raw_data['Label']

    cat_tuple_list = get_cat_tuple_list(train_data, cat_cols)

    X = torch.tensor(train_data.drop(['label'], axis=1).values, dtype=torch.float)
    y = torch.tensor(train_data.label.values, dtype=torch.long)

    dataset = Data.TensorDataset(X, y)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    model = NFM(num_cols, cat_cols, cat_tuple_list)
    # print(model)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # print(model)
    train(model, data_iter, device, optimizer, loss, epochs=200)
