import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from deepcrossing import DeepCrossing
from collections import namedtuple

def preprocess_data(df, num_cols, cat_cols):
    df[num_cols] = df[num_cols].fillna(0.0)
    for col in num_cols:
        df[col] = df[col].apply(lambda x: np.log(x+1) if x > -1 else -1)

    df[cat_cols] = df[cat_cols].fillna("-1")

    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    return df[cat_cols+num_cols]


def load_data():
    raw_data = pd.read_csv('../../DeepRecommendationModel/code/data/criteo_sample.txt')
    cat_cols = [col for col in raw_data.columns if 'C' in col]
    num_cols = [col for col in raw_data.columns if 'I' in col]

    return raw_data, cat_cols, num_cols

def get_cat_tuples_list(df, cat_cols):
    cat_tuples = namedtuple('cat_tuples', ('name', 'vocab_size'))
    cat_tuples_list = [cat_tuples(name=col, vocab_size=df[col].nunique()) for col in cat_cols]
    return cat_tuples_list


def train(model, data_iter, device, optimizer, loss, epochs=5):
    model = model.to(device)
    for epoch in range(epochs):
        # l_sum = 0
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            # print(output.squeeze(-1).shape)
            # print(y.shape)
            l = loss(output, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d loss %.2f' % (epoch, l.item()))


if __name__ == '__main__':
    raw_data, cat_cols, num_cols = load_data()
    data = preprocess_data(raw_data, num_cols, cat_cols)
    data['label'] = raw_data['label']
    # print(data)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    num_col_len = len(num_cols)
    cat_col_len = len(cat_cols)
    emb_len = 4
    cat_tuples_list = get_cat_tuples_list(data, cat_cols)

    model = DeepCrossing(input_dim=num_col_len+emb_len*cat_col_len, output_dim=1,
                         hidden_dim=64, cat_tuples_list=cat_tuples_list)
    # model.to(device)
    # print(model)
    #
    X = torch.tensor(data.drop('label', axis=1).values, dtype=torch.float)
    y = torch.tensor(data['label'].values, dtype=torch.long)

    dataset = Data.TensorDataset(X, y)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    train(model, data_iter, device, optimizer, loss)