import torch
import pandas as pd

feat_idx = [i for i in range(10)] # TODO: Select suitable feature columns.
print(feat_idx)

print(torch.cuda.is_available())
train_data  = pd.read_csv('./data/covid.test_un.csv').values
print(type(train_data))
print(train_data.shape)
print(train_data)