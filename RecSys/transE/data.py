import numpy as np
import pandas as pd
import torch
from torch.utils import data

def load_data(path = './WN18/train.txt'):
   data_df = pd.read_csv(path, sep = '\t', names = ['head_word','tail_word','relation'])

   data_df.relation = data_df.relation.astype('category').cat.codes
   data_df.head_word = data_df.head_word.astype('category').cat.codes
   data_df.tail_word = data_df.tail_word.astype('category').cat.codes

   data_df = data_df[['head_word', 'relation', 'tail_word']]

   data_df=data_df.sample(frac=1).reset_index(drop=True)
   frac_idx = int(len(data_df)*0.8)
   train_df = data_df.iloc[:frac_idx,:]
   test_df = data_df.iloc[frac_idx:,:]

   data_tensor = torch.LongTensor(train_df.values)
   train_tensor = torch.LongTensor(train_df.values)
   test_tensor = torch.LongTensor(test_df.values)

   return data_df, data_tensor, train_df, train_tensor, test_df, test_tensor


class WB18Dataset(data.Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        return head, relation, tail