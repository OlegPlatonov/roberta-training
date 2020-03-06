from abc import ABC, ABCMeta

import pandas as pd
import torch
import torch.utils.data as data
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer('vocabs/roberta-large-vocab.json',
                             'vocabs/roberta-large-merges.txt',
                             additional_special_tokens=['<gap>'])


gap_token_id = tokenizer.convert_tokens_to_ids(['<gap>'])[0]
pad_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]


def fragment_transform(fragment):
    return ['<s>'] + fragment.split() + ['</s>', '</s>']


def text_transform(text):
    return text.split() + ['</s>']


class DatasetRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.task] = new_cls
        return new_cls

    @classmethod
    def get_dataset(mcs, task):
        return mcs.registry[task]


class BaseDataset(ABC, data.Dataset, metaclass=DatasetRegistry):
    task = None


class DatasetForGappedText(BaseDataset):
    task = 'GT'

    def __init__(self, data_path, num_fragments=8):
        super(DatasetForGappedText, self).__init__()
        self.num_fragments = num_fragments
        self.data = dict()
        self.data['text'] = list(pd.read_csv(data_path, usecols=['text'], squeeze=True,
                                             dtype='str', engine='c'))
        for i in range(1, num_fragments + 1):
            self.data[f'fragment_{i}'] = list(pd.read_csv(data_path, usecols=[f'fragment_{i}'], squeeze=True,
                                                          dtype='str', engine='c'))
            self.data[f'target_gap_{i}'] = list(pd.read_csv(data_path, usecols=[f'target_gap_{i}'], squeeze=True,
                                                            dtype='int', engine='c'))

    def __len__(self):
        return len(self.data['text']) * self.num_fragments

    def __getitem__(self, idx):
        text_idx = idx // self.num_fragments
        fragment_idx = idx % self.num_fragments + 1

        text = self.data['text'][text_idx]
        fragment = self.data[f'fragment_{fragment_idx}'][text_idx]
        target_gap = self.data[f'target_gap_{fragment_idx}'][text_idx]

        return text, fragment, target_gap


def pad_2d(array_2d, pad_value=0):
    row_lengths = [len(row) for row in array_2d]
    max_len = max(row_lengths)
    for i in range(len(array_2d)):
        array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

    return array_2d


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    gap_ids = []
    target_gaps = []

    for text, fragment, target_gap in batch:
        text_sequence = text_transform(text)
        text_sequence = tokenizer.convert_tokens_to_ids(text_sequence)
        fragment_sequence = fragment_transform(fragment)
        fragment_sequence = tokenizer.convert_tokens_to_ids(fragment_sequence)
        full_sequence = fragment_sequence + text_sequence
        input_ids.append(full_sequence)
        attention_mask.append([1 for _ in full_sequence])
        gap_ids.append([i for i in range(len(full_sequence)) if full_sequence[i] == gap_token_id])
        target_gaps.append(target_gap)

    input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_id))
    attention_mask = torch.tensor(pad_2d(attention_mask))
    gap_ids = torch.tensor(gap_ids)
    target_gaps = torch.tensor(target_gaps)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'gap_ids': gap_ids, 'target_gaps': target_gaps}
