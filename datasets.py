import os
import gzip
import json
from itertools import islice
from abc import ABC, ABCMeta, abstractmethod

import torch
import torch.utils.data as data
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer('vocabs/roberta-large-vocab.json', 'vocabs/roberta-large-merges.txt')
gap_token_id = tokenizer.convert_tokens_to_ids(['<gap>'])[0]
pad_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]


def tokenize_first_sequence(fragment):
    return ['<s>'] + fragment.split() + ['</s>', '</s>']


def tokenize_second_sequence(text):
    return text.split() + ['</s>']


def pad_2d(array_2d, pad_value=0):
    row_lengths = [len(row) for row in array_2d]
    max_len = max(row_lengths)
    for i in range(len(array_2d)):
        array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

    return array_2d


class DatasetRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.task] = new_cls
        return new_cls

    @classmethod
    def get_dataset(mcs, task):
        return mcs.registry[task]


class BaseDataset(ABC, data.IterableDataset, metaclass=DatasetRegistry):
    task = None

    def __init__(self, data_file, size, local_rank, world_size=None):
        assert os.path.isfile(data_file), f'File f{data_file} not found'

        self.data_file = data_file
        self.size = size

        if local_rank == -1:
            self.start = 0
            self.step = 1
        else:
            self.start = local_rank
            self.step = world_size

    def __len__(self):
        return self.size

    def __iter__(self):
        file_iter = gzip.open(self.data_file, 'rt')
        islice_iter = islice(file_iter, self.start, None, self.step)
        processed_iter = map(self.process_line, islice_iter)
        return processed_iter

    @staticmethod
    @abstractmethod
    def process_line(line):
        pass

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        pass


class DatasetForGT(BaseDataset):
    task = 'GT'

    @staticmethod
    def process_line(line):
        data_sample = json.loads(line)
        text = data_sample['text']
        fragment = data_sample['fragment']
        target_gap = int(data_sample['target_gap'])

        fragment_sequence = tokenize_first_sequence(fragment)
        fragment_sequence = tokenizer.convert_tokens_to_ids(fragment_sequence)
        text_sequence = tokenize_second_sequence(text)
        text_sequence = tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = fragment_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        gap_ids = [i for i in range(len(input_ids)) if input_ids[i] == gap_token_id]

        return input_ids, attention_mask, gap_ids, target_gap

    @staticmethod
    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        gap_ids = []
        target_gaps = []

        for cur_input_ids, cur_attention_mask, cur_gap_ids, cur_target_gap in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            gap_ids.append(cur_gap_ids)
            target_gaps.append(cur_target_gap)

        input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_id))
        attention_mask = torch.tensor(pad_2d(attention_mask))
        gap_ids = torch.tensor(gap_ids)
        target_gaps = torch.tensor(target_gaps)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'gap_ids': gap_ids,
                'target_gaps': target_gaps}


class DatasetForQA(BaseDataset):
    task = 'QA'

    @staticmethod
    def process_line(line):
        data_sample = json.loads(line)
        text = data_sample['text']
        question = data_sample['question']
        answer_start = int(data_sample['answer_start'])
        answer_end = int(data_sample['answer_end'])

        question_sequence = tokenize_first_sequence(question)
        question_sequence = tokenizer.convert_tokens_to_ids(question_sequence)
        text_sequence = tokenize_second_sequence(text)
        text_sequence = tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = question_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        answer_start = 0 if answer_start == -1 else answer_start + len(question_sequence)
        answer_end = 0 if answer_end == -1 else answer_end + len(question_sequence)

        return input_ids, attention_mask, answer_start, answer_end

    @staticmethod
    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        answer_start = []
        answer_end = []

        for cur_input_ids, cur_attention_mask, cur_answer_start, cur_answer_end in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            answer_start.append(cur_answer_start)
            answer_end.append(cur_answer_end)

        input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_id))
        attention_mask = torch.tensor(pad_2d(attention_mask))
        answer_start = torch.tensor(answer_start)
        answer_end = torch.tensor(answer_end)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'answer_start': answer_start,
                'answer_end': answer_end}
