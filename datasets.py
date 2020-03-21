import os
import gzip
import json
from itertools import islice
from abc import ABC, ABCMeta, abstractmethod

import torch
import torch.utils.data as data
from transformers import RobertaTokenizer


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
    tokenizer = RobertaTokenizer('vocabs/roberta-large-vocab.json', 'vocabs/roberta-large-merges.txt')
    gap_token_id = tokenizer.convert_tokens_to_ids(['<gap>'])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]

    def __init__(self, data_file, data_size, local_rank, world_size=None):
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f'{data_file} does not exist or is a directory')

        self.data_file = data_file
        self.size = data_size

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
    def tokenize_first_segment(segment):
        return ['<s>'] + segment.split() + ['</s>']

    @staticmethod
    def tokenize_second_segment(segment):
        return ['</s>'] + segment.split() + ['</s>']

    @staticmethod
    def pad_2d(array_2d, pad_value=0):
        row_lengths = [len(row) for row in array_2d]
        max_len = max(row_lengths)
        for i in range(len(array_2d)):
            array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

        return array_2d

    @classmethod
    @abstractmethod
    def process_line(cls, line):
        pass

    @classmethod
    @abstractmethod
    def collate_fn(cls, batch):
        pass


class DatasetForGT(BaseDataset):
    task = 'GT'

    @classmethod
    def process_line(cls, line):
        data_sample = json.loads(line)
        text = data_sample['text']
        fragment = data_sample['fragment']
        target_gap = int(data_sample['target_gap'])

        fragment_sequence = cls.tokenize_first_segment(fragment)
        fragment_sequence = cls.tokenizer.convert_tokens_to_ids(fragment_sequence)
        text_sequence = cls.tokenize_second_segment(text)
        text_sequence = cls.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = fragment_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        gap_ids = [i for i in range(len(input_ids)) if input_ids[i] == cls.gap_token_id]

        return input_ids, attention_mask, gap_ids, target_gap

    @classmethod
    def collate_fn(cls, batch):
        input_ids = []
        attention_mask = []
        gap_ids = []
        target_gaps = []

        for cur_input_ids, cur_attention_mask, cur_gap_ids, cur_target_gap in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            gap_ids.append(cur_gap_ids)
            target_gaps.append(cur_target_gap)

        input_ids = torch.tensor(cls.pad_2d(input_ids, pad_value=cls.pad_token_id))
        attention_mask = torch.tensor(cls.pad_2d(attention_mask))
        gap_ids = torch.tensor(gap_ids)
        target_gaps = torch.tensor(target_gaps)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'gap_ids': gap_ids,
                'target_gaps': target_gaps}


class DatasetForQA(BaseDataset):
    task = 'QA'

    @classmethod
    def process_line(cls, line):
        data_sample = json.loads(line)
        text = data_sample['text']
        question = data_sample['question']
        answer_start = int(data_sample['answer_start'])
        answer_end = int(data_sample['answer_end'])

        question_sequence = cls.tokenize_first_segment(question)
        question_sequence = cls.tokenizer.convert_tokens_to_ids(question_sequence)
        text_sequence = cls.tokenize_second_segment(text)
        text_sequence = cls.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = question_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        answer_start = 0 if answer_start == -1 else answer_start + len(question_sequence) + 1
        answer_end = 0 if answer_end == -1 else answer_end + len(question_sequence) + 1

        return input_ids, attention_mask, answer_start, answer_end

    @classmethod
    def collate_fn(cls, batch):
        input_ids = []
        attention_mask = []
        answer_start = []
        answer_end = []

        for cur_input_ids, cur_attention_mask, cur_answer_start, cur_answer_end in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            answer_start.append(cur_answer_start)
            answer_end.append(cur_answer_end)

        input_ids = torch.tensor(cls.pad_2d(input_ids, pad_value=cls.pad_token_id))
        attention_mask = torch.tensor(cls.pad_2d(attention_mask))
        answer_start = torch.tensor(answer_start)
        answer_end = torch.tensor(answer_end)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'answer_start': answer_start,
                'answer_end': answer_end}
