import random
import os
import gzip
import json
from string import punctuation
from itertools import islice
from abc import ABC, ABCMeta, abstractmethod

import torch
from torch.utils.data import IterableDataset
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


class BaseDataset(ABC, IterableDataset, metaclass=DatasetRegistry):
    task = None
    tokenizer = RobertaTokenizer(vocab_file='vocabs/roberta-vocab.json', merges_file='vocabs/roberta-merges.txt')
    pad_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    mask_token_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
    gap_token_id = tokenizer.convert_tokens_to_ids(['<gap>'])[0]

    def __init__(self, data_file, data_size, local_rank, world_size=None):
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f'{data_file} does not exist or is a directory.')

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

    @abstractmethod
    def process_line(self, line):
        pass

    @classmethod
    @abstractmethod
    def collate_fn(cls, batch):
        pass


class DatasetForMLM(BaseDataset):
    task = 'MLM'

    def __init__(self, data_file, data_size, local_rank, world_size=None, mask_proportion=0.15):
        super().__init__(data_file, data_size, local_rank, world_size)
        self.mask_proportion = mask_proportion

    def process_line(self, line):
        data_sample = json.loads(line)
        text = data_sample['text']
        text_sequence = self.tokenize_first_segment(text)
        words_to_mask = self.find_words_to_mask(text_sequence)

        text_sequence = self.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids, mask_ids, mask_targets = self.mask_words(text_sequence, words_to_mask)
        attention_mask = [1 for _ in input_ids]

        return input_ids, attention_mask, mask_ids, mask_targets

    @classmethod
    def collate_fn(cls, batch):
        input_ids = []
        attention_mask = []
        mask_ids = []
        mask_targets = []

        for cur_input_ids, cur_attention_mask, cur_mask_ids, cur_mask_targets in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            mask_ids.append(cur_mask_ids)
            mask_targets.append(cur_mask_targets)

        input_ids = torch.tensor(cls.pad_2d(input_ids, pad_value=cls.pad_token_id))
        attention_mask = torch.tensor(cls.pad_2d(attention_mask))
        mask_ids = torch.tensor([[row, col] for row, line in enumerate(mask_ids) for col in line])
        mask_targets = torch.tensor([token_idx for line in mask_targets for token_idx in line])

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'mask_ids': mask_ids,
                'mask_targets': mask_targets}

    def find_words_to_mask(self, text_sequence):
        """Whole word masking."""
        word_ids = []
        for i, token in enumerate(text_sequence):
            if token.startswith('<') and token.endswith('>'):
                continue
            elif token.startswith('Ġ') or token in punctuation:
                word_ids.append([i, 1])
            elif word_ids:
                word_ids[-1][1] += 1

        num_to_mask = int(len(word_ids) * self.mask_proportion)
        word_ids_to_mask = random.sample(word_ids, num_to_mask)
        word_ids_to_mask.sort()

        return word_ids_to_mask

    def mask_words(self, text_sequence, words_to_mask):
        mask_ids = []
        mask_targets = []
        for start_idx, length in words_to_mask:
            for i in range(length):
                current_idx = start_idx + i
                mask_ids.append(current_idx)
                mask_targets.append(text_sequence[current_idx])
                text_sequence[current_idx] = self.mask_token_id

        return text_sequence, mask_ids, mask_targets


class DatasetForGT(BaseDataset):
    task = 'GT'

    def process_line(self, line):
        data_sample = json.loads(line)
        text = data_sample['text']
        fragment = data_sample['fragment']
        target_gap = int(data_sample['target_gap'])

        fragment_sequence = self.tokenize_first_segment(fragment)
        fragment_sequence = self.tokenizer.convert_tokens_to_ids(fragment_sequence)
        text_sequence = self.tokenize_second_segment(text)
        text_sequence = self.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = fragment_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        gap_ids = [i for i in range(len(input_ids)) if input_ids[i] == self.gap_token_id]

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

    def process_line(self, line):
        data_sample = json.loads(line)
        text = data_sample['text']
        question = data_sample['question']
        answer_start = int(data_sample['answer_start'])
        answer_end = int(data_sample['answer_end'])

        question_sequence = self.tokenize_first_segment(question)
        question_sequence = self.tokenizer.convert_tokens_to_ids(question_sequence)
        text_sequence = self.tokenize_second_segment(text)
        text_sequence = self.tokenizer.convert_tokens_to_ids(text_sequence)
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
