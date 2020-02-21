import pandas as pd
import torch
import torch.utils.data as data
from transformers import BertTokenizer, RobertaTokenizer

bert_tokenizer = BertTokenizer('./models/vocabs/bert-base-uncased-vocab.txt',
                               additional_special_tokens=['[GAP]'],
                               do_basic_tokenize=False)

roberta_tokenizer = RobertaTokenizer('./models/vocabs/roberta-large-vocab.json',
                                     './models/vocabs/roberta-large-merges.txt',
                                     additional_special_tokens=['<gap>'],
                                     do_basic_tokenize=False)

tokenizers = {
    'bert-base-uncased': bert_tokenizer,
    'roberta': roberta_tokenizer
}

special_tokens = {
    'bert-base-uncased': ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[GAP]'],
    'roberta': ['<s>', '<pad>', '</s>', '<unk>', '<mask>', '<gap>']
}

special_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(special_tokens['bert-base-uncased']),
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(special_tokens['roberta'])
}

gap_tokens = {
    'bert-base-uncased': '[GAP]',
    'roberta': '<gap>'
}

pad_tokens = {
    'bert-base-uncased': '[PAD]',
    'roberta': '<pad>'
}

gap_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(['[GAP]'])[0],
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(['<gap>'])[0]
}

pad_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(['[PAD]'])[0],
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(['<pad>'])[0]
}

fragment_transforms = {
    'bert-base-uncased': lambda fragment: ['[CLS]'] + fragment.split() + ['[SEP]'],
    'roberta': lambda fragment: ['<s>'] + fragment.split() + ['</s>', '</s>']
}

text_transforms = {
    'bert-base-uncased': lambda text: text.split() + ['[SEP]'],
    'roberta': lambda text: text.split() + ['</s>']
}


class GT_Dataset(data.Dataset):
    def __init__(self, data_path, num_fragments=8):
        super(GT_Dataset, self).__init__()
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


def GT_collate_fn(batch):
    model_type = 'roberta'

    input_ids = []
    token_type_ids = []
    attention_mask = []
    word_mask = []
    gap_ids = []
    target_gaps = []

    for text, fragment, target_gap in batch:
        text_sequence = text_transforms[model_type](text)
        text_sequence = tokenizers[model_type].convert_tokens_to_ids(text_sequence)
        fragment_sequence = fragment_transforms[model_type](fragment)
        fragment_sequence = tokenizers[model_type].convert_tokens_to_ids(fragment_sequence)
        full_sequence = fragment_sequence + text_sequence
        input_ids.append(full_sequence)
        token_type_ids.append([0 for _ in range(len(fragment_sequence))] + [1 for _ in range(len(text_sequence))])
        attention_mask.append([1 for _ in full_sequence])
        word_mask.append([1 if idx not in special_token_ids[model_type] else 0 for idx in full_sequence])
        gap_ids.append([i for i in range(len(full_sequence)) if full_sequence[i] == gap_token_ids[model_type]])
        target_gaps.append(target_gap)

    input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_ids[model_type]))
    token_type_ids = torch.tensor(pad_2d(token_type_ids, pad_value=1))
    attention_mask = torch.tensor(pad_2d(attention_mask))
    word_mask = torch.tensor(pad_2d(word_mask))
    gap_ids = torch.tensor(gap_ids)
    target_gaps = torch.tensor(target_gaps)

    return input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps

