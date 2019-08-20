import pandas as pd
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer


special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', '[GAP]']
tokenizer = BertTokenizer.from_pretrained('./models/vocabs/bert-base-uncased-vocab.txt',
                                          do_basic_tokenize=False,
                                          never_split=special_tokens)

GAP_TOKEN_ID = tokenizer.vocab['[GAP]']


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
    input_ids = []
    token_type_ids = []
    attention_mask = []
    word_mask = []
    gap_ids = []
    target_gaps = []

    for text, fragment, target_gap in batch:
        text_sequence = tokenizer.convert_tokens_to_ids(['[CLS]'] + text.split() + ['[SEP]'])
        fragment_sequence = tokenizer.convert_tokens_to_ids(fragment.split() + ['[SEP]'])
        full_sequence = text_sequence + fragment_sequence
        input_ids.append(full_sequence)
        token_type_ids.append([0 for _ in range(len(text_sequence))] + [1 for _ in range(len(fragment_sequence))])
        attention_mask.append([1 for _ in full_sequence])
        word_mask.append([1 if idx > 900 else 0 for idx in full_sequence])
        gap_ids.append([i for i in range(len(text_sequence)) if text_sequence[i] == GAP_TOKEN_ID])
        target_gaps.append(target_gap)

    input_ids = torch.tensor(pad_2d(input_ids))
    token_type_ids = torch.tensor(pad_2d(token_type_ids, pad_value=1))
    attention_mask = torch.tensor(pad_2d(attention_mask))
    word_mask = torch.tensor(pad_2d(word_mask))
    gap_ids = torch.tensor(gap_ids)
    target_gaps = torch.tensor(target_gaps)

    return input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps
