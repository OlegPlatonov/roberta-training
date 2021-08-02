import random
import os
import string
import re
import argparse

import numpy as np
import pandas as pd
import spacy

from multiprocessing import Pool
from functools import partial
from bs4 import BeautifulSoup
from unicodedata import normalize
from transformers import RobertaTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of processes to use for data preprocessing.')
parser.add_argument('--data_dir',
                    type=str,
                    default='wikipedia/extracted',
                    help='Directory with data to preprocess.')
parser.add_argument('--save_dir',
                    type=str,
                    default='GT/text',
                    help='Directory for saving preprocessed data.')
parser.add_argument('--seed',
                    type=int,
                    default=111,
                    help='Random seed.')
args = parser.parse_args()

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])

tokenizer = RobertaTokenizer(vocab_file='../vocabs/roberta-vocab.json', merges_file='../vocabs/roberta-merges.txt',
                             additional_special_tokens=['<gap>'])

GAP_TOKEN = '<gap>'
UNK_TOKEN = '<unk>'
MAX_PAIR_LENGTH = 508
LOWER = False

# Sentences that are too long will be split on these tokens
split_tokens_1 = {'.', '?', '!', ',', ':', ';', 'that', 'which', 'who', 'whom', 'whose', 'when', 'where', 'of', 'for',
                  'from', 'was', 'is', 'are', 'were', 'and', 'or', 'but', 'if', 'whether', 'while', 'because', 'though',
                  'as', 'to'}
split_tokens_2 = {'what', 'instead', 'have', 'has', 'had', 'will', 'there', 'those', 'this', 'these', 'then', 'so',
                  'such', 'by'}


def get_raw_texts(file):
    texts = []

    with open(file, encoding='utf8') as file:
        for line in file.readlines():
            line = line.strip()
            if LOWER:
                line = line.lower()
            if line.startswith('<doc'):  # begin new document
                current_text = []
                skip_next = True
            elif skip_next:  # skip title
                skip_next = False
            elif line == '</doc>':  # end document
                texts.append(current_text)
            else:
                if line != '':
                    line = process_raw_text(line)
                    if line != '':
                        current_text.append(line)

    return texts


def process_raw_text(text):
    text = BeautifulSoup(text).get_text()
    text = normalize('NFKD', text)
    text = re.sub('\(.*?\)', '', text, flags=re.DOTALL)
    text = re.sub('\[.*?\]', '', text, flags=re.DOTALL)
    text = fix_spaces_and_punctuation(text)
    return text


def fix_spaces_and_punctuation(text):
    text = re.sub('[\s\s+]', ' ', text)
    text = re.sub('[,;:] [,;:]', ',', text)
    text = re.sub('[,.:] [,.:]', '.', text)
    text = re.sub(',,', ',', text)
    text = ' '.join(word for word in text.split() if not word.isspace())
    return text


def fix_roberta_punctuation(text):
    for i in range(len(text)):
        if text[i] == 'Ġ.':
            text[i] = '.'
        elif text[i] == 'Ġ,':
            text[i] = ','

    return text


def process_all(num_workers,
                load_path=args.data_dir,
                save_path=args.save_dir,
                text_size=20,
                num_gaps=5,
                min_space=2,
                num_random_sent=3,
                target_len=25,
                min_len=15,
                max_len=40):
    directories = [os.path.join(load_path, directory) for directory in os.listdir(load_path)]

    directory_processer = partial(process_directory,
                                  save_path=save_path,
                                  text_size=text_size,
                                  num_gaps=num_gaps,
                                  min_space=min_space,
                                  num_random_sent=num_random_sent,
                                  target_len=target_len,
                                  min_len=min_len,
                                  max_len=max_len)

    with Pool(num_workers) as pool:
        pool.map(directory_processer, directories)


def process_directory(directory, save_path, text_size, num_gaps, min_space, num_random_sent,
                      target_len, min_len, max_len):
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    directory = os.path.basename(os.path.normpath(directory))

    for file in files:
        process_file(file=file, directory=directory, save_path=save_path, chunk_size=text_size,
                     num_gaps=num_gaps, min_space=min_space, num_random_sent=num_random_sent,
                     target_len=target_len, min_len=min_len, max_len=max_len)

    print(f'Finished preprocessing data from {directory}.')


def process_file(file, directory, save_path, chunk_size, num_gaps, min_space, num_random_sent,
                 target_len, min_len, max_len):
    texts = get_raw_texts(file)
    file = os.path.basename(os.path.normpath(file))

    data = dict(
        [('text', [])] +
        [(f'fragment_{i}', []) for i in range(1, num_gaps + num_random_sent + 1)] +
        [(f'target_gap_{i}', []) for i in range(1, num_gaps + num_random_sent + 1)]
    )

    # Concatenate all texts for faster spaCy processing.
    text_lengths = [len(text) for text in texts]
    text_ends = [0]
    for length in text_lengths:
        text_ends.append(text_ends[-1] + length)

    paras = []
    for text in texts:
        paras += text
    paras = list(nlp.pipe(paras, batch_size=500))

    texts = []
    for i in range(1, len(text_ends)):
        texts.append(paras[text_ends[i - 1]:text_ends[i]])

    for text in texts:
        phrases = []
        for para in text:
            for sent in para.sents:
                if len(sent) <= max_len:
                    sent = tokenizer.tokenize(sent.text, add_prefix_space=True)
                    sent = fix_roberta_punctuation(sent)
                    if not phrase_is_unk(sent):
                        phrases.append(' '.join(sent))
                else:
                    split = split_sent(sent, target_len=target_len, min_len=min_len, max_len=max_len)
                    split = [tokenizer.tokenize(phrase.text, add_prefix_space=True) for phrase in split]
                    split = [fix_roberta_punctuation(phrase) for phrase in split]
                    split = [' '.join(phrase) for phrase in split if not phrase_is_unk(phrase)]
                    phrases += split

        if len(phrases) >= 2 * chunk_size:
            for gapped_text in process_text(phrases,
                                            chunk_size=chunk_size,
                                            num_gaps=num_gaps,
                                            min_space=min_space,
                                            num_random_sent=num_random_sent):

                text, fragments, target_gaps = gapped_text
                data['text'].append(text)
                for i in range(len(fragments)):
                    data[f'fragment_{i + 1}'].append(fragments[i])
                    data[f'target_gap_{i + 1}'].append(target_gaps[i])

    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, f'{directory}_{file}.csv'), index=False)


def process_text(text, chunk_size, num_gaps, min_space, num_random_sent, max_pair_len=MAX_PAIR_LENGTH):
    chunk_bounds = list(range(0, len(text), chunk_size))
    for i in range(1, len(chunk_bounds)):
        chunk = text[chunk_bounds[i - 1]:chunk_bounds[i]]
        if len(chunk) < chunk_size - 4:
            continue

        gap_sent_ids = get_random_gap_ids(chunk_size=chunk_size, num_gaps=num_gaps, min_space=min_space)

        fragments = []
        gapped_chunk = [phrase for phrase in chunk]
        for idx in gap_sent_ids:
            fragments.append(chunk[idx])
            gapped_chunk[idx] = GAP_TOKEN

        if max_pair_len is not None:
            text_length = len(' '.join(gapped_chunk).split())
            fragment_lengths = [len(fragment.split()) for fragment in fragments]
            if text_length + max(fragment_lengths) > max_pair_len:
                x = 1
                while len(chunk) - x >= chunk_size - 4:
                    gap_sent_ids = get_random_gap_ids(chunk_size=chunk_size - x, num_gaps=num_gaps, min_space=min_space)
                    fragments = []
                    gapped_chunk = [phrase for phrase in chunk[:-x]]
                    for idx in gap_sent_ids:
                        fragments.append(chunk[idx])
                        gapped_chunk[idx] = GAP_TOKEN
                    text_length = len(' '.join(gapped_chunk).split())
                    fragment_lengths = [len(fragment.split()) for fragment in fragments]
                    if text_length + max(fragment_lengths) <= max_pair_len:
                        break
                    x += 1

            if text_length + max(fragment_lengths) > max_pair_len:
                continue

        target_gaps = list(range(1, num_gaps + 1))

        nonchunk_sent_ids = get_random_nonchunk_sent_ids(text_len=len(text),
                                                         chunk_size=chunk_size,
                                                         chunk_start=chunk_bounds[i - 1],
                                                         num_random_sent=num_random_sent)

        for idx in nonchunk_sent_ids:
            fragment = text[idx]

            if max_pair_len is not None:
                fragment_length = len(fragment.split())
                while text_length + fragment_length > max_pair_len and fragment_length >= 5:
                    fragment = fragment.split()
                    fragment = fragment[:-3]
                    fragment_length = len(fragment)
                    fragment = ' '.join(fragment)

            fragments.append(fragment)
            target_gaps.append(0)

        if max_pair_len is not None:
            fragment_lengths = [len(fragment.split()) for fragment in fragments[-num_random_sent:]]
            if text_length + max(fragment_lengths) > max_pair_len:
                continue

        gapped_chunk = ' '.join(gapped_chunk)

        yield gapped_chunk, fragments, target_gaps


def get_all_split_idx(sent):
    split_idx = [0]
    for i, token in enumerate(sent):
        if token.text in split_tokens_1 and split_idx[-1] != i + 1:
            split_idx.append(i + 1)
        elif token.text in split_tokens_2 and split_idx[-1] != i:
            split_idx.append(i)

    if split_idx[-1] != len(sent):
        split_idx.append(len(sent))

    return split_idx


def select_split_idx(all_split_idx, target_len):
    split_idx = [0]
    start = 1
    fragment_start = 0
    next_target = target_len
    while next_target < all_split_idx[-1]:
        for i in range(start, len(all_split_idx)):
            if all_split_idx[i] >= next_target:
                if all_split_idx[i] - next_target < next_target - all_split_idx[i - 1] or \
                        all_split_idx[i - 1] == split_idx[-1]:
                    best_i = i
                else:
                    best_i = i - 1
                split_idx.append(all_split_idx[best_i])
                start = best_i + 1
                fragment_start = all_split_idx[best_i]
                next_target = fragment_start + target_len
                break

    if split_idx[-1] != all_split_idx[-1]:
        split_idx.append(all_split_idx[-1])

    return split_idx


def balance_split_idx(split_idx, sent, target_len, min_len, max_len):
    not_start = {'.', '?', '!', ',', ':', ';', '-', '\'s'}

    balanced_split_idx = [0]
    for i in range(1, len(split_idx)):
        length = split_idx[i] - balanced_split_idx[-1]

        if length < min_len:
            if len(balanced_split_idx) >= 2 and balanced_split_idx[-1] - balanced_split_idx[-2] + length <= max_len:
                balanced_split_idx.pop(-1)
            elif len(split_idx) > i + 1 and split_idx[i + 1] - split_idx[i] + length <= max_len:
                continue

        elif length > max_len:
            while length > max_len:
                balanced_split_idx.append(balanced_split_idx[-1] + target_len)
                length -= target_len
            if length == 0:
                continue
            elif length < min_len:
                balanced_split_idx[-1] -= 8

        balanced_split_idx.append(split_idx[i])

    if balanced_split_idx[-1] != split_idx[-1]:
        balanced_split_idx.append(split_idx[-1])

    if len(balanced_split_idx) >= 3 and balanced_split_idx[-1] - balanced_split_idx[-2] < min_len:
        balanced_split_idx[-2] -= 5

    for i in range(1, len(balanced_split_idx) - 1):
        if sent[balanced_split_idx[i]].text in not_start:
            balanced_split_idx[i] += 1

    return balanced_split_idx


def split_sent(sent, target_len, min_len, max_len):
    split_idx = get_all_split_idx(sent)
    split_idx = select_split_idx(split_idx, target_len=target_len)
    split_idx = balance_split_idx(split_idx, sent=sent, target_len=target_len, min_len=min_len, max_len=max_len)

    split = []
    for i in range(1, len(split_idx)):
        split.append(sent[split_idx[i - 1]:split_idx[i]])

    return split


def get_random_gap_ids(chunk_size, num_gaps, min_space, always_use_last=False):
    if always_use_last:
        ids = np.array(sorted(random.sample(range(chunk_size - min_space * num_gaps - 1), k=num_gaps - 1)))
        ids += np.arange(1, num_gaps) * min_space
        ids = np.concatenate([ids, np.array([chunk_size - 1])], axis=0)

    else:
        ids = np.array(sorted(random.sample(range(chunk_size - min_space * num_gaps - 1), k=num_gaps)))
        ids += np.arange(1, num_gaps + 1) * min_space

    return ids


def get_random_nonchunk_sent_ids(text_len, chunk_size, chunk_start, num_random_sent):
    range_1 = range(max([0, chunk_start - 6]), max([0, chunk_start - 1]))
    range_2 = range(min([text_len, chunk_start + chunk_size + 1]), min([text_len, chunk_start + chunk_size + 6]))
    available_ids = list(range_1) + list(range_2)
    ids = np.array(sorted(random.sample(available_ids, k=num_random_sent)))

    return ids


def phrase_is_unk(phrase):
    for token in phrase:
        if token != UNK_TOKEN and token not in string.punctuation:
            return False

    return True


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('Starting data preprocessing for Gapped Text task...')
    process_all(num_workers=args.num_workers)
