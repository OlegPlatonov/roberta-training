import numpy as np
import pandas as pd
import spacy
import re
import os
import random
import string
import argparse
from pathos.multiprocessing import Pool
from functools import partial
from pytorch_transformers import BertTokenizer, RobertaTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--model_type',
                    default='bert-base-uncased',
                    choices=['bert-base-uncased', 'roberta'],
                    help='Tokenizer from which model to use.')
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of sub-processes to use for data processing.')
parser.add_argument('--data_dir',
                    type=str,
                    default='./Wikipedia/Extracted')
parser.add_argument('--save_dir',
                    type=str,
                    default='./GT/Text')
parser.add_argument('--seed',
                    type=int,
                    default=111,
                    help='Random seed for reproducibility.')
args = parser.parse_args()


nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])

if args.model_type == 'bert-base-uncased':
    tokenizer = BertTokenizer('./../models/vocabs/bert-base-uncased-vocab.txt',
                              additional_special_tokens=['[GAP]'],
                              do_basic_tokenize=True)

    GAP_TOKEN = '[GAP]'
    UNK_TOKEN = '[UNK]'
    MAX_PAIR_LENGTH = 509
    LOWER = True

elif args.model_type == 'roberta':
    tokenizer = RobertaTokenizer('./../models/vocabs/roberta-large-vocab.json',
                                 './../models/vocabs/roberta-large-merges.txt',
                                 additional_special_tokens=['<gap>'],
                                 do_basic_tokenize=False)

    GAP_TOKEN = '<gap>'
    UNK_TOKEN = '<unk>'
    MAX_PAIR_LENGTH = 508
    LOWER = False

MIN_ARTICLE_LENGTH = 10


def fix_roberta_punctuation(text):
    for i in range(len(text)):
        if text[i] == 'Ġ.':
            text[i] = '.'
        elif text[i] == 'Ġ,':
            text[i] = ','

    return text


def get_raw_texts(file):
    texts = []

    with open(file, encoding='utf8') as file:
        for line in file.readlines():
            line = line.strip()
            if LOWER:
                line = line.lower()
            if line.startswith('<doc'):   # begin new document
                current_text = []
                skip_next = True
            elif skip_next:               # skip title
                skip_next = False
            elif line == '</doc>':        # end document
                texts.append(current_text)
            else:
                if line != '':
                    line = process_raw_line(line)
                    if line != '':
                        current_text.append(line)

    return texts


def process_raw_line(line):
    line = re.sub('<ref.*/ref>', '', line, flags=re.DOTALL)
    line = re.sub('<ref.*">', '', line, flags=re.DOTALL)
    line = re.sub('<blockquote>', '', line)
    line = re.sub('</blockquote>', '', line)
    line = re.sub('<sub.*/sub>', '', line, flags=re.DOTALL)
    line = re.sub('<sup.*/sup>', '', line, flags=re.DOTALL)
    line = re.sub('<li.*/li>', '', line, flags=re.DOTALL)
    line = re.sub('<.*>', '', line, flags=re.DOTALL)
    line = re.sub('\(.*?\)', '', line, flags=re.DOTALL)
    line = re.sub('\[.*?\]', '', line, flags=re.DOTALL)
    line = re.sub('see also:', '', line)
    line = re.sub('[\s\s+]', ' ', line)
    line = re.sub('[,;:] [,;:]', ',', line)
    line = re.sub('[,.:] [,.:]', '.', line)
    line = re.sub(',,', ',', line)
    line = ' '.join(word for word in line.split() if not word.isspace())
    return line


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
    folders = [os.path.join(load_path, folder) for folder in os.listdir(load_path)]

    folder_processer = partial(process_folder,
                               save_path=save_path,
                               text_size=text_size,
                               num_gaps=num_gaps,
                               min_space=min_space,
                               num_random_sent=num_random_sent,
                               target_len=target_len,
                               min_len=min_len,
                               max_len=max_len)

    with Pool(num_workers) as pool:
        pool.map(folder_processer, folders)


def process_folder(folder, save_path, text_size, num_gaps, min_space, num_random_sent,
                   target_len, min_len, max_len):

    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    folder = os.path.basename(os.path.normpath(folder))

    for file in files:
        process_file(file=file, folder=folder, save_path=save_path, chunk_size=text_size,
                     num_gaps=num_gaps, min_space=min_space, num_random_sent=num_random_sent,
                     target_len=target_len, min_len=min_len, max_len=max_len)

    print(f'Finished processing data from {folder}.')


def process_file(file, folder, save_path, chunk_size, num_gaps, min_space, num_random_sent,
                 target_len, min_len, max_len):

    texts = get_raw_texts(file)
    file = os.path.basename(os.path.normpath(file))

    data = dict([('segment_1', []), ('segment_2', [])])

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
                sent = tokenizer.tokenize('A ' + sent.text)[1:]   # Includes leading whitespace for correct RoBERTa tokenization.
                if args.model_type == 'roberta':
                    sent = fix_roberta_punctuation(sent)
                if not phrase_is_unk(sent):
                    phrases.append(sent)

        if len(phrases) >= MIN_ARTICLE_LENGTH:
            for segment_pair in process_text(phrases,
                                             chunk_size=chunk_size,
                                             num_gaps=num_gaps,
                                             min_space=min_space,
                                             num_random_sent=num_random_sent):

                data['segment_1'].append(segment_pair[0])
                data['segment_2'].append(segment_pair[1])

    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, f'{folder}_{file}.csv'), index=False)


def process_text(text, chunk_size, num_gaps, min_space, num_random_sent, max_pair_len=MAX_PAIR_LENGTH):
    examples = [[]]
    current_length = 0
    for sent in text:
        if current_length + len(sent) <= MAX_PAIR_LENGTH:
            examples[-1].append(sent)
            current_length += len(sent)
        else:
            examples.append([sent])
            current_length = len(sent)

    for example in examples:
        if len(example) >= 2 and sum(len(sent) for sent in example) <= MAX_PAIR_LENGTH:
            example = [' '.join(sent) for sent in example]
            split_idx = (len(example) + 1) // 2
            segment_1 = ' '.join(example[:split_idx])
            segment_2 = ' '.join(example[split_idx:])
            yield segment_1, segment_2


def phrase_is_unk(phrase):
    for token in phrase:
        if token != UNK_TOKEN and token not in string.punctuation:
            return False

    return True


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('Starting data preprocessing for Sentence Order Prediction task...')
    process_all(num_workers=args.num_workers)
