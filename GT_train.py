import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import os
import yaml
from argparse import ArgumentParser
from json import dumps
from collections import OrderedDict

from tensorboardX import SummaryWriter
from tqdm import tqdm

from Models.BERT import BertForGappedText
from Models.Datasets import GT_Dataset, GT_collate_fn
from Models.util import CheckpointSaver, AverageMeter, get_logger, get_save_dir, get_num_data_samples

from pytorch_pretrained_bert.optimization import BertAdam


"""
Adapted from https://github.com/chrischute/squad.
"""


def main():
    parser = ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Experiment name.')
    parser.add_argument('--model',
                        type=str,
                        default='bert-base-uncased',
                        help='Pretrained model name or path.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./Experiments',
                        help='Base directory for saving information.')
    parser.add_argument('--data_folder',
                        type=str,
                        default='./Data/GT')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='')
    parser.add_argument('--seed',
                        type=int,
                        default=111)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        default=1e-5,
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use for training data loader.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=30)
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Evaluate model after processing this many training samples.')
    parser.add_argument('--eval_after_epoch',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to evaluate model at the end of every epoch.')
    parser.add_argument('--maximize_metric',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether metric should be maximized or minimized.')

    args = parser.parse_args()

    args.save_dir = get_save_dir(args.save_dir, args.name, training=True)

    with open(os.path.join(args.save_dir, 'args.yaml'), 'w', encoding='utf8') as file:
        yaml.dump(vars(args), file, default_flow_style=False, allow_unicode=True)

    log = get_logger(args.save_dir, args.name)
    tbx_writer = SummaryWriter(args.save_dir)
    device = 'cpu' if args.gpu_ids == '' else 'cuda'
    args.gpu_ids = [int(idx) for idx in args.gpu_ids]

    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids))

    num_data_samples, num_unique_data_epochs = get_num_data_samples(args.data_folder, args.num_epochs, log)
    num_optimization_steps = sum(num_data_samples) // args.batch_size // args.accumulation_steps
    log.info(f'Total number of optimization steps: {num_optimization_steps}')

    # Set random seed
    log.info(f'Using random seed {args.seed}.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info(f'Loading model {args.model}...')
    model = BertForGappedText.from_pretrained(args.model)

    with open(os.path.join(args.save_dir, 'config.yaml'), 'w', encoding='utf8') as file:
        yaml.dump(model.config, file, default_flow_style=False, allow_unicode=True)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    model = model.to(device)
    model.train()

    # Get saver
    saver = CheckpointSaver(args.save_dir,
                            max_checkpoints=args.max_checkpoints,
                            metric_name='Accuracy',
                            maximize_metric=args.maximize_metric,
                            log=log)

    # Get optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_optimization_steps)

    global_step = 0
    samples_processed = 0

    # Get dev data loader
    dev_data_file = os.path.join(args.data_folder, f'Dev.csv')
    log.info(f'Creating dev dataset from {dev_data_file}')
    dev_dataset = GT_Dataset(dev_data_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=GT_collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    for epoch in range(1, args.num_epochs + 1):
        train_data_file_num = ((epoch - 1) % num_unique_data_epochs) + 1
        train_data_file = os.path.join(args.data_folder, f'Epoch_{train_data_file_num}.csv')
        log.info(f'Creating training dataset from {train_data_file}')
        train_dataset = GT_Dataset(train_data_file)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       collate_fn=GT_collate_fn)

        log.info(f'Starting epoch {epoch}...')
        model.train()
        optimizer.zero_grad()
        loss_val = 0
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for step, batch in enumerate(train_loader, 1):
                batch = tuple(x.to(device) for x in batch)
                input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps = batch
                current_batch_size = input_ids.shape[0]

                loss = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             word_mask=word_mask,
                             gap_ids=gap_ids,
                             target_gaps=target_gaps)

                if len(args.gpu_ids) > 1:
                    loss = loss.mean()

                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps

                loss_val += loss.item()

                loss.backward()

                samples_processed += current_batch_size
                steps_till_eval -= current_batch_size
                progress_bar.update(current_batch_size)

                if step % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log info
                    current_lr = optimizer.get_lr()[0]
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val, lr=current_lr)
                    tbx_writer.add_scalar('train/Loss', loss_val, global_step)
                    tbx_writer.add_scalar('train/LR', current_lr, global_step)
                    loss_val = 0

                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps
                        evaluate_and_save(model=model,
                                          optimizer=optimizer,
                                          data_loader=dev_loader,
                                          device=device,
                                          tbx_writer=tbx_writer,
                                          log=log,
                                          global_step=global_step,
                                          saver=saver)

            if args.eval_after_epoch:
                evaluate_and_save(model=model,
                                  optimizer=optimizer,
                                  data_loader=dev_loader,
                                  device=device,
                                  tbx_writer=tbx_writer,
                                  log=log,
                                  global_step=global_step,
                                  saver=saver)


def evaluate_and_save(model, optimizer, data_loader, device, tbx_writer, log, global_step, saver):
    log.info('Evaluating...')
    results = evaluate(model, data_loader, device)
    log.info('Saving checkpoint at step {}...'.format(global_step))
    saver.save(step=global_step,
               model=model,
               optimizer=optimizer,
               metric_val=results['Accuracy'])

    # Log to console
    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                            for k, v in results.items())
    log.info('Dev {}'.format(results_str))

    # Log to TensorBoard
    log.info('Visualizing in TensorBoard...')
    for k, v in results.items():
        tbx_writer.add_scalar('dev/{}'.format(k), v, global_step)


def evaluate(model, data_loader, device):
    loss_meter = AverageMeter()

    model.eval()
    correct_preds = 0
    correct_avna = 0
    zero_preds = 0
    total_preds = 0
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch in data_loader:
            batch = tuple(x.to(device) for x in batch)
            input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps = batch
            current_batch_size = input_ids.shape[0]

            gap_scores = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               word_mask=word_mask,
                               gap_ids=gap_ids,
                               target_gaps=None)

            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            loss_meter.update(loss.item(), current_batch_size)

            preds = torch.argmax(gap_scores, dim=1)
            correct_preds += torch.sum(preds == target_gaps).item()
            correct_avna += torch.sum((preds > 0) == (target_gaps > 0)).item()
            zero_preds += torch.sum(preds == 0).item()
            total_preds += current_batch_size

            # Log info
            progress_bar.update(current_batch_size)
            progress_bar.set_postfix(Loss=loss_meter.avg)

    model.train()

    results_list = [('Loss', loss_meter.avg),
                    ('Accuracy', correct_preds / total_preds),
                    ('AvNA', correct_avna / total_preds),
                    ('NA_share', zero_preds / total_preds)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main()
