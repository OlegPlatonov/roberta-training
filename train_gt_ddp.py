import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import os
import yaml
from argparse import ArgumentParser
from json import dumps
from collections import OrderedDict

from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.bert import BertForGappedText
from models.optimizer import BertAdam
from utils.datasets_gt import GT_Dataset, GT_collate_fn
from utils.utils_gt import CheckpointSaver, AverageMeter, get_logger, get_save_dir, get_num_data_samples

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel



"""
Adapted from https://github.com/chrischute/squad and https://github.com/huggingface/pytorch-pretrained-BERT.
"""


def get_args():
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
                        default='./experiments',
                        help='Base directory for saving information.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/GT')
    parser.add_argument('--seed',
                        type=int,
                        default=111)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='This is the number of training samples processed together by one GPU. '
                             'The effective batch size (number of training samples processed per one '
                             'optimization step) is equal to batch_size * num_gpus * accumulation_steps.')
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
                             'E.g., 0.1 = 10% of training.')
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help='Local rank for distributed training.')

    args = parser.parse_args()

    return args


def main(args, log):
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    if args.local_rank == 0:
        with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

    tb_writer = SummaryWriter(args.save_dir) if args.local_rank == 0 else None

    device = torch.device('cuda', args.local_rank)
    log.warning(f'Using GPU {args.local_rank}.')

    world_size = torch.distributed.get_world_size()
    log.info(f'Total number of GPUs used: {world_size}.')
    log.info(f'Effective batch size: {args.batch_size * world_size * args.accumulation_steps}.')
    args.eval_steps = args.eval_steps // world_size

    # Set random seed
    log.info(f'Using random seed {args.seed}.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    if args.local_rank != 0:
        torch.distributed.barrier()

    log.info(f'Loading model {args.model}...')
    model = BertForGappedText.from_pretrained(args.model)

    if args.local_rank == 0:
        torch.distributed.barrier()
        with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as file:
            yaml.dump(model.config.__dict__, file)

    model.to(device)

    # Get saver
    saver = CheckpointSaver(args.save_dir,
                            max_checkpoints=args.max_checkpoints,
                            metric_name='Accuracy',
                            maximize_metric=True,
                            log=log)

    # Get train data loader
    train_data_file = os.path.join(args.data_dir, f'Epoch_1.csv')
    log.info(f'Creating training dataset from {train_data_file}...')
    train_dataset = GT_Dataset(train_data_file)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   sampler=train_sampler,
                                   num_workers=args.num_workers,
                                   collate_fn=GT_collate_fn)

    # Get dev data loader
    dev_data_file = os.path.join(args.data_dir, f'Dev.csv')
    log.info(f'Creating dev dataset from {dev_data_file}...')
    dev_dataset = GT_Dataset(dev_data_file)
    dev_sampler = DistributedSampler(dev_dataset)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 sampler=dev_sampler,
                                 num_workers=2,
                                 collate_fn=GT_collate_fn)

    num_optimization_steps = len(train_loader) // args.accumulation_steps * args.num_epochs
    log.info(f'Total number of optimization steps: {num_optimization_steps}.')

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

    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)

    global_step = 0
    samples_processed = 0

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    for epoch in range(1, args.num_epochs + 1):
        torch.distributed.barrier()
        log.info(f'Starting epoch {epoch}...')
        model.train()
        optimizer.zero_grad()
        loss_val = 0
        with torch.enable_grad(), tqdm(total=len(train_loader) * args.batch_size, disable=args.local_rank != 0) as progress_bar:
            for step, batch in enumerate(train_loader, 1):
                batch = tuple(x.to(device) for x in batch)
                input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps = batch
                current_batch_size = input_ids.shape[0]

                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                word_mask=word_mask,
                                gap_ids=gap_ids,
                                target_gaps=target_gaps)

                loss = outputs[0]

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
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val, step=global_step, lr=current_lr)
                    if args.local_rank == 0:
                        tb_writer.add_scalar('train/Loss', loss_val, global_step)
                        tb_writer.add_scalar('train/LR', current_lr, global_step)
                    loss_val = 0

                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps
                        evaluate_and_save(model=model,
                                          optimizer=optimizer,
                                          data_loader=dev_loader,
                                          device=device,
                                          tb_writer=tb_writer,
                                          log=log,
                                          global_step=global_step,
                                          saver=saver,
                                          args=args)

            if args.eval_after_epoch:
                evaluate_and_save(model=model,
                                  optimizer=optimizer,
                                  data_loader=dev_loader,
                                  device=device,
                                  tb_writer=tb_writer,
                                  log=log,
                                  global_step=global_step,
                                  saver=saver,
                                  args=args)


def evaluate_and_save(model, optimizer, data_loader, device, tb_writer, log, global_step, saver, args):
    log.info('Evaluating...')
    results = evaluate(model, data_loader, device)

    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                            for k, v in results.items())
    log.info('Dev {}'.format(results_str))

    if args.local_rank == 0:
        log.info('Visualizing in TensorBoard...')
        for k, v in results.items():
            tb_writer.add_scalar('dev/{}'.format(k), v, global_step)

        log.info('Saving checkpoint at step {}...'.format(global_step))
        saver.save(step=global_step,
                   model=model,
                   optimizer=optimizer,
                   metric_val=results['Accuracy'])


def evaluate(model, data_loader, device):
    loss_meter = AverageMeter()

    model.eval()
    correct_preds = 0
    correct_avna = 0
    zero_preds = 0
    total_preds = 0
    with torch.no_grad(), tqdm(total=len(data_loader) * args.batch_size, disable=args.local_rank != 0) as progress_bar:
        for batch in data_loader:
            batch = tuple(x.to(device) for x in batch)
            input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps = batch
            current_batch_size = input_ids.shape[0]

            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            word_mask=word_mask,
                            gap_ids=gap_ids,
                            target_gaps=target_gaps)

            loss, gap_scores = outputs[:2]
            loss_meter.update(loss.item(), current_batch_size)

            preds = torch.argmax(gap_scores, dim=1)
            correct_preds += torch.sum(preds == target_gaps).item()
            correct_avna += torch.sum((preds > 0) == (target_gaps > 0)).item()
            zero_preds += torch.sum(preds == 0).item()
            total_preds += current_batch_size

            # Log info
            progress_bar.update(current_batch_size)
            progress_bar.set_postfix(loss=loss_meter.avg)

    model.train()

    results_list = [('Loss', loss_meter.avg),
                    ('Accuracy', correct_preds / total_preds),
                    ('AvNA', correct_avna / total_preds),
                    ('NA_share', zero_preds / total_preds)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    args = get_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    if args.local_rank == 0:
        args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
        log = get_logger(args.save_dir, args.name, log_file=f'log_0.txt')
        log.info(f'Results will be saved in {args.save_dir}.')
    else:
        torch.distributed.barrier()
        args.save_dir = get_save_dir(args.save_dir, args.name, training=True, use_existing_dir=True)
        log = get_logger(args.save_dir, args.name, verbose=False, log_file=f'log_{args.local_rank}.txt')

    if args.local_rank == 0:
        torch.distributed.barrier()

    try:
        main(args, log)
    except:
        log.exception('An error occured...')
