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

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.bert import BertForGappedText
from models.datasets import GT_Dataset, GT_collate_fn
from utils.utils_gt import CheckpointSaver, AverageMeter, get_logger, get_save_dir, get_num_data_samples

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


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
    parser.add_argument('--no_cuda',
                        type=lambda s: s.lower().startswith('t'),
                        default=False)
    parser.add_argument('--fp16',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use 16-bit float precision instead of 32-bit')
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
    parser.add_argument('--freeze_proportion',
                        default=0,
                        type=float,
                        help='Proportion of training to freeze encoder for. '
                             'E.g., 0.01 = 1% of training.'
                             'Everything except for embedding layer and output layer will be frozen.')
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

    return args


def main(args, log):
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as file:
        yaml.dump(vars(args), file)

    tb_writer = SummaryWriter(args.save_dir)
    device = 'cpu' if args.no_cuda else 'cuda'
    num_gpus = 0 if args.no_cuda else torch.cuda.device_count()
    log.info(f'Number of GPUs to use: {num_gpus}.')
    log.info(f'Effective batch size: {args.batch_size * num_gpus * args.accumulation_steps}.')

    args.batch_size *= max(1, num_gpus)

    num_data_samples, num_unique_data_epochs = get_num_data_samples(args.data_dir, args.num_epochs, log)
    num_optimization_steps = sum(num_data_samples) // args.batch_size // args.accumulation_steps
    log.info(f'Total number of optimization steps: {num_optimization_steps}.')

    # Set random seed
    log.info(f'Using random seed {args.seed}.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info(f'Loading model {args.model}...')
    model = BertForGappedText.from_pretrained(args.model)

    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as file:
        yaml.dump(model.config.__dict__, file)

    if args.fp16:
        log.info('Using 16-bit float precision.')
        model.half()
        model.output_layer.dtype = torch.float16

    if num_gpus > 1:
        model = nn.DataParallel(model)

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

    if args.fp16:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_optimization_steps)

    global_step = 0
    samples_processed = 0

    # Get dev data loader
    dev_data_file = os.path.join(args.data_dir, f'Dev.csv')
    log.info(f'Creating dev dataset from {dev_data_file}...')
    dev_dataset = GT_Dataset(dev_data_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=GT_collate_fn)

    # Train
    log.info('Training...')

    frozen = False
    if args.freeze_proportion > 0:
        freeze_steps = int(num_optimization_steps * args.freeze_proportion)
        log.info(f'Freezing encoder for {freeze_steps} training steps. '
                 'Everything except for embedding layer and output layer will be frozen.')
        frozen = True

        if hasattr(model, 'module'):
            for param in model.module.bert.parameters():
                param.requires_grad = False
            for param in model.module.bert.embeddings.parameters():
                param.requires_grad = True
        else:
            for param in model.bert.parameters():
                param.requires_grad = False
            for param in model.bert.embeddings.parameters():
                param.requires_grad = True

    steps_till_eval = args.eval_steps
    for epoch in range(1, args.num_epochs + 1):
        train_data_file_num = ((epoch - 1) % num_unique_data_epochs) + 1
        train_data_file = os.path.join(args.data_dir, f'Epoch_{train_data_file_num}.csv')
        log.info(f'Creating training dataset from {train_data_file}...')
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

                if num_gpus > 1:
                    loss = loss.mean()

                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps

                loss_val += loss.item()

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                samples_processed += current_batch_size
                steps_till_eval -= current_batch_size
                progress_bar.update(current_batch_size)

                if step % args.accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        current_lr = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log info
                    if not args.fp16:
                        current_lr = optimizer.get_lr()[0]
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val, step=global_step, lr=current_lr)
                    tb_writer.add_scalar('train/Loss', loss_val, global_step)
                    tb_writer.add_scalar('train/LR', current_lr, global_step)
                    loss_val = 0

                    if frozen and global_step >= freeze_steps:
                        log.info(f'Unfreezing encoder at step {global_step}.')
                        frozen = False
                        if hasattr(model, 'module'):
                            for param in model.module.bert.parameters():
                                param.requires_grad = True
                        else:
                            for param in model.bert.parameters():
                                param.requires_grad = True

                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps
                        evaluate_and_save(model=model,
                                          optimizer=optimizer,
                                          data_loader=dev_loader,
                                          device=device,
                                          tb_writer=tb_writer,
                                          log=log,
                                          global_step=global_step,
                                          saver=saver)

            if args.eval_after_epoch:
                evaluate_and_save(model=model,
                                  optimizer=optimizer,
                                  data_loader=dev_loader,
                                  device=device,
                                  tb_writer=tb_writer,
                                  log=log,
                                  global_step=global_step,
                                  saver=saver)


def evaluate_and_save(model, optimizer, data_loader, device, tb_writer, log, global_step, saver):
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
        tb_writer.add_scalar('dev/{}'.format(k), v, global_step)


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
    args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
    log = get_logger(args.save_dir, args.name)

    try:
        main(args, log)
    except:
        log.exception('An error occured...')
