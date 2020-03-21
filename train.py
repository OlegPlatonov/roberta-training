import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

import os
import json
import yaml
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from models import ModelRegistry
from datasets import DatasetRegistry
from evaluation import EvaluatorRegistry
from utils import CheckpointSaver, get_logger, get_save_dir, get_data_sizes


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Experiment name.')
    parser.add_argument('--model',
                        type=str,
                        default='roberta-base',
                        help='Pretrained model name or path.')
    parser.add_argument('--task',
                        type=str,
                        default='GT',
                        choices=['GT', 'QA'],
                        help='Training task name.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./experiments',
                        help='Base directory for saving information.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/GT')
    parser.add_argument('--seed',
                        type=int,
                        default=12)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='This is the number of training samples processed together by one GPU. '
                             'The effective batch size (number of training samples processed per one '
                             'optimization step) is equal to batch_size * num_gpus * accumulation_steps.')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--fp16',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1)
    parser.add_argument('--max_steps',
                        type=int,
                        default=-1)
    parser.add_argument('--learning_rate',
                        default=1e-5,
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=30)
    parser.add_argument('--eval_every',
                        type=int,
                        default=50000,
                        help='Evaluate model after processing this many training samples.')
    parser.add_argument('--eval_after_epoch',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to evaluate model at the end of every epoch.')
    parser.add_argument('--apex_level',
                        default='O1',
                        choices=['O0', 'O1', 'O2', 'O3'],
                        help='Apex optimization level. Only used if fp16 is True.')
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help='Local rank for distributed training. Use -1 for single GPU training')

    args = parser.parse_args()

    return args


def train(args, logger, tb_writer):
    logger.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))
    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

    device_id = args.local_rank if args.local_rank != -1 else 0
    device = torch.device('cuda', device_id)
    logger.warning(f'Using GPU {args.local_rank}.')

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    logger.info(f'Total number of GPUs used: {world_size}.')
    effective_batch_size = args.batch_size * world_size * args.accumulation_steps
    logger.info(f'Effective batch size: {effective_batch_size}.')

    num_train_samples_per_epoch, num_dev_samples, num_unique_train_epochs = get_data_sizes(data_dir=args.data_dir,
                                                                                           num_epochs=args.num_epochs,
                                                                                           logger=logger)
    num_optimization_steps = sum(num_train_samples_per_epoch) // world_size // args.batch_size // \
                             args.accumulation_steps
    if args.max_steps > 0:
        num_optimization_steps = min(num_optimization_steps, args.max_steps)
    logger.info(f'Total number of optimization steps: {num_optimization_steps}.')

    # Set random seed
    logger.info(f'Using random seed {args.seed}.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    logger.info(f'Loading model {args.model} for task {args.task}...')
    model = ModelRegistry.get_model(args.task).from_pretrained(args.model)

    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as file:
            json.dump(model.config.__dict__, file)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)

    # Get optimizer
    logger.info('Creating optimizer...')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=num_optimization_steps,
                                                num_warmup_steps=num_optimization_steps * args.warmup_proportion)

    if args.fp16:
        amp.register_half_function(torch, 'einsum')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_level)

    if args.local_rank != -1:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)

    # Get dev data loader
    dev_data_file = os.path.join(args.data_dir, f'dev.jsonl.gz')
    logger.info(f'Creating dev dataset from {dev_data_file}...')
    dev_dataset = DatasetRegistry.get_dataset(args.task)(data_file=dev_data_file,
                                                         data_size=num_dev_samples,
                                                         local_rank=-1)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=2 * args.batch_size,
                            num_workers=1,
                            collate_fn=dev_dataset.collate_fn)

    # Get evaluator
    evaluator = EvaluatorRegistry.get_evaluator(args.task)(data_loader=dev_loader,
                                                           logger=logger,
                                                           tb_writer=tb_writer,
                                                           device=device,
                                                           world_size=world_size,
                                                           args=args)

    # Get saver
    saver = CheckpointSaver(save_dir=args.save_dir,
                            max_checkpoints=args.max_checkpoints,
                            primary_metric=evaluator.primary_metric,
                            logger=logger)

    global_step = 0
    samples_processed = 0

    # Train
    logger.info('Training...')
    samples_till_eval = args.eval_every
    for epoch in range(1, args.num_epochs + 1):
        # Get train data loader for current epoch
        train_data_file_num = ((epoch - 1) % num_unique_train_epochs) + 1
        train_data_file = os.path.join(args.data_dir, f'epoch_{train_data_file_num}.jsonl.gz')
        logger.info(f'Creating training dataset from {train_data_file}...')
        train_dataset = DatasetRegistry.get_dataset(args.task)(train_data_file,
                                                               data_size=num_train_samples_per_epoch[epoch - 1],
                                                               local_rank=args.local_rank,
                                                               world_size=world_size)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=1,
                                  collate_fn=train_dataset.collate_fn)

        logger.info(f'Starting epoch {epoch}...')
        model.train()
        model.zero_grad()
        loss_values = defaultdict(float)
        samples_till_end = (num_optimization_steps - global_step) * effective_batch_size
        with torch.enable_grad(), tqdm(total=min([len(train_loader.dataset), samples_till_end]),
                                       disable=args.local_rank not in [-1, 0]) as progress_bar:
            for step, batch in enumerate(train_loader, 1):
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                current_batch_size = batch['input_ids'].shape[0]

                outputs = model(**batch)
                loss, current_loss_values = outputs[:2]

                loss = loss / args.accumulation_steps
                for name, value in current_loss_values.items():
                    loss_values[name] += value / args.accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                samples_processed += current_batch_size * world_size
                samples_till_eval -= current_batch_size * world_size
                progress_bar.update(current_batch_size * world_size)

                if step % args.accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # Log info
                    current_lr = scheduler.get_lr()[0]
                    progress_bar.set_postfix(epoch=epoch, step=global_step, lr=current_lr, **loss_values)
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('train/LR', current_lr, global_step)
                        for name, value in loss_values.items():
                            tb_writer.add_scalar(f'train/{name}', value, global_step)
                    loss_values = {name: 0 for name in loss_values}

                    if global_step == args.max_steps:
                        logger.info('Reached maximum number of optimization steps.')
                        break

                    if samples_till_eval <= 0:
                        samples_till_eval = args.eval_every
                        eval_results = evaluator.evaluate(model, global_step)
                        if args.local_rank in [-1, 0]:
                            saver.save(model, global_step, eval_results)

            if args.eval_after_epoch:
                eval_results = evaluator.evaluate(model, global_step)
                if args.local_rank in [-1, 0]:
                    saver.save(model, global_step, eval_results)


if __name__ == '__main__':
    args = get_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    if args.local_rank in [-1, 0]:
        args.save_dir = get_save_dir(args.save_dir, args.name)
        logger = get_logger(args.save_dir, args.name, log_file=f'log_0.txt')
        logger.info(f'Results will be saved to {args.save_dir}.')
        tb_writer = SummaryWriter(args.save_dir)
    else:
        torch.distributed.barrier()
        args.save_dir = get_save_dir(args.save_dir, args.name, use_existing_dir=True)
        logger = get_logger(args.save_dir, args.name, verbose=False, log_file=f'log_{args.local_rank}.txt')
        tb_writer = None

    if args.local_rank == 0:
        torch.distributed.barrier()

    try:
        train(args, logger, tb_writer)
    except:
        logger.exception('An error occured...')

    if tb_writer is not None:
        tb_writer.close()
