import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import AdamW

try:
    from apex import amp
except ImportError:
    pass

import os
import json
import yaml
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import ModelRegistry
from datasets import DatasetRegistry
from evaluation import EvaluatorRegistry
from utils import CheckpointSaver, get_logger, get_save_dir, get_data_sizes, get_parameter_groups, get_lr_scheduler


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
                        help='Training task. The options are: '
                             'GT (Gapped Text), '
                             'QA (Question Answering).')
    parser.add_argument('--save_dir',
                        type=str,
                        default='experiments',
                        help='Base directory for saving information.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/GT',
                        help='Directory with training and evaluation data.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='This is the number of training samples processed simultaneously by one GPU. '
                             'The effective batch size (number of training samples processed per one '
                             'optimization step) is equal to batch_size * num_gpus * accumulation_steps.')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help='Use apex amp for mixed precision training.')
    parser.add_argument('--amp_opt_level',
                        type=str,
                        default='O1',
                        choices=['O0', 'O1', 'O2', 'O3'],
                        help='Apex amp optimization level. Only used if amp is True.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of training epochs.')
    parser.add_argument('--max_steps',
                        type=int,
                        default=-1,
                        help='Maximum number of training steps. '
                             'Can be used to stop training before the end of an epoch.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='Maximum learning rate for AdamW.')
    parser.add_argument('--warmup_proportion',
                        type=float,
                        default=0.1,
                        help='Proportion of training steps to perform linear learning rate warmup for.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Maximum number of model and optimizer checkpoints to keep.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=50000,
                        help='Evaluate model after processing this many training samples.')
    parser.add_argument('--do_not_eval_after_epoch',
                        default=False,
                        action='store_true',
                        help='Do not evaluate model at the end of every epoch.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='Regularization.')
    parser.add_argument('--seed',
                        type=int,
                        default=111,
                        help='Random seed.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='Local rank for distributed training. '
                             'This argument is provided by torch.distributed.launch.')

    args = parser.parse_args()

    return args


def train(args, logger, tb_writer):
    logger.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))
    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

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
    parameter_groups = get_parameter_groups(model)
    optimizer = AdamW(parameter_groups, lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = get_lr_scheduler(optimizer, num_steps=num_optimization_steps, warmup_proportion=args.warmup_proportion)

    if args.amp:
        amp.register_half_function(torch, 'einsum')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

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
        samples_in_cur_epoch = min([len(train_loader.dataset), samples_till_end])
        disable_progress_bar = (args.local_rank not in [-1, 0])
        with tqdm(total=samples_in_cur_epoch, disable=disable_progress_bar) as progress_bar:
            for step, batch in enumerate(train_loader, 1):
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                current_batch_size = batch['input_ids'].shape[0]

                outputs = model(**batch)
                loss, current_loss_values = outputs[:2]

                loss = loss / args.accumulation_steps
                for name, value in current_loss_values.items():
                    loss_values[name] += value / args.accumulation_steps

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                samples_processed += current_batch_size * world_size
                samples_till_eval -= current_batch_size * world_size
                progress_bar.update(current_batch_size * world_size)

                if step % args.accumulation_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]

                    if args.amp:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # Log info
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

            if not args.do_not_eval_after_epoch:
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
