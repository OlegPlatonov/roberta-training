from abc import ABC, ABCMeta, abstractmethod
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import torch

from utils import AverageMeter


class EvaluatorRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.task] = new_cls
        return new_cls

    @classmethod
    def get_evaluator(mcs, task):
        return mcs.registry[task]


class BaseEvaluator(ABC, metaclass=EvaluatorRegistry):
    task = None
    primary_metric = None

    def __init__(self, data_loader, logger, tb_writer, device, world_size, args):
        self.data_loader = data_loader
        self.logger = logger
        self.tb_writer = tb_writer
        self.device = device
        self.world_size = world_size
        self.args = args

    def evaluate(self, model, step):
        self.logger.info('Evaluating...')
        loss_meters = defaultdict(AverageMeter)
        metrics = defaultdict(float)

        model.eval()
        with torch.no_grad(), tqdm(total=len(self.data_loader.dataset),
                                   disable=self.args.local_rank not in [-1, 0]) as progress_bar:

            for batch in self.data_loader:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                current_batch_size = batch['input_ids'].shape[0]

                outputs = model(**batch)
                _, current_loss_values = outputs[:2]

                for name, value in current_loss_values.items():
                    loss_meters[name].update(value, current_batch_size)

                self.update_metrics(metrics, batch, outputs)

                progress_bar.update(current_batch_size * self.world_size)
                progress_bar.set_postfix(**{name: loss_meter.avg for name, loss_meter in loss_meters.items()})

        model.train()

        results = self.get_final_metrics(metrics, loss_meters)
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        self.logger.info('Dev {}'.format(results_str))

        if self.args.local_rank in [-1, 0]:
            for k, v in results.items():
                self.tb_writer.add_scalar('dev/{}'.format(k), v, step)

        return results

    @abstractmethod
    def update_metrics(self, metrics, batch, outputs):
        pass

    @abstractmethod
    def get_final_metrics(self, metrics, loss_meters):
        pass


class EvaluatorForGappedText(BaseEvaluator):
    task = 'GT'
    primary_metric = 'Accuracy'

    def update_metrics(self, metrics, batch, outputs):
        gap_scores = outputs[2]
        preds = torch.argmax(gap_scores, dim=1)
        target_gaps = batch['target_gaps']
        metrics['correct_preds'] += torch.sum(preds == target_gaps).item()
        metrics['correct_avna'] += torch.sum((preds > 0) == (target_gaps > 0)).item()
        metrics['zero_preds'] += torch.sum(preds == 0).item()
        metrics['total_preds'] += batch['input_ids'].shape[0]

    def get_final_metrics(self, metrics, loss_meters):
        results = OrderedDict([
            ('Loss', loss_meters['Loss'].avg),
            ('Accuracy', metrics['correct_preds'] / metrics['total_preds']),
            ('AvNA', metrics['correct_avna'] / metrics['total_preds']),
            ('NA_share', metrics['zero_preds'] / metrics['total_preds'])
        ])

        return results
