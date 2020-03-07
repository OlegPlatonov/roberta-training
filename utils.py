import os
import logging
import queue
import torch
import tqdm
import shutil
import yaml


def get_logger(log_dir, name, verbose=True, log_file='log.txt'):
    """
    Adapted from https://github.com/chrischute/squad.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """
        Let `logging` print without breaking `tqdm` progress bars.

        See also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if verbose:
        # Log everything (i.e., DEBUG level and above) to a file
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        # Log everything except DEBUG level (i.e., INFO level and above) to console
        console_handler = StreamHandlerWithTQDM()
        console_handler.setLevel(logging.INFO)

    else:
        # Log INFO level and above to a file
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Log WARN level and above to console
        console_handler = StreamHandlerWithTQDM()
        console_handler.setLevel(logging.WARN)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class CheckpointSaver:
    """
    Adapted from https://github.com/chrischute/squad.
    """
    def __init__(self, save_dir, max_checkpoints, primary_metric,
                 maximize_metric=True, logger=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.primary_metric = primary_metric
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.logger = logger
        self._print('Saver will {}imize {}.'
                    .format('max' if maximize_metric else 'min', primary_metric))

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.logger is not None:
            self.logger.info(message)

    def save(self, model, step, eval_results, optimizer=None):
        self._print('Saving model...')
        if hasattr(model, 'module'):
            model = model.module

        metric_val = eval_results[self.primary_metric]

        checkpoint_path = os.path.join(self.save_dir,
                                       'model_step_{}.bin'.format(step))
        torch.save(model.state_dict(), checkpoint_path)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_path + '.optim')
        self._print('Saved checkpoint: {}'.format(checkpoint_path))

        # Last checkpoint
        last_path = os.path.join(self.save_dir, 'model_last.bin')
        shutil.copy(checkpoint_path, last_path)
        if optimizer is not None:
            shutil.copy(checkpoint_path + '.optim', last_path + '.optim')
        self._print('{} is now checkpoint from step {}.'.format(last_path, step))

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'model_best.bin')
            shutil.copy(checkpoint_path, best_path)
            if optimizer is not None:
                shutil.copy(checkpoint_path + '.optim', best_path + '.optim')
            self._print('New best checkpoint!')
            self._print('{} is now checkpoint from step {}.'.format(best_path, step))

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                if optimizer is not None:
                    os.remove(worst_ckpt + '.optim')
                self._print('Removed checkpoint: {}'.format(worst_ckpt))
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def get_data_sizes(data_dir, num_epochs, logger=None):
    num_train_samples_per_epoch = []
    config_file = os.path.join(data_dir, f'data_config.yaml')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        num_dev_samples = config['dev_size']
        for epoch in range(1, num_epochs + 1):
            if f'epoch_{epoch}_size' in config:
                num_train_samples_per_epoch.append(config[f'epoch_{epoch}_size'])
            else:
                break

    num_unique_train_epochs = len(num_train_samples_per_epoch)

    if logger is not None:
        logger.info(f'{num_unique_train_epochs} unique epochs of data found.')

    for i in range(num_epochs - len(num_train_samples_per_epoch)):
        num_train_samples_per_epoch.append(num_train_samples_per_epoch[i])

    if logger is not None:
        logger.info(f'Number of samples per epoch: {num_train_samples_per_epoch}')

    return num_train_samples_per_epoch, num_dev_samples, num_unique_train_epochs


class AverageMeter:
    """
    Taken from https://github.com/chrischute/squad.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, num_samples=1):
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


def get_save_dir(base_dir, name, training, id_max=100, use_existing_dir=False):
    """
    Adapted from https://github.com/chrischute/squad.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, '{}-{:02d}'.format(name, uid))
        if not os.path.exists(save_dir):
            if not use_existing_dir:
                os.makedirs(save_dir)
                return save_dir
            else:
                save_dir = os.path.join(base_dir, subdir, '{}-{:02d}'.format(name, uid - 1))
                return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')
