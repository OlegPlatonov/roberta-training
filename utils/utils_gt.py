import os
import logging
import queue
import torch
import tqdm
import shutil
import yaml


def get_logger(log_dir, name, verbose=True, log_file='log.txt'):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.

    Adapted from https://github.com/chrischute/squad.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
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
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.

    Adapted from https://github.com/chrischute/squad.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'
                    .format('max' if maximize_metric else 'min', metric_name))

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
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, args, metric_val, optimizer=None):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            optimizer (torch.optimizer): If not None, also saves optimizer.
        """

        if hasattr(model, 'module'):
            model = model.module

        if args.model_type == 'roberta':
            model = model.roberta

        checkpoint_path = os.path.join(self.save_dir,
                                       'step_{}.pth.tar'.format(step))
        torch.save(model.state_dict(), checkpoint_path)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_path + '.optim')
        self._print('Saved checkpoint: {}'.format(checkpoint_path))

        # Last checkpoint
        last_path = os.path.join(self.save_dir, 'last.pth.tar')
        shutil.copy(checkpoint_path, last_path)
        if optimizer is not None:
            shutil.copy(checkpoint_path + '.optim', last_path + '.optim')
        self._print('{} is now checkpoint from step {}...'.format(last_path, step))

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            if optimizer is not None:
                shutil.copy(checkpoint_path + '.optim', best_path + '.optim')
            self._print('New best checkpoint at step {}...'.format(step))

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


def get_num_data_samples(data_folder, num_epochs, log=None):
    """
    Adapted from https://github.com/huggingface/pytorch-pretrained-BERT.
    """

    num_data_samples = []
    for i in range(1, num_epochs + 1):
        data_file = os.path.join(data_folder, f'Epoch_{i}.csv')
        config_file = os.path.join(data_folder, f'Epoch_{i}_config.yaml')
        if os.path.isfile(data_file) and os.path.isfile(config_file):
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            num_data_samples.append(config['size'])
        else:
            break

    num_unique_data_epochs = len(num_data_samples)

    if log is not None:
        log.info(f'{num_unique_data_epochs} unique epochs of data found.')

    for i in range(num_epochs - len(num_data_samples)):
        num_data_samples.append(num_data_samples[i])

    if log is not None:
        log.info(f'Number of samples per epoch: {num_data_samples}')

    return num_data_samples, num_unique_data_epochs


class AverageMeter:
    """Keep track of average values over time.

    Taken from https://github.com/chrischute/squad.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


def get_save_dir(base_dir, name, training, id_max=100, use_existing_dir=False):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.

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
