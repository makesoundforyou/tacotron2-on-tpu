import os
import time
import argparse
import math
from numpy import finfo
import numpy as np
import jax

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader
from hk_trainer import TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset,
                              num_workers=1,
                              shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    return logger


def validate(trainer, valset, iteration, batch_size, collate_fn, logger):
    """Handles all the validation scoring and printing"""
    val_loader = DataLoader(valset,
                            num_workers=1,
                            shuffle=False,
                            batch_size=batch_size,
                            pin_memory=False,
                            collate_fn=collate_fn)

    val_loss = 0.0
    for i, batch in enumerate(val_loader):
        x, y = trainer.parse_batch(batch)
        reduced_val_loss, y_pred = trainer.validate(x, y)
        val_loss += float(reduced_val_loss)
    val_loss = val_loss / (i + 1)

    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    y_pred = jax.tree_map(lambda x: torch.tensor(np.copy(x)), y_pred)
    y = jax.tree_map(lambda x: torch.tensor(np.copy(x)), y)
    logger.log_validation(val_loss, None, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start,
          hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    hparams (object): comma separated list of "name=value" pairs.
    """

    torch.manual_seed(hparams.seed)
    logger = prepare_directories_and_logger(output_directory, log_directory)
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    from hk_trainer import Trainer
    trainer = Trainer(config=hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if checkpoint_path != None:
        if warm_start:
            trainer.create_model()
            print(
                "Warm start the training with weights from a pretrained model at",
                checkpoint_path)
            trainer.warm_start_model(checkpoint_path, hparams.hk_ignore_layers)
        else:
            print("Loading checkpoint at", checkpoint_path)
            iteration = trainer.load_checkpoint(checkpoint_path)
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    else:
        print("Create a new network with random weights")
        trainer.create_model()

    # ================ MAIN TRAINNIG LOOP! ===================
    start = time.perf_counter()
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            x, y = trainer.parse_batch(batch)
            reduced_loss, norm = trainer.step(x, y)

            duration = time.perf_counter() - start
            start = time.perf_counter()
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                iteration, reduced_loss, norm, duration))
            logger.log_training(reduced_loss, norm,
                                trainer.config.learning_rate, duration,
                                iteration)

            if iteration % hparams.iters_per_checkpoint == 0:
                validate(trainer, valset, iteration, hparams.batch_size,
                         collate_fn, logger)
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}.hk".format(iteration))
                trainer.save_checkpoint(checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output_directory',
                        type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l',
                        '--log_directory',
                        type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c',
                        '--checkpoint_path',
                        type=str,
                        default=None,
                        required=False,
                        help='checkpoint path')
    parser.add_argument(
        '--warm_start',
        action='store_true',
        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams',
                        type=str,
                        required=False,
                        help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, hparams)
