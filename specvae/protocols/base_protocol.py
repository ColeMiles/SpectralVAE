#!/usr/bin/env python3

import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from specvae.utils.logger import log
from specvae.utils import ml_utils


# At the time that the module is called, this should be a global variable
CUDA_AVAIL = torch.cuda.is_available()


class TrainProtocol:
    """Base class for performing ML training. Attributes that are not also
    inputs to __init__ are listed below. A standard use case for this type of
    class might look like this:

    Example
    -------
    > x = TrainProtocol(train_loader, valid_loader)
    > x.initialize_model(model_params)
    > x.initialize_support()  # inits the optimizer, loss and scheduler
    > x.train()

    Attributes
    ----------
    trainLoader : torch.utils.data.dataloader.DataLoader
    validLoader : torch.utils.data.dataloader.DataLoader
    device : str
        Will be 'cuda:0' if at least one GPU is available, else 'cpu'.
    parallel : bool
    model : torch.nn.Module
        Class inheriting the torch.nn.Module back end.
    self.criterion : torch.nn._Loss
        Note, I think this is the correct object type. The criterion is the
        loss function.
    self.optimizer : torch.optim.Optimizer
        The numerical optimization protocol. Usually, we should choose Adam for
        this.
    scheduler : torch.optim.lr_scheduler
        Defines a protocol for training, usually for updating the learning
        rate.
    best_model_state_dict : Dict
        The torch model state dictionary corresponding to the best validation
        result. Used as a lightweight way to store the model parameters.
    """

    def __init__(self, trainLoader, validLoader, seed=None, parallel=True):
        """Initializer.

        Parameters
        ----------
        trainLoader : torch.utils.data.dataloader.DataLoader
            The training loader object.
        validLoader : torch.utils.data.dataloader.DataLoader
            The cross validation (or testing) loader object.
        seed : int
            Seeds random, numpy and torch.
        parallel : bool
            Switch from parallel GPU training to single, if desired. Ignored if
            no GPUs are available. Default is True
        """

        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.device = torch.device('cuda:0' if CUDA_AVAIL else 'cpu')
        self.parallel = parallel and CUDA_AVAIL
        self._log_cuda_info()
        if seed is not None:
            ml_utils.seed_all(seed)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_state_dict = None

    def initialize_model(self):
        raise NotImplementedError

    def _eval_valid_pass(self, valLoader, cache=False):
        raise NotImplementedError

    def _train_single_epoch(self, clip):
        raise NotImplementedError

    def _send_model_to_device(self):
        """Initializes the parallel model if specified, and sends the model to
        the used device."""

        if self.parallel:
            self.model = nn.DataParallel(self.model)
            log.info("Parallel model defined")
        self.model.to(self.device)
        log.info(f"Model sent to {self.device}")

    def _log_model_info(self):
        """Computes the number of trainable parameters and logs it."""

        n_trainable = sum(
            param.numel() for param in self.model.parameters()
            if param.requires_grad
        )
        log.info(f"model has {n_trainable} trainable parameters")

    def _log_cuda_info(self):
        """Informs the user about the available GPU status."""

        log.info(
            f"device is {self.device}; CUDA_AVAIL {CUDA_AVAIL}; "
            f"parallel {self.parallel}"
        )
        if CUDA_AVAIL:
            gpu_count = torch.cuda.device_count()
            log.info(f"number of GPUs is {gpu_count}")

    def initialize_support(
        self, criterion, optimizer, scheduler
    ):
        """Initializes the criterion, optimize and scheduler.

        Parameters
        ----------
        criterion : Tuple[str, Dict[str, Any]]
            String (and kwargs) indexing which criterion to load (and its
            parameters).
        optimizer : Tuple[str, Dict[str, Any]]
            Tuple containing the initialization string for the optimizer, and
            optional dictionary containing the kwargs to pass to the optimizer
            initializer.
        scheduler : Tuple[str, Dict[str, Any]]
            Save as the optimizer, but for the scheduler.
        """

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _eval_valid(self):
        """Similar to _train_single_epoch above, this method will evaluate a
        full pass on the validation data.

        Returns
        -------
        float
            The average loss on the validation data / batch.
        """

        self.model.eval()  # Freeze weights, set model to evaluation mode

        # Double check that torch will not be updating gradients or doing
        # anything we don't want it to do.
        with torch.no_grad():
            losses, __ = self._eval_valid_pass(self.validLoader)

        return losses

    def _update_state_dict(self, best_valid_loss, valid_loss, epoch):
        """Updates the best_model_state_dict attribute if the valid loss is
        less than the best up-until-now valid loss.

        Parameters
        ----------
        best_valid_loss : float
            The best validation loss so far.
        valid_loss : float
            The current validation loss on the provided epoch.
        epoch : int
            The current epoch.

        Returns
        -------
        float
            min(best_valid_loss, valid_loss)
        """

        if valid_loss < best_valid_loss or epoch == 0:
            self.best_model_state_dict = self.model.state_dict()
            log.info(
                f'\tVal. Loss: {valid_loss:.05e} < Best Val. Loss '
                f'{best_valid_loss:.05e}'
            )
            log.info("\tUpdating best_model_state_dict")
        else:
            log.info(f'\tVal. Loss: {valid_loss:.05e}')

        return min(best_valid_loss, valid_loss)

    def _step_scheduler(self, valid_loss):
        """Steps the scheduler and outputs information about the learning
        rate.

        Parameters
        ----------
        valid_loss : float
            The current epoch's validation loss.

        Returns
        -------
        float
            The current learning rate.
        """

        clr = self.scheduler.get_lr()[0]
        log.info(f"\t Learning rate {clr:.03e}")
        self.scheduler.step()
        return clr

    def train(self, epochs, clip=None):
        """Executes model training.

        Parameters
        ----------
        epochs : int
            Number of full passes through the training data.
        clip : float, optional
            Gradient clipping.

        Returns
        -------
        train loss, validation loss, learning rates : list
        """

        # Keep track of the best validation loss so that we know when to save
        # the model state dictionary.
        best_valid_loss = float('inf')

        # Begin training
        train_loss_list = []
        valid_loss_list = []
        learning_rates = []
        for epoch in range(epochs):

            # Train a single epoch
            t0 = time.time()
            train_losses = self._train_single_epoch(clip)
            t_total = time.time() - t0
            log.info(f"Epoch {epoch:04} [{t_total:3.02f}s]")
            log.info(f'\tTrain Loss: {train_losses}')

            # Evaluate on the validation data
            valid_losses = self._eval_valid()

            # Step the scheduler - returns the current learning rate (clr)
            total_valid_loss = np.sum(valid_losses).item()
            clr = self._step_scheduler(total_valid_loss)

            # Update the best state dictionary of the model for loading in
            # later on in the process
            best_valid_loss = self._update_state_dict(
                best_valid_loss, total_valid_loss, epoch
            )

            train_loss_list.append(train_losses)
            valid_loss_list.append(valid_losses)
            learning_rates.append(clr)

        log.info("Setting model to best state dict")
        self.model.load_state_dict(self.best_model_state_dict)

        return train_loss_list, valid_loss_list, learning_rates

    def eval(self, loader_override=None):
        """Systematically evaluates the validation, or dataset corresponding to
        the loader specified in the loader_override argument, dataset."""

        if loader_override is not None:
            log.warning(
                "Default validation loader is overridden - ensure this is "
                "intentional, as this is likely evaluating on a testing set"
            )

        loader = self.validLoader if loader_override is None \
            else loader_override

        with torch.no_grad():
            losses, cache = self._eval_valid_pass(loader, cache=True)
        log.info(f"Eval complete: loss {losses}")

        return losses, cache
