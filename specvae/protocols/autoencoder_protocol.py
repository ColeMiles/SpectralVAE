#!/usr/bin/env python3

import torch
import numpy as np

from specvae.protocols.base_protocol import TrainProtocol
from specvae.models.autoencoder import Autoencoder, ConvAutoencoder, VariationalAutoencoder, ConvVAE


def vae_kl_loss(mu, logvar):
    """ KL-Divergence of VAE ouputted (mu, logvar) with unit Gaussian """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class AutoencoderTrainProtocol(TrainProtocol):
    """Training protocol for autoencoder systems."""

    def __init__(self, trainLoader, validLoader, seed=None, parallel=True,
                 kl_strength=0.0):
        super().__init__(
            trainLoader, validLoader, seed=seed, parallel=parallel
        )
        self.kl_strength = kl_strength
        self.ramping_kl = False
        self.kl_ramp_epochs = None
        self.kl_prefactor = 1.0
        self.epochs_ramped = 0

    # TODO: Do this in a more elegant way
    def ramp_kl(self, ramp_epochs):
        # We're going to ramp up the kl_strength, starting from 0, up to its full value.
        self.kl_ramp_epochs = ramp_epochs
        self.kl_prefactor = 0.0
        self.ramping_kl = True
        self.epochs_ramped = 0

    def initialize_model(self, model_type='vae', **kwargs):
        """Initializer for the model itself.

        Parameters
        ----------
        model_type : str
            String setting which type of model to initialize.
            Options: ['ae', 'conv', 'vae', 'conv_vae']
        """

        if model_type == 'conv':
            self.model = ConvAutoencoder(**kwargs)
        elif model_type == 'ae':
            self.model = Autoencoder(**kwargs)
        elif model_type == 'vae':
            self.model = VariationalAutoencoder(**kwargs)
        elif model_type == 'conv_vae':
            self.model = ConvVAE(**kwargs)
        else:
            raise RuntimeError(f"Unknown model type {model_type}")

        self.model_type = model_type
        self._send_model_to_device()
        self._log_model_info()

    def _train_single_epoch(self, clip=None):
        """Executes the training of the model over a single full pass of
        training data.

        Parameters
        ----------
        clip : float
            Gradient clipping.

        Returns
        -------
        float
            The average training loss/batch.
        """

        self.model.train()  # Unfreeze weights, set model in train mode
        epoch_loss = []
        for batch, in self.trainLoader:
            # Zero the gradients
            self.optimizer.zero_grad()

            # Send batch to device
            batch = batch.to(device=self.device)

            # Run forward prop
            if self.model_type in ['vae', 'conv_vae']:
                output, mu, log_var = self.model.forward(batch)
                loss = self.criterion(output, batch) / len(batch)
                kl_loss = self.kl_prefactor * self.kl_strength * vae_kl_loss(mu, log_var) / len(batch)
                epoch_loss.append([loss.detach().item(), kl_loss.detach().item()])
                loss += kl_loss

            else:
                output = self.model.forward(batch)
                loss = self.criterion(output, batch) / len(batch)
                epoch_loss.append(loss.detach().item())

            # Run back prop
            loss.backward()

            # Clip the gradients
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            # Step the optimizer
            self.optimizer.step()

        # Steps the ramp-up of the kl strength
        if self.ramping_kl:
            self.epochs_ramped += 1
            self.kl_prefactor = np.tanh(2 * self.epochs_ramped / self.kl_ramp_epochs)

        return np.mean(epoch_loss, axis=0)  # mean loss over this epoch

    def _eval_valid_pass(self, valLoader, cache=False):
        """Performs the for loop in evaluating the validation sets. This allows
        for interchanging the passed loaders as desired.

        Parameters
        ----------
        loader : torch.utils.data.dataloader.DataLoader
            Input loader to evaluate on.
        cache : bool
            If true, will save every output on the full pass to a dictionary so
            the user can evaluate every result individually.

        Returns
        -------
        float, List[defaults.Result]
            The average loss on the validation data / batch. Also returns a
            cache of the individual evaluation results if cache is True.
        """

        total_loss = []
        cache_list = []

        for batch, in valLoader:
            batch = batch.to(device=self.device)

            # Run forward prop
            if self.model_type in ['vae', 'conv_vae']:
                output, mu, log_var = self.model.forward(batch)
                loss = self.criterion(output, batch) / len(batch)
                kl_loss = self.kl_prefactor * self.kl_strength * vae_kl_loss(mu, log_var) / len(batch)
                total_loss.append((loss.detach().item(), kl_loss.detach().item()))
                loss += kl_loss
            else:
                output = self.model.forward(batch)
                loss = self.criterion(output, batch) / len(batch)
                total_loss.append(loss.detach().item())

            if cache:
                output = output.cpu().detach().numpy()
                batch = batch.cpu().detach().numpy()
                cache_list_batch = [
                    (
                        output[i], batch[i],
                        np.square(output[i]-batch[i]).mean().item()
                    )
                    for i in range(len(batch))
                ]
                cache_list.extend(cache_list_batch)

        cache_list.sort(key=lambda x: x[-1])
        return np.mean(total_loss, axis=0), cache_list
