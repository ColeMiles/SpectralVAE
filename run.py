#!/usr/bin/env python3
import sys
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from specvae.utils.arg_parser import global_parser
from specvae.utils.plot_util import plot_training
from specvae.utils.logger import log
from specvae.data_util.data_util import prepare_autoencoder_data
from specvae.protocols.autoencoder_protocol import AutoencoderTrainProtocol


if __name__ == '__main__':
    args = global_parser(sys.argv[1:])

    indices = pickle.load(open(args.index_file, 'rb')) if args.index_file is not None else None

    conv_model = args.model in ["conv", "conv_vae"]

    train_loader, val_loader, test_loader = prepare_autoencoder_data(
        args.data_dir, batch_size=args.batch_size, conv=conv_model, indices=indices
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

    # Peek at an example to get the training data size
    ex_batch, = next(iter(val_loader))
    input_size = ex_batch.shape[-1]

    protocol = AutoencoderTrainProtocol(
        train_loader, val_loader, seed=args.seed, parallel=False,
        kl_strength=args.kl_strength
    )
    if conv_model:
        protocol.initialize_model(
            model_type=args.model, input_size=input_size,
            conv_kernel_sizes=[5, 5, 5, 5],
            channels=[1, 2, 4, 8],
            pool_kernel_sizes=[2, 2, 2, 2],
            latent_space_size=args.latent_size,
        )
    else:
        protocol.initialize_model(
            model_type=args.model, input_size=input_size,
            hidden_sizes=[240, 160, 80, 60],
            latent_space_size=args.latent_size
        )
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(protocol.model.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.00002,
    )
    protocol.initialize_support(criterion, optimizer, scheduler)
    protocol.ramp_kl(args.epochs // 2)

    train_losses, val_losses, learning_rates = protocol.train(args.epochs)

    if args.save_model is not None:
        torch.save(protocol.model.state_dict(), args.save_model)
        log.info("Saved best model to {}".format(args.save_model))
        fig, (loss_ax, lr_ax) = plot_training(train_losses, val_losses, learning_rates)
        modelfile_name = os.path.splitext(args.save_model)[0]
        plt.savefig(modelfile_name + ".png")
        plt.clf()
