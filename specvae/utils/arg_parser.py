#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from operator import attrgetter
import os

# Highest-level directory for this repo
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(ArgumentDefaultsHelpFormatter, HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def global_parser(sys_argv):
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)
    ap.add_argument(
        '--data-dir', dest='data_dir', type=str,
        default=os.path.join(REPO_DIR, 'data', 'anderson'),
        help='Directory where data will be pulled from.'
    )
    ap.add_argument(
        '--indices', dest='index_file', type=str,
        default=None,
        help='.npy file containing indices into the loaded data set to be used for'
    )
    ap.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Sets the random seed controlling training. '
        '(E.g. initialization, batching, ...)'
    )
    ap.add_argument(
        '--batch-size', dest='batch_size', type=int, default=256,
        help='Sets the batch size.'
    )
    ap.add_argument(
        '--debug', dest='debug', type=int, default=-1,
        help='Sets the debug flag for max number of loaded data points. '
        'If unspecified, disables debug mode.'
    )
    ap.add_argument(
        '--force', dest='force', default=False, action='store_true',
        help='Overrides failsafes for e.g. overwriting datasets.'
    )
    ap.add_argument(
        '--model', dest='model', default='vae',
        help='String of a model name ["ae", "conv", "vae", "conv_vae"].'
    )
    ap.add_argument(
        '--start-lr', dest='start_lr', type=float, default=0.001,
        help='The learning rate at the beginning of training.'
    )
    ap.add_argument(
        '-e', '--epochs', dest='epochs', type=int, default=2000,
        help='The total number of epochs to train.'
    )
    ap.add_argument(
        '--latent-size', dest='latent_size', type=int, default=10,
        help='The size of the bottleneck latent space.'
    )
    ap.add_argument(
        '--save-model', dest='save_model', type=str, default=None,
        help="The name of a file to save the best learned model to."
    )
    ap.add_argument(
        '--kl-strength', dest='kl_strength', type=float, default=0.05,
        help="Coefficient on the KL divergence loss."
    )

    return ap.parse_args(sys_argv)
