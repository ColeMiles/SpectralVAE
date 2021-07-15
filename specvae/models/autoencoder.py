#!/usr/bin/env python3

import torch
import torch.nn as nn
from math import floor

from specvae.utils.logger import log


class Reshape(nn.Module):
    """ Helper module which performs a reshape operation."""
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.reshape(self.new_shape)


class Encoder(nn.Module):
    """Compresses an input vector down to a smaller representation."""

    def __init__(
        self, input_size, hidden_sizes, latent_space_size, dropout=0.0,
        activation=torch.relu
    ):
        """Initializer.

        Parameters
        ----------
        input_size : int
            The size (number of features) in the input.
        hidden_sizes : list
            A list of integers in which each entry represents a hidden layer
            of size hidden_sizes[ii].
        latent_space_size : int
            The number of neurons in the latent space, which can also be
            considered as the encoder output.
        dropout : float
            Default is 0.
        activation
            Activation function.
        """

        super().__init__()

        self.input_layer = torch.nn.Linear(input_size, hidden_sizes[0])

        hidden_layers = []

        for ii in range(0, len(hidden_sizes) - 1):
            hidden_layers.append(
                torch.nn.Linear(hidden_sizes[ii], hidden_sizes[ii + 1])
            )
            hidden_layers.append(
                torch.nn.BatchNorm1d(hidden_sizes[ii + 1])
            )

        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = torch.nn.Linear(
            hidden_sizes[-1], latent_space_size
        )

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward propagation for the Encoder. Applies activations to every
        layer, in addition to dropout."""

        x = self.dropout(self.activation(self.input_layer(x)))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.output_layer(x)


class Decoder(nn.Module):
    """Reconstruct an input vector from the latent space."""

    def __init__(
        self, latent_space_size, hidden_sizes, output_size, dropout=0.0,
        activation=torch.relu
    ):
        """Initializer.

        Parameters
        ----------
        latent_space_size : int
            The number of neurons in the latent space, which can also be
            considered as the encoder output.
        hidden_sizes : list
            A list of integers in which each entry represents a hidden layer
            of size hidden_sizes[ii].
        output_size : int
            The size (number of features) in the output.
        dropout : float
            Default is 0.
        activation
            Activation function.
        """

        super().__init__()

        self.input_layer = torch.nn.Linear(latent_space_size, hidden_sizes[0])

        hidden_layers = []

        for ii in range(0, len(hidden_sizes) - 1):
            hidden_layers.append(
                nn.Linear(hidden_sizes[ii], hidden_sizes[ii + 1])
            )
            hidden_layers.append(
                nn.BatchNorm1d(hidden_sizes[ii + 1])
            )

        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = torch.nn.Linear(
            hidden_sizes[-1], output_size
        )

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward propagation for the Decoder. Applies activations to every
        layer, in addition to dropout."""

        x = self.dropout(self.activation(self.input_layer(x)))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))

        # Linear activation at the end!
        return self.output_layer(x)


class Autoencoder(nn.Module):
    """Vector-to-vector (fixed-length) autoencoder."""

    def __init__(
        self, input_size, hidden_sizes, latent_space_size, dropout=0.0
    ):
        """Initializer.

        Parameters
        ----------
        input_size : int
            The size of the input and output of the autoencoder.
        hidden_sizes : list
            A list of integers determining the number of and number of neurons
            per layer.
        latent_space_size : int
        """

        super().__init__()

        if not isinstance(hidden_sizes, list):
            critical = "Parameter hidden_sizes must be of type List[int]"
            log.critical(critical)
            raise ValueError(critical)

        if latent_space_size > input_size:
            log.warning(
                f"The latent space size {latent_space_size} is greater than "
                f"the input size {input_size} - ensure this is intended"
            )

        self.encoder = Encoder(
            input_size, hidden_sizes, latent_space_size, dropout=dropout
        )
        self.decoder = Decoder(
            latent_space_size, list(reversed(hidden_sizes)), input_size,
            dropout=dropout
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Forward propagation for the autoencoder."""

        latent_space = self.encoder(x)
        decoded = self.decoder(latent_space)
        return self.softplus(decoded)


class VariationalAutoencoder(nn.Module):
    """Vector-to-vector (fixed-length) variational autoencoder."""

    def __init__(
            self, input_size, hidden_sizes, latent_space_size, dropout=0.0
    ):
        """Initializer.

        Parameters
        ----------
        input_size : int
            The size of the input and output of the autoencoder.
        hidden_sizes : list
            A list of integers determining the number of and number of neurons
            per layer.
        latent_space_size : int
        """

        super().__init__()

        if not isinstance(hidden_sizes, list):
            critical = "Parameter hidden_sizes must be of type List[int]"
            log.critical(critical)
            raise ValueError(critical)

        if latent_space_size > input_size:
            log.warning(
                f"The latent space size {latent_space_size} is greater than "
                f"the input size {input_size} - ensure this is intended"
            )

        self.encoder = Encoder(
            input_size, hidden_sizes, 2 * latent_space_size, dropout=dropout
        )
        self.decoder = Decoder(
            latent_space_size, list(reversed(hidden_sizes)), input_size,
            dropout=dropout
        )
        self.latent_space_size = latent_space_size
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Forward propagation for the autoencoder."""

        x = self.encoder(x)
        x = x.view(-1, 2, self.latent_space_size)
        # Use first set of outputs as mean of distributions
        mu = x[:, 0, :]
        # And the second set as the log variances
        log_var = x[:, 1, :]
        z = reparameterize(mu, log_var)
        return self.softplus(self.decoder(z)), mu, log_var


class ConvEncoder(nn.Module):
    """ A convolutional encoder, intended as a drop-in replacement for Encoder.
    """
    def __init__(
        self, input_size, conv_kernel_sizes, channels,
        pool_kernel_sizes, latent_space_size, conv_strides=None,
        pool_strides=None, pool_op=nn.MaxPool1d, input_channels=1,
        dropout=0.0, activation=nn.ReLU,
    ):
        """ Initializer. The overall structure of this module is:

        Input -> Conv -> Conv -> Pool -> <repeat> -> Linear

        Where the number of [Conv -> Conv -> Pool] units is equal to
        len(conv_kernel_sizes).

        conv_kernel_sizes, channels, pool_kernel_sizes, conv_strides, and
        pool_strides must all be the same length (if provided).

        Parameters
        ----------
        input_size : int
            The spatial size of the input.
        conv_kernel_sizes : List[int]
            The kernel sizes of the convolutions at each stage.
            Padding at each conv is of size (kernel_size - 1) // 2, to maintain
            spatial resolution if conv_stride = 1.
        channels : List[int]
            The number of channels at each stage.
        pool_kernel_sizes : List[int]
            The kernel sizes of the pooling operations at each stage. There is
            no padding done.
        latent_space_size : int
            The size of the final latent space.
        conv_strides : Optional[List[int]]
            The stride of the convolutions at each stage. If not provided, is 1
            for all stages.
        pool_strides : Optional[List[int]]
            The stride of the pooling operations at each stage. If not
            provided, is 2 for all stages.
        pool_op : Callable
            Constructor for a nn.Module to perform the pooling operation. By
            default, nn.MaxPool1d.
        input_channels : int
            The number of channels the input has.
        dropout : float
            Probability of dropping internal values while training.
        activation : Callable
            Constructor for a nn.Module defining the activation function.
        """

        super().__init__()

        if not len(conv_kernel_sizes) == len(channels) \
           == len(pool_kernel_sizes):
            raise ValueError(
                "Specification lists must all be the same length."
            )

        if conv_strides is None:
            conv_strides = [1] * len(conv_kernel_sizes)
        elif len(conv_strides) != len(conv_kernel_sizes):
            raise ValueError(
                "len(conv_strides) must equal len(conv_kernel_sizes)."
            )

        if pool_strides is None:
            pool_strides = [2] * len(pool_kernel_sizes)
        elif len(pool_strides) != len(pool_kernel_sizes):
            raise ValueError(
                "len(pool_strides) must equal len(pool_kernel_sizes)."
            )

        dropout_module = nn.Dropout(p=dropout, inplace=True)
        activation_module = activation(inplace=True)

        modules = []
        curr_size = input_size

        for i in range(len(conv_kernel_sizes)):
            # Padding chosen such that, for odd kernel sizes and stride = 1,
            #  the result after convolution is the same length as the input.
            conv_kernel_size = conv_kernel_sizes[i]
            conv_padding = (conv_kernel_sizes[i] - 1) // 2
            conv_stride = conv_strides[i]
            prev_channels = input_channels if i == 0 else channels[i-1]
            new_channels = channels[i]

            # The overall structure is Conv -> Conv -> MaxPool, to give two
            # convolutions at each spatial resolution to allow the receptive
            # field to be large enough before pooling.
            modules.append(
                nn.Conv1d(
                    prev_channels, new_channels,
                    kernel_size=conv_kernel_size, padding=conv_padding,
                    stride=conv_stride
                )
            )
            curr_size = floor(
                (curr_size + 2 * conv_padding - conv_kernel_size)
                / conv_stride + 1
            )
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(new_channels))

            if dropout != 0.0:
                modules.append(dropout_module)

            modules.append(
                nn.Conv1d(
                    new_channels, new_channels,
                    kernel_size=conv_kernel_size, padding=conv_padding
                )
            )
            curr_size = floor(
                (curr_size + 2 * conv_padding - conv_kernel_size)
                / conv_stride + 1
            )

            pool_kernel_size = pool_kernel_sizes[i]
            pool_stride = pool_strides[i]
            modules.append(
                pool_op(pool_kernel_size, stride=pool_stride)
            )
            curr_size = floor((curr_size - pool_kernel_size) / pool_stride + 1)
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(channels[i]))

            if dropout != 0.0:
                modules.append(dropout_module)

        modules.append(nn.Flatten())
        modules.append(
            nn.Linear(curr_size * channels[-1], latent_space_size)
        )

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """Forward propagation."""

        return self.module(x)


class ConvDecoder(nn.Module):
    """A convolutional decoder, intended as a drop-in replacement for Decoder
    """

    def __init__(
        self, latent_space_size, conv_kernel_sizes, channels,
        pool_kernel_sizes, output_size, conv_strides=None, pool_strides=None,
        output_channels=1, dropout=0.0, activation=nn.ReLU
    ):
        """ Initializer. The overall structure of this module is:

        Linear -> ConvTranspose -> Conv -> <repeat> -> Output

        Where the number of [ConvTranspose -> Conv] units is equal to
        len(conv_kernel_sizes).

        conv_kernel_sizes, channels, pool_kernel_sizes, conv_strides, and
        pool_strides must all be the same length (if provided).

        The spatial size of the first hidden layer (following the nn.Linear)
        is calculated as if this was a ConvEncoder, running in reverse. [i.e.
        if the arguments to both ConvEncoder, ConvDecoder are symmetric, this
        size should match on either side of the nn.Linear.

        Parameters
        ----------
        latent_space_size : int
            The size of the input latent space.
        conv_kernel_sizes : List[int]
            The kernel sizes of the convolutions at each stage. Padding at
            each ConvTranspose is 0, and at each Conv is of size
            (kernel_size - 1) // 2, to maintain spatial resolution if
            conv_stride = 1.
        channels : List[int]
            The number of channels at each stage.
        pool_kernel_sizes : List[int]
            The kernel sizes of the "pooling" operations at each stage that the
            decoder must undo.
        output_size : int
            The spatial size of the output.
        conv_strides : Optional[List[int]]
            The stride of the convolutions at each stage. If not provided, is 1
            for all stages.
        pool_strides : Optional[List[int]]
            The stride of the "pooling" operations at each stage the decoder
            must undo. If not provided, is 2 for all stages.
        output_channels : int
            The number of channels the output has.
        dropout : float
            Probability of dropping internal values while training.
        activation : Callable
            Constructor for a nn.Module defining the activation function.
        """

        super().__init__()

        if not len(conv_kernel_sizes) == len(channels) \
           == len(pool_kernel_sizes):
            raise ValueError(
                "Specification lists must all be the same length."
            )

        if conv_strides is None:
            conv_strides = [1] * len(conv_kernel_sizes)
        elif len(conv_strides) != len(conv_kernel_sizes):
            raise ValueError(
                "len(conv_strides) must equal len(conv_kernel_sizes)."
            )

        if pool_strides is None:
            pool_strides = [2] * len(pool_kernel_sizes)
        elif len(pool_strides) != len(pool_kernel_sizes):
            raise ValueError(
                "len(pool_strides) must equal len(pool_kernel_sizes)."
            )

        activation_module = activation(inplace=True)
        dropout_module = nn.Dropout(p=dropout, inplace=True)

        # Do some calculations to figure out what the spatial sizes should be
        #  to match up with a symmetric ConvEncoder
        sizes = [output_size]
        reversed_layers = zip(
            reversed(conv_kernel_sizes), reversed(conv_strides),
            reversed(pool_kernel_sizes), reversed(pool_strides)
        )
        cur_size = output_size
        for c_ksize, c_str, p_ksize, p_str in reversed_layers:
            c_pad = (c_ksize - 1) // 2
            # Conv -> Conv
            cur_size = floor((cur_size + 2 * c_pad - c_ksize) / c_str + 1)
            # Pool
            cur_size = floor((cur_size - p_ksize) / p_str + 1)
            sizes.append(cur_size)
        sizes.reverse()

        modules = []

        # Initial dense layer, then reshape to convolutional shape
        modules.append(
            nn.Linear(latent_space_size, sizes[0] * channels[0])
        )
        modules.append(
            Reshape((-1, channels[0], sizes[0]))
        )

        for i in range(len(conv_kernel_sizes)):
            # Padding chosen such that, for odd kernel sizes and stride = 1,
            #  the result after convolution is the same length as the input.
            conv_kernel_size = conv_kernel_sizes[i]
            conv_padding = (conv_kernel_sizes[i] - 1) // 2
            conv_stride = conv_strides[i]
            pool_kernel_size = pool_kernel_sizes[i]
            pool_stride = pool_strides[i]
            prev_channels = channels[i]
            new_channels = output_channels if i == len(conv_kernel_sizes) - 1 \
                else channels[i+1]
            prev_size = sizes[i]
            new_size = sizes[i+1]

            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(prev_channels))
            if dropout != 0.0:
                modules.append(dropout_module)

            # The overall structure is ConvTranspose -> Conv
            convt_size = (prev_size - 1) * pool_stride + pool_kernel_size
            modules.append(
                nn.ConvTranspose1d(
                    prev_channels, prev_channels,
                    kernel_size=pool_kernel_size, stride=pool_stride,

                    # Resolves ambiguity about resulting size
                    output_padding=new_size - convt_size
                )
            )
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(prev_channels))
            if dropout != 0.0:
                modules.append(dropout_module)

            modules.append(
                nn.Conv1d(
                    prev_channels, new_channels,
                    kernel_size=conv_kernel_size, stride=conv_stride,
                    padding=conv_padding
                )
            )

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """Forward propagation"""

        return self.module(x)


class ConvAutoencoder(nn.Module):
    """ Convolutional autoencoder.
    """

    def __init__(
        self, input_size, conv_kernel_sizes, channels, pool_kernel_sizes,
        latent_space_size, conv_strides=None, pool_strides=None,
        pool_op=nn.MaxPool1d, dropout=0.0, activation=nn.ReLU,
    ):
        super().__init__()

        # Multiply the encoder output by 2 if using the VAE
        self.encoder = ConvEncoder(
            input_size, conv_kernel_sizes, channels, pool_kernel_sizes,
            latent_space_size,
            conv_strides=conv_strides,
            pool_strides=pool_strides, pool_op=pool_op, dropout=dropout,
            activation=activation
        )
        self.decoder = ConvDecoder(
            latent_space_size, list(reversed(conv_kernel_sizes)),
            list(reversed(channels)), list(reversed(pool_kernel_sizes)),
            input_size, conv_strides=conv_strides, pool_strides=pool_strides,
            dropout=dropout, activation=activation
        )
        self.latent_space_size = latent_space_size
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.decoder(self.encoder(x)))


class ConvVAE(nn.Module):
    """ Convolutional variational autoencoder.
    """
    def __init__(
            self, input_size, conv_kernel_sizes, channels, pool_kernel_sizes,
            latent_space_size, conv_strides=None, pool_strides=None,
            pool_op=nn.MaxPool1d, dropout=0.0, activation=nn.ReLU,
    ):
        super().__init__()

        # Multiply the encoder output by 2 if using the VAE
        self.encoder = ConvEncoder(
            input_size, conv_kernel_sizes, channels, pool_kernel_sizes,
            2 * latent_space_size,
            conv_strides=conv_strides,
            pool_strides=pool_strides, pool_op=pool_op, dropout=dropout,
            activation=activation
        )
        self.decoder = ConvDecoder(
            latent_space_size, list(reversed(conv_kernel_sizes)),
            list(reversed(channels)), list(reversed(pool_kernel_sizes)),
            input_size, conv_strides=conv_strides, pool_strides=pool_strides,
            dropout=dropout, activation=activation
        )
        self.latent_space_size = latent_space_size
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 2, self.latent_space_size)
        # Use first set of outputs as mean of distributions
        mu = x[:, 0, :]
        # And the second set as the log variances
        log_var = x[:, 1, :]
        z = reparameterize(mu, log_var)
        return self.softplus(self.decoder(z)), mu, log_var


def reparameterize(mu, log_var):
    """ Implements the 'reparameterization trick'
    """

    std = torch.exp(0.5*log_var)  # standard deviation
    eps = torch.randn_like(std)  # `randn_like` as we need the same size
    sample = mu + (eps * std)  # sampling as if coming from the input space
    return sample
