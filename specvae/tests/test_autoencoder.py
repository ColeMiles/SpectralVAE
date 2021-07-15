import torch

from specvae.models.autoencoder import ConvAutoencoder, Reshape


class TestArrayAutoencoders:

    def test_convautoencoder_sizes(self):
        input_size = 100
        conv_kernel_sizes = [3, 3, 3, 3]
        channels = [1, 2, 4, 8]
        pool_kernel_sizes = [2, 2, 2, 2]
        latent_space_size = 8

        model = ConvAutoencoder(input_size, conv_kernel_sizes, channels, pool_kernel_sizes,
                                latent_space_size)

        x = torch.ones(1, 1, 100)

        stage_shapes = [x.shape]

        # Record the size of each encoder stage
        for layer in model.encoder.module:
            x = layer(x)
            # After pooling/linear, we should be at the next set of (size, channel), record
            if isinstance(layer, torch.nn.MaxPool1d):
                stage_shapes.append(x.shape)

        # Run through the decoder, checking that the shape at each stage matches the encoder
        i = len(stage_shapes) - 1
        for layer in model.decoder.module:
            x = layer(x)
            if isinstance(layer, (Reshape, torch.nn.Conv1d)):
                assert x.shape == stage_shapes[i]
                i -= 1
