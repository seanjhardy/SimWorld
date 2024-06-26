import numpy as np
import math

from modules.io.encoders.ScalarEncoder import encode_scalar, decode_scalar


def encode_rgb(input, bits_per_channel=5, overlap=0.):
    length = input.shape[0]
    output = np.zeros(length * bits_per_channel)

    for i in range(length):
        output[i*bits_per_channel: (i + 1)*bits_per_channel] = encode_scalar(input[i], 0, 1, bits_per_channel, overlap)

    return output


def decode_rgb(input, bits_per_channel=5, overlap=0.):
    length = len(input) // bits_per_channel

    output = np.zeros(length)

    for i in range(length):
        output[i] = decode_scalar(input[i*bits_per_channel:(i+1)*bits_per_channel], 0, 1, bits_per_channel, overlap)

    return output
