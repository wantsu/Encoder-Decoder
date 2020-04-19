from modules.Encoder import *
from modules.Decoder import *

import torch.nn as nn
import torch

import numpy as np

class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.encoder = Encoder(input_size)
        self.decoder = Decoder(output_size)

    def forward(self, input):
        output, state = self.encoder(input)
        output, softmax, state = self.decoder(output, state)
        return output