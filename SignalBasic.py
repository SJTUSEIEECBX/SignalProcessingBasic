# This is a pack of basic signal generation and processing algorithms. Support pytorch and numpy, support cuda.
# Author: Boxuan Chang

import numpy as np
import torch


class Signal:
    def __init__(self, length, batch_size=1, device='cpu'):
        self.data = torch.empty(size=[batch_size, length], device=device)
        self.length = length
        self.batch_size = batch_size
        self.device = device
        self.size = [batch_size, length]

    def __str__(self):
        return 'Signal of size {}*{}. Device: {}'\
            .format(self.batch_size, self.length, self.device)


class BinarySignal(Signal):
    def __init__(self, length, batch_size=1, is_sparse=False, device='cpu'):
        super(BinarySignal, self).__init__(length, batch_size, device)
        self.is_sparse = is_sparse

    def random_generate(self, p=0.5):
        if not self.is_sparse:
            self.data = torch.randint(0, 2, self.size, dtype=torch.uint8, device=self.device)
        else:
            tmp = torch.rand(self.size)
            tmp[tmp < (1 - p)] = 0
            tmp[tmp > (1 - p)] = 1
            self.data = tmp.to(device=self.device, dtype=torch.uint8)

    def ones(self):
        self.data = torch.ones(size=self.size, dtype=torch.uint8, device=self.device)

    def zeros(self):
        self.data = torch.zeros(size=self.size, dtype=torch.uint8, device=self.device)

    def ones_ratio(self):
        return self.data.sum().float() / (self.batch_size * self.length)


if __name__ == '__main__':
    signal = BinarySignal(100, 100, is_sparse=True, device='cuda')
    signal.random_generate(0.2)
    print(signal.ones_ratio())
