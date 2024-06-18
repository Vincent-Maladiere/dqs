import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
from dqs.torch.distribution import DistributionLinear
from dqs.torch.loss import NegativeLogLikelihood


class MLP(nn.Module):
    def __init__(self, input_len, n_output):
        super(MLP,self).__init__()

        num_neuron = 128
        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, n_output)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim = 1)

if __name__ == '__main__':
    # prepare data
    from time import time

    tic = time()
    x = np.random.rand(100_000,2000)
    y = np.random.rand(100_000)
    e = (np.random.rand(100_000) > 0.5)

    # from hazardous.data._competing_weibull import make_synthetic_competing_weibull

    # x, y_ = make_synthetic_competing_weibull(
    #     n_events=1,
    #     n_samples=1000,
    #     n_features=20,
    #     return_X_y=True,
    #     complex_features=True,
    # )
    # x = x.values
    # y = y_["duration"].values.ravel()
    # e = y_["event"].values.ravel()

    data_x = torch.from_numpy(x.astype(np.float32)).clone()
    data_y = torch.from_numpy(y.astype(np.float32)).clone()
    data_e = torch.from_numpy(e).clone()

    # prepare model
    boundaries = torch.linspace(0.0, 1, 5)
    dist = DistributionLinear(boundaries)
    loss_fn = NegativeLogLikelihood(dist, boundaries)
    mlp = MLP(2000, 4)

    # train model
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(100):
        pred = mlp(data_x)
        loss = loss_fn.loss(pred, data_y, data_e)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch=%d, loss=%f' % (epoch,loss))
    toc = time()
    print(toc - tic)