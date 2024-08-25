"""
  Variational Autoencoder (VAE) model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
  def __init__(self, img_channels, latent_size):
    super(Encoder, self).__init__()

    self._img_channels = img_channels
    self._latent_size = latent_size

    self._conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=4, stride=2) 
    self._conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
    self._conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
    self._conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

    self._fc_mu = nn.Linear(2*2*256, latent_size)
    self._fc_logsigma = nn.Linear(2*2*256, latent_size)

  def forward(self, x):
    x = F.relu(self._conv1(x))
    x = F.relu(self._conv2(x))
    x = F.relu(self._conv3(x))
    x = F.relu(self._conv4(x))

    x = x.view(x.size(0), -1)

    mu = self._fc_mu(x)
    logsigma = self._fc_logsigma(x)

    return mu, logsigma

class Decoder(nn.Module):
  def __init__(self, img_channels, latent_size):
    super(Decoder, self).__init__()

    self._img_channels = img_channels
    self._latent_size = latent_size
