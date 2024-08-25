import unittest

import torch

from models.vae import Encoder

# Test the Encoder class
class TestEncoder(unittest.TestCase):
  def test_encoder(self):
    img_channels = 3
    latent_size = 32
    encoder = Encoder(img_channels, latent_size)
    self.assertIsNotNone(encoder)
  
  def test_encoder_forward(self):
    img_channels = 3
    latent_size = 32
    encoder = Encoder(img_channels, latent_size)
    x = torch.randn(1, img_channels, 64, 64)
    mu, logsigma = encoder(x)
    self.assertEqual(mu.size(), (1, latent_size))
    self.assertEqual(logsigma.size(), (1, latent_size))

  def test_encoder_forward_batch(self):
    img_channels = 3
    latent_size = 32
    encoder = Encoder(img_channels, latent_size)
    # Test the forward pass in batch mode
    x = torch.randn(4, img_channels, 64, 64)
    mu, logsigma = encoder(x)
    self.assertEqual(mu.size(), (4, latent_size))
    self.assertEqual(logsigma.size(), (4, latent_size))


# Run the tests
if __name__ == '__main__':
  unittest.main()
