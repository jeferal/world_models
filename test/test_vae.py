import unittest

import torch

from models.vae import Encoder, Decoder, VAE

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

# Test the Decoder class
class TestDecoder(unittest.TestCase):
  def test_decoder(self):
    img_channels = 3
    latent_size = 32
    decoder = Decoder(img_channels, latent_size)
    self.assertIsNotNone(decoder)
  
  def test_decoder_forward(self):
    img_channels = 3
    latent_size = 32
    decoder = Decoder(img_channels, latent_size)
    x = torch.randn(1, latent_size)
    reconstruction = decoder(x)
    self.assertEqual(reconstruction.size(), (1, img_channels, 64, 64))

  def test_decoder_forward_batch(self):
    img_channels = 3
    latent_size = 32
    decoder = Decoder(img_channels, latent_size)
    # Test the forward pass in batch mode
    x = torch.randn(4, latent_size)
    reconstruction = decoder(x)
    self.assertEqual(reconstruction.size(), (4, img_channels, 64, 64))

# Test the VAE class
class TestVAE(unittest.TestCase):
  def test_vae(self):
    img_channels = 3
    latent_size = 32
    vae = VAE(img_channels, latent_size)
    self.assertIsNotNone(vae)
  
  def test_vae_forward(self):
    img_channels = 3
    latent_size = 32
    vae = VAE(img_channels, latent_size)
    x = torch.randn(1, img_channels, 64, 64)
    reconstruction, mu, logsigma = vae(x)
    self.assertEqual(reconstruction.size(), (1, img_channels, 64, 64))
    self.assertEqual(mu.size(), (1, latent_size))
    self.assertEqual(logsigma.size(), (1, latent_size))

  def test_vae_forward_batch(self):
    img_channels = 3
    latent_size = 32
    vae = VAE(img_channels, latent_size)
    # Test the forward pass in batch mode
    x = torch.randn(4, img_channels, 64, 64)
    reconstruction, mu, logsigma = vae(x)
    self.assertEqual(reconstruction.size(), (4, img_channels, 64, 64))
    self.assertEqual(mu.size(), (4, latent_size))
    self.assertEqual(logsigma.size(), (4, latent_size))


# Run the tests
if __name__ == '__main__':
  unittest.main()
