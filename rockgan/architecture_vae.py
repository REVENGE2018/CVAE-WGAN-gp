import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm 
from torchsummary import summary
import torch.nn.functional as F

         

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)   
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)           
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)          
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)         
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)         
        self.conv6_mu = nn.Conv3d(512, latent_dim, kernel_size=4, stride=1, padding=0)       
        self.conv6_log_var = nn.Conv3d(512, latent_dim, kernel_size=4, stride=1, padding=0) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        mu = self.conv6_mu(x)
        log_var = self.conv6_log_var(x)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, out_channels=1):
        super(VAEDecoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose3d(latent_dim, 512, kernel_size=4, stride=1, padding=0)   
        self.conv_trans2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)          
        self.conv_trans3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)          
        self.conv_trans4 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)           
        self.conv_trans5 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)            
        self.conv_trans6 = nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1)  
        self.output_layer = nn.Sigmoid()

    def forward(self, z):
        z = F.gelu(self.conv_trans1(z))
        z = F.gelu(self.conv_trans2(z))
        z = F.gelu(self.conv_trans3(z))
        z = F.gelu(self.conv_trans4(z))
        z = F.gelu(self.conv_trans5(z))
        z = self.conv_trans6(z)
        return self.output_layer(z)


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, out_channels=1):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, out_channels)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEGenerator(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, latent_dim=64):
        super(VAEGenerator, self).__init__()
        self.vae = VAE(in_channel, latent_dim, out_channel)

    def forward(self, x):
        recon_x, mu, log_var = self.vae(x)
        return recon_x, mu, log_var


class Discriminator(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        
        super(Discriminator, self).__init__()
        
        
        
        self.conv_net = torch.nn.Sequential(

            
            torch.nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            
            
            torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(16),
            torch.nn.LeakyReLU(0.2),
            
            
            torch.nn.Conv3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(8),
            torch.nn.LeakyReLU(0.2),
            
            
            torch.nn.Conv3d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(4),
            torch.nn.LeakyReLU(0.2),
            
            
            torch.nn.Conv3d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(2),
            torch.nn.LeakyReLU(0.2),
            
            
            torch.nn.Conv3d(in_channels=2, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, z):
        
        return self.conv_net(z)


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W, D = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, H, W, D).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)

