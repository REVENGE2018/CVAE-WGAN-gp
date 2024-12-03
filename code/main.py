import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, num_classes=3, label_embedding_dim=16):
        super(VAEEncoder, self).__init__()
        self.label_emb = nn.Embedding(num_classes, label_embedding_dim)
        self.conv1 = nn.Conv3d(in_channels + label_embedding_dim, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_mu = nn.Conv3d(64, latent_dim, kernel_size=4, stride=1, padding=0)
        self.conv4_log_var = nn.Conv3d(64, latent_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x, label_embedding):
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        label_embedding = label_embedding.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        x = torch.cat([x, label_embedding], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        mu = self.conv4_mu(x)
        log_var = self.conv4_log_var(x)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=32, out_channels=1, num_classes=3, label_embedding_dim=16):
        super(VAEDecoder, self).__init__()
        self.label_emb = nn.Embedding(num_classes, label_embedding_dim)
        self.conv_trans1 = nn.ConvTranspose3d(latent_dim + label_embedding_dim, 32, kernel_size=4, stride=1, padding=0)
        self.conv_trans2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose3d(16, 4, kernel_size=3, stride=1, padding=1)
        self.conv_trans4 = nn.ConvTranspose3d(4, out_channels, kernel_size=4, stride=2, padding=1)
        self.output_layer = nn.Sigmoid()

    def forward(self, z, label_embedding):
        
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        label_embedding = label_embedding.expand(-1, -1, z.shape[2], z.shape[3], z.shape[4])  
        z = torch.cat([z, label_embedding], dim=1)
        z = F.gelu(self.conv_trans1(z))
        z = F.gelu(self.conv_trans2(z))
        z = F.gelu(self.conv_trans3(z))
        z = self.conv_trans4(z)
        return self.output_layer(z)


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, out_channels=1, num_classes=3, label_embedding_dim=16):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, num_classes, label_embedding_dim)
        self.decoder = VAEDecoder(latent_dim, out_channels, num_classes, label_embedding_dim)

    def forward(self, x, label_embedding):
        mu, log_var = self.encoder(x, label_embedding)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, label_embedding)
        return recon_x, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEGenerator(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, latent_dim=32, num_classes=3, label_embedding_dim=16):
        super(VAEGenerator, self).__init__()
        self.vae = VAE(in_channel, latent_dim, out_channel, num_classes, label_embedding_dim)

    def forward(self, x, label_embedding):
        recon_x, mu, log_var = self.vae(x, label_embedding)
        return recon_x, mu, log_var



class Discriminator(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_classes=3, label_embedding_dim=16):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, label_embedding_dim)
        self.conv_net = nn.Sequential(
            
            nn.Conv3d(in_channels=in_channel + label_embedding_dim, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2),

            nn.Conv3d(16, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.InstanceNorm3d(4),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(4, out_channel, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, z, label_embedding):
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        label_embedding = label_embedding.expand(-1, -1, z.shape[2], z.shape[3], z.shape[4])
        z = torch.cat([z, label_embedding], dim=1)
        return self.conv_net(z)


def gradient_penalty(critic, real, fake, label_embedding, device="cpu"):
    BATCH_SIZE, C, H, W, D = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1, 1), device=device)
    beta = beta.expand_as(real)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, label_embedding)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)

class SubpatchDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.indices = []

        for file_idx, (file_path, label) in enumerate(zip(file_paths, labels)):
            
            data_shape = np.load(file_path, mmap_mode='r').shape
            num_samples = data_shape[0]
            self.indices.extend([(file_idx, i) for i in range(num_samples)])

        self.label_dict = dict(enumerate(labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.indices[idx]
        file_path = self.file_paths[file_idx]
        label = self.label_dict[file_idx]
        
        data = np.load(file_path, mmap_mode='r')[sample_idx]
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def prepare_data():
    
    subpatch_files = [
        '../data/2_subpatches.npy',
        '../data/5_subpatches.npy',
        '../data/6_subpatches.npy'
    ]
    labels = [0, 1, 2]  
    return subpatch_files, labels


if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

subpatch_files, labels = prepare_data()

dataset = SubpatchDataset(subpatch_files, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

num_epochs = 100
latent_dim = 32  
num_classes = 3
label_embedding_dim = 16  
lr = 2e-4
lambda_gp = 5
CRITIC_ITERATIONS = 2

vae_generator = VAEGenerator(in_channel=1, out_channel=1, latent_dim=latent_dim, num_classes=num_classes,
                             label_embedding_dim=label_embedding_dim).to(device)
discriminator = Discriminator(in_channel=1, out_channel=1, num_classes=num_classes, label_embedding_dim=label_embedding_dim).to(device)

optimizer_G = torch.optim.Adam(vae_generator.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, 4 * num_epochs)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=5*lr, betas=(0.5, 0.999))
scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, 4 * num_epochs * CRITIC_ITERATIONS)

training_history = pd.DataFrame(columns=['epoch', 'generator_loss', 'discriminator_loss'])

for epoch in range(num_epochs):
    generator_loss_epoch = 0.0  
    discriminator_loss_epoch = 0.0  
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for i, (x, y) in enumerate(tepoch):
            
            x = x.to(device)
            y = y.to(device)

            label_embedding_layer = vae_generator.vae.encoder.label_emb
            label_embedding = label_embedding_layer(y)

            for _ in range(CRITIC_ITERATIONS):
    
                recon_x, mu, log_var = vae_generator(x, label_embedding)
                fake_images = recon_x
    
                
                label_embedding_disc = discriminator.label_emb(y)
    
                
                real_validity = discriminator(x, label_embedding_disc)
                
                fake_validity = discriminator(fake_images, label_embedding_disc)
    
                
                gp = gradient_penalty(discriminator, x, fake_images, label_embedding_disc, device=device)
    
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
    
                optimizer_D.zero_grad()
                d_loss.backward(retain_graph=True)
                optimizer_D.step()
                
                discriminator_loss_epoch += d_loss.item()

            label_embedding = vae_generator.vae.encoder.label_emb(y)
            label_embedding_disc = discriminator.label_emb(y)

            
            recon_x, mu, log_var = vae_generator(x, label_embedding)
            fake_validity = discriminator(recon_x, label_embedding_disc)

            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
            
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            
            g_adv = -torch.mean(fake_validity)
            g_loss = recon_loss + kl_loss + g_adv

            optimizer_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_generator.parameters(), max_norm=1.0)  
            optimizer_G.step()

            
            generator_loss_epoch += g_loss.item()
            
            
            tepoch.set_postfix(loss_G=g_loss.item(), loss_D=d_loss.item())

        scheduler_gen.step()
        scheduler_critic.step()

        generator_loss_epoch /= len(dataloader)
        discriminator_loss_epoch /= len(dataloader)
        
        new_row = pd.DataFrame({
            'epoch': [epoch + 1],
            'generator_loss': [generator_loss_epoch / len(dataloader)],
            'discriminator_loss': [discriminator_loss_epoch / len(dataloader)]
        })
        training_history = pd.concat([training_history, new_row], ignore_index=True)
        
        training_history.to_csv('training_history.csv', index=False)

        if (epoch + 1) % 5 == 0:
            torch.save(vae_generator.state_dict(), f'vae_generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')
    
    with torch.no_grad():
        vae_generator.eval()
        
        real_data, y_real = next(iter(dataloader))
        real_data = real_data.to(device)
        y_real = y_real.to(device)
        batch_size_real = real_data.size(0)

        label_embedding_real = vae_generator.vae.encoder.label_emb(y_real)

        mu_real, log_var_real = vae_generator.vae.encoder(real_data, label_embedding_real)
        z_real = vae_generator.vae.reparameterize(mu_real, log_var_real)

        generated_samples = vae_generator.vae.decoder(z_real, label_embedding_real)

        slice_index = generated_samples.shape[2] // 2  
        generated_sample = generated_samples[0, 0, slice_index, :, :].detach().cpu().numpy()

        output_path = f"generated_images/epoch_{epoch+1}.png"
        plt.imsave(output_path, generated_sample, cmap='gray')

        vae_generator.train()  

training_history.to_csv('training_history.csv', index=False)

torch.save(vae_generator.state_dict(), 'vae_generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

