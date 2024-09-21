import sys
import os

# Add the root directory (Creative-Art-GAN) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from artbench10 import ArtBench10


def train_dcgan(dataloader, num_epochs, latent_dim, lr, beta1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(latent_dim=latent_dim, channels=3).to(device)
    netD = Discriminator(channels=3).to(device)
    
    criterion = nn.BCELoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Update D network
            ###########################
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            
            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Update G network
            ###########################
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

        # Save generated images
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            torchvision.utils.save_image(fake, f'results/fake_samples_epoch_{epoch}.png', normalize=True)

    return netG, netD

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    image_size = 64
    num_epochs = 5
    lr = 0.0001
    beta1 = 0.5
    latent_dim = 100

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # CIFAR10
    #dataset = torchvision.datasets.CIFAR10(root='../data', download=True, transform=transform)
    # ARTBENCH10
    dataset = ArtBench10(root='../data', download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Train the model
    generator, discriminator = train_dcgan(dataloader, num_epochs, latent_dim, lr, beta1)

    # Save the trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
