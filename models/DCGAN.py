def train_dcgan(dataloader, num_epochs, latent_dim, lr, beta1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(latent_dim, channels=3).to(device)
    netD = Discriminator(channels=3).to(device)
    
    criterion = nn.BCELoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
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
            # Update G network: maximize log(D(G(z)))
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
            torchvision.utils.save_image(fake, f'fake_samples_epoch_{epoch}.png', normalize=True)

    return netG, netD