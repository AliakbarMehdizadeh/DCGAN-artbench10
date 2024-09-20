if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    image_size = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    latent_dim = 100

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(root='../data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Train the model
    generator, discriminator = train_dcgan(dataloader, num_epochs, latent_dim, lr, beta1)

    # Save the trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')