# ArtBench-10 GAN

This project implements a Generative Adversarial Network (GAN) to generate artistic images using the [ArtBench-10](https://artbench.eecs.berkeley.edu) dataset. The dataset consists of 10 categories of artworks, and this project builds on the `torchvision.datasets.CIFAR10` class to create a custom loader for `ArtBench-10`. The aim of the project is to experiment with image generation, improving the ability of GANs to create convincing artistic styles.

## Project Features

- **GAN Model Implementation**: A standard Deep Convolutional GAN (DCGAN) is implemented for generating new artwork images based on the ArtBench-10 dataset.
- **Training Process**: The training process includes detailed logging of the losses for both the generator and discriminator, along with saving generated image samples at each epoch.
- **Support for Multiple Image Datasets**: While it works primarily with ArtBench-10, the architecture can be adapted for other datasets like CIFAR-10 with minimal changes.
- **Image Transformations**: Includes options for preprocessing and normalizing images to ensure the GAN receives standardized input.
- **Save and Load Functionality**: Checkpoints of the model are saved during training for later reuse or fine-tuning.

## Dataset

The [ArtBench-10](https://artbench.eecs.berkeley.edu) dataset contains artwork images across 10 categories:
- Categories: Baroque, Impressionism, Renaissance, Cubism, Surrealism, etc.
- The dataset consists of low-resolution 32x32 images similar in structure to CIFAR-10, which makes it compatible with existing deep learning pipelines.

## Usage

1. Clone the repository
2. `pip install -r requirements.txt`
3. `python scripts/train_gan.py
4. Check results folder for generated images
5. check home directory for saved/trained generator and discriminator   
