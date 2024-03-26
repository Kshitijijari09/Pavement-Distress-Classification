import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    latent_dim = 100
    img_shape = (1, 300, 300)  # Grayscale image shape
    num_epochs = 200
    batch_size = 64
    lr = 0.0002

    # Initialize generator and discriminator
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Create directory to save generated images
    os.makedirs("generated_images", exist_ok=True)

    # Load dataset
    dataset = datasets.ImageFolder(root="/data/pavement/datasets/train/train", transform=transforms.Compose([
        transforms.Resize((300, 300)),  # Resize images to 300x300
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # Adversarial ground truths and fake images
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            z = torch.randn(imgs.size(0), latent_dim).to(device)

            # Train generator
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs.to(device)), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Print progress
            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(25, latent_dim).to(device)
                gen_imgs = generator(z)
                gen_imgs = gen_imgs * 0.5 + 0.5  # Denormalize images
                for j in range(gen_imgs.shape[0]):
                    label = labels[j].item()
                    class_folder = os.path.join("generated_images", str(label))
                    os.makedirs(class_folder, exist_ok=True)
                    img_path = os.path.join(class_folder, f"{epoch}_{j}.png")
                    utils.save_image(gen_imgs[j], img_path)

if __name__ == "__main__":
    main()

