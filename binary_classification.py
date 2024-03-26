import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images, self.labels = self.load_images()

    def load_images(self):
        images = []
        labels = []
        for label, category in enumerate(os.listdir(self.dataset_path)):
            category_path = os.path.join(self.dataset_path, category)
            for image_name in os.listdir(category_path):
                img_path = os.path.join(category_path, image_name)
                img = Image.open(img_path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
                # Label "Normal" class as 0, others as 1 (Cracked)
                labels.append(0 if category == "normal" else 1)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Define transforms
# Define data transformations for data augmentation and normalization
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300),  # Randomly crop and resize the image to 300x300
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
        transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image horizontally and vertically
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Randomly apply perspective transformation
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
    ])

val_transform = transforms.Compose([
    transforms.Resize(300),  # Resize the image to 300x300
    transforms.CenterCrop(300),  # Crop the center 300x300 region of the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])

# Define dataset paths
train_dataset_path = '/data/pavement/datasets/train/train'
val_dataset_path = '/data/pavement/datasets/val/val'
test_dataset_path = '/data/pavement/datasets/test/test'

# Create datasets
train_dataset = CustomDataset(train_dataset_path, transform=train_transform)
val_dataset = CustomDataset(val_dataset_path, transform=val_transform)
test_dataset = CustomDataset(test_dataset_path, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model
NUM_CLASSES = 2  # Binary classification: Normal or Cracked
model = models.efficientnet_b3(pretrained=False, num_classes=NUM_CLASSES)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Add weight decay for regularization

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce learning rate by a factor of 0.1 every 5 epochs

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Step the scheduler
    scheduler.step()

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print("Test Accuracy:", test_accuracy)

# Optionally, test the model on unseen data
img_path = '9.jpg'
img = Image.open(img_path).convert("RGB")
img = val_transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    print("Predicted label:", predicted.item())

