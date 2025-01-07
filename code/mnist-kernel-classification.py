import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter dataset for binary classification
def filter_dataset(dataset, class_threshold=4):
    dataset.targets = (dataset.targets <= class_threshold).long()
    return dataset


train_dataset = filter_dataset(train_dataset)
test_dataset = filter_dataset(test_dataset)
"""
# Visualize in two dimensions
stencil_x = np.ones((28,28))
for k in range(28):
    stencil_x[k] = np.arange(28)
stencil_y = stencil_x.T
num_images = len(train_dataset)
phi1 = np.zeros(num_images)
phi2 = np.zeros(num_images)
labels = np.zeros(num_images)
for i, data_i in enumerate(train_dataset):
    image_i = data_i[0][0].numpy()
    sum_image_i = np.sum(image_i)
    phi1[i] = np.sum(stencil_x*image_i)/sum_image_i
    #phi2[i] = np.sum(stencil_y*image_i)/sum_image_i
    phi2[i] = np.argmax(image_i)
    labels[i] = data_i[1]


fig, ax = plt.subplots()
ax.scatter(phi1, phi2, c=labels, s=5.0)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.set_xlabel("center of mass - x", fontsize=24)
ax.set_ylabel("center of mass - y", fontsize=24)
"""
bias = 0
n_p = 0
n_m = 0
for i, data_i in enumerate(train_dataset):
    image_i = data_i[0][0].numpy()
    y_i = data_i[1]
    if y_i == 1:
        n_p += 1
    else:
        n_m += 1
#for i, data_i in enumerate(train_dataset):

     

"""

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1)
        self.bias = nn.Parameter(torch.tensor(0.0))
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x) + self.bias
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = BinaryClassifier().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            labels = labels.view(-1, 1)  # Reshape to match model output

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).long()
            correct += (predicted.view(-1) == labels).sum().item()
            total += labels.size(0)
        print(f'Accuracy: {100 * correct / total:.2f}%')

# Run training and evaluation
train()
evaluate()
"""