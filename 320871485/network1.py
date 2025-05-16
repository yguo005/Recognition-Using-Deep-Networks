#!/usr/bin/env python3
# MNIST Digit Recognition using PyTorch
# This program implements a convolutional neural network to recognize handwritten digits
# from the MNIST dataset using PyTorch.

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # First conv block: 1 input channel (grayscale), 10 output channels, 5x5 kernel sliding window
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # Second conv block: 10 input channels, 20 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # Other layers
        self.dropout = nn.Dropout(0.25)  # 25% dropout rate
        self.pool = nn.MaxPool2d(2)  # 2x2 max pooling
        self.relu = nn.ReLU() # ReLU activation function
        
        # Calculate input size for first linear layer
        # After 2 conv (5x5) and pool (2x2) layers: 28->12->6->3->1
        self.fc1 = nn.Linear(320, 50)  # 320 = 20 channels * 4 * 4
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # First conv block with pooling and ReLU
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block with dropout, pooling and ReLU
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(self.relu(x))
        
        # Flatten the output for the linear layers
        x = x.view(-1, 320)
        
        # First fully connected layer with ReLU
        x = self.relu(self.fc1(x))
        
        # Final layer with log_softmax
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

def load_data():
    """Load and prepare MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) 
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def train_network(model, train_loader, test_loader, num_epochs=5):
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = model.to(device)
    
    # Loss function and optimizer
    # Loss Function: how wrong the predictions are
    # Optimizer: how to improve the parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # Stochastic Gradient Descent: Gets all trainable parameters; lr=0.01 (Learning Rate): Controls how much to adjust weights in each step; momentum=0.5: helps accelerate SGD in relevant direction
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training evaluation happens during training loops with gradient computation enabled
        # Training results are printed per batch and epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass:
            optimizer.zero_grad() # zeroes out the gradients of all parameters before each batch because PyTorch accumulates gradients by default
            outputs = model(images)
            loss = criterion(outputs, labels) # outputs: scores for each digit class (0-9); labels: Ground truth digit values (0-9)
            
            # Backward pass and optimize:
            # Backward Pass: computes the gradient of the loss with respect to the model parameters
            # Gradients: the rate of change of the loss with respect to each model parameter 
            # Optimization Step: updates the model parameters using the computed gradients: SGD update rule: parameter = parameter - learning_rate * gradient
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        # Evaluate on training set
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Training Loss: {train_loss:.3f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%')
    
    # Plot training and test metrics
    # First plot: Training and test metrics over epochs
    fig1 = plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('accuracy_results.png')
    plt.close()

    # Second plot: Loss over number of training examples
    fig2 = plt.figure(figsize=(10, 6))
    
    # Calculate training examples seen
    train_counter = [(i*len(train_loader.dataset)) for i in range(1, num_epochs + 1)]
    test_counter = [(i*len(train_loader.dataset)) for i in range(1, num_epochs + 1)]
    
    # Plot combined losses
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.tight_layout()
    plt.savefig('loss_results.png')
    plt.close()
    
    print('Finished Training')
    return model

# Final model evaluation
def test_network(model, test_loader):
    """Test the neural network"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Is used during training to track progress
# Works for both training and test sets
def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on given data loader"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def display_sample_digits(test_loader):
    """Display the first 6 digits from the test set"""
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(10, 4))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {labels[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('sample_digits.png')
    plt.close()

def main(argv):
    """Main function to run the MNIST digit recognition program"""
    # Load data
    train_loader, test_loader = load_data()
    
    # Display sample digits
    display_sample_digits(test_loader)
    
    # Create and train the model
    model = MyNetwork()
    model = train_network(model, train_loader, test_loader)
    
    # Final test evaluation
    test_network(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')

if __name__ == "__main__":
    main(sys.argv)
