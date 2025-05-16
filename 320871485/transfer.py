# Yunyu Guo
# CS5330 Project 5

# Transfer Learning for Greek Letters Recognition

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from network1 import MyNetwork, load_data
import os
import numpy as np
import shutil
from PIL import Image

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0) # resize (36/128 ≈ 0.28)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x) # white to black

def load_greek_data(data_path):
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )
    
    return greek_train

# Create a network for Greek letter recognition by modifying MNIST network
def create_greek_network():
    # Load the original MNIST model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last layer (fc2) with a new one for 3 classes
    # when creating a new PyTorch layer, its parameters are automatically initialized with requires_grad=True by default
    model.fc2 = nn.Linear(50, 3)  # Replace with 3 outputs for alpha, beta, gamma; 50 input features (matching the output of the previous layer), 3 output features 
    
    return model

# Train the network on Greek data
def train_greek_network(model, greek_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() 
    
    # Only optimize parameters of the last layer (which are not frozen)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=0.01, momentum=0.9) # lambda p: p.requires_grad: a function that returns True only for parameters where requires_grad is True
    # Since all original parameters were frozen with requires_grad=False and only the newly created fc2 layer has requires_grad=True by default, the optimizer receives only the fc2 parameters
    
    losses = []
    accuracies = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(greek_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(greek_loader)
        epoch_accuracy = 100 * correct / total
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # If accuracy is very high, consider early stopping
        if epoch_accuracy > 95 and epoch >= 5:
            print(f"High accuracy achieved at epoch {epoch+1}. Early stopping.")
            break
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses)+1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('greek_training_metrics.png')
    plt.close()
    
    return model, losses, accuracies

def test_greek_network(model, greek_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    class_names = ['alpha', 'beta', 'gamma']
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        for images, labels in greek_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # torch.max():finding the highest value and its position in each row of the model outputs
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Print overall accuracy
    print(f'Overall accuracy: {100 * correct / total:.2f}%')
    
    # Print per-class accuracy
    for i in range(3):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Accuracy for {class_names[i]}: {accuracy:.2f}%')
    
def main():
    training_set_path = './greek_train'
    
    # Create network for Greek letters
    print("Creating network for Greek letter recognition...")
    greek_model = create_greek_network()
    
    # Print model structure
    print("Network Structure:")
    print(greek_model)
    
    # Load Greek data
    print("Loading Greek letter data...")
    greek_loader = load_greek_data(training_set_path)
    
    # Train network
    print("Training network on Greek letters...")
    trained_model, losses, accuracies = train_greek_network(greek_model, greek_loader)
    
    # Test the trained model
    print("Testing network on Greek letters...")
    test_greek_network(trained_model, greek_loader)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'greek_model.pth')
    print("Model saved as 'greek_model.pth'")
    
    # Report how many epochs it took to converge
    convergence_epoch = len(losses)
    print(f"Training completed in {convergence_epoch} epochs")
    print(f"Final accuracy: {accuracies[-1]:.2f}%")

# Test the network on handwritten Greek letters
def test_handwritten_greek(model_path='greek_model.pth'):
    """Test handwritten Greek letters in the current directory"""
    # Load the trained model
    model = MyNetwork()
    model.fc2 = nn.Linear(50, 3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get all PNG files in current directory, excluding output files
    exclude_files = ['greek_training_metrics.png', 'filters_color.png', 
                     'greek_handwritten_results.png', 'filter_effects.png']
    
    handwritten_files = [f for f in os.listdir('.') 
                         if f.lower().endswith('.png') and f not in exclude_files]
    
    if not handwritten_files:
        print("No PNG files found in current directory.")
        return
    
    print(f"Found {len(handwritten_files)} files to test: {handwritten_files}")
    
    # Limit to 6 images for our 2×3 grid
    if len(handwritten_files) > 6:
        print(f"Only showing the first 6 of {len(handwritten_files)} images")
        handwritten_files = handwritten_files[:6]
    
    # Create a fixed 2×3 grid for the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    class_names = ['alpha', 'beta', 'gamma']
    processed_images = []
    predictions = []
    confidences = []
    
    # Process each image
    for idx, img_file in enumerate(handwritten_files):
        # Load and preprocess image
        original_img = Image.open(img_file).convert('RGB')
        
        # Resize to approximately 128x128 while preserving aspect ratio
        width, height = original_img.size
        scale = 128 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new image with white background and place resized image in center
        new_img = Image.new('RGB', (128, 128), color='white')
        paste_x = (128 - new_width) // 2
        paste_y = (128 - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        
        # Convert to tensor and apply Greek transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        input_tensor = transform(new_img).unsqueeze(0)
        
        # Save the processed image for display
        processed_img = input_tensor[0, 0].cpu().numpy()  # Extract grayscale channel
        processed_images.append(processed_img)
        
        input_tensor = input_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        predictions.append(prediction)
        confidences.append(confidence)
    
    # Display all processed images in the 2×3 grid
    for idx in range(len(handwritten_files)):
        row, col = idx // 3, idx % 3
        
        # Show processed image
        axes[row, col].imshow(processed_images[idx], cmap='gray')
        axes[row, col].set_title(f"Predicted: {class_names[predictions[idx]]}\n"
                                f"Confidence: {confidences[idx]:.1f}%\n"
                                f"File: {handwritten_files[idx]}")
        axes[row, col].axis('off')
    
    # Hide unused subplots if we have fewer than 6 images
    for idx in range(len(handwritten_files), 6):
        row, col = idx // 3, idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('greek_processed_grid.png')
    plt.show()
    print(f"Results saved as 'greek_processed_grid.png'")

def main():
    source_dir = '.'  # Directory with all PNG files
    training_set_path = './greek_train'  # Directory for organized images

    # Organize images first
    if organize_greek_images(source_dir, training_set_path):
        # Create network for Greek letters
        print("Creating network for Greek letter recognition...")
        greek_model = create_greek_network()
    
    
        # Print model structure
        print("Network Structure:")
        print(greek_model)
        
        # Load Greek data
        print("Loading Greek letter data...")
        greek_loader = load_greek_data(training_set_path)
        
        # Train network
        print("Training network on Greek letters...")
        trained_model, losses, accuracies = train_greek_network(greek_model, greek_loader)
        
        # Test the trained model
        print("Testing network on Greek letters...")
        test_greek_network(trained_model, greek_loader)
        
        # Save the trained model
        torch.save(trained_model.state_dict(), 'greek_model.pth')
        print("Model saved as 'greek_model.pth'")
        
        # Report how many epochs it took to converge
        convergence_epoch = len(losses)
        print(f"Training completed in {convergence_epoch} epochs")
        print(f"Final accuracy: {accuracies[-1]:.2f}%")
  
    test_handwritten_greek('greek_model.pth')
  
if __name__ == "__main__":
    main()


