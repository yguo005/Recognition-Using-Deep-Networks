# Yunyu Guo
# CS5330 Project 5
# March 26 2025

# Analyze the first layer

import torch
import matplotlib.pyplot as plt
from network1 import MyNetwork
import cv2

def examine_network():
    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    # Print model structure
    print("Network Structure:")
    print(model)
    
    # Get first layer weights
    weights = model.conv1.weight
    print("\nFirst Layer Filter Shape:", weights.shape)
    
    # Create a figure with a 3x4 grid of subplots for the filters
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Flatten the axes for easier indexing
    axes = axes.flatten()
    
    # Plot each filter in its own subplot
    for i in range(10):  # 10 filters
        # Get the i-th filter
        filter_weights = weights[i, 0].detach().numpy() # detach(): Separates the tensor from its computation history, Returns a new tensor without gradients 
        
        # Plot in the corresponding subplot with viridis colormap
        im = axes[i].imshow(filter_weights, cmap='viridis')
        
        # Add title for each filter
        axes[i].set_title(f'Filter {i+1}')
        
        # Remove ticks for a cleaner look
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Turn off the unused subplots
    for i in range(10, 12):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('filters_color.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print filter weights
    print("\nFilter Weights:")
    for i in range(10):
        print(f"\nFilter {i+1}:")
        print(weights[i, 0].detach().numpy())
        print(weights[i, 0].detach().numpy())

# Show the effect of the filters on the first training example in task1
def show_filter_effects():
    # Load model and get filter weights
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    with torch.no_grad():
        weights = model.conv1.weight.detach().cpu().numpy()
    
    # Load the first training example
    train_loader, _ = load_data()
    images, _ = next(iter(train_loader))
    sample_image = images[0, 0].numpy()  # Get first image, first channel
    
    # Create a figure with a 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Show original image in first subplot
    axes[0].imshow(sample_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Apply each filter to the image and show results
    for i in range(10):
        current_filter = weights[i, 0]  # weights[i, 0]: The i-th filter (0-9), 0:the first input channel 
        
        # Apply filter using OpenCV's filter2D
        filtered_image = cv2.filter2D(sample_image, -1, current_filter)
        
        # Show filtered image in corresponding subplot
        axes[i+1].imshow(filtered_image, cmap='gray')
        axes[i+1].set_title(f'After Filter {i+1}')
        axes[i+1].set_xticks([])
        axes[i+1].set_yticks([])
    
    # Turn off the unused subplot
    axes[11].axis('off')
    
    plt.tight_layout()
    plt.savefig('filter_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    examine_network()
    show_filter_effects() 