import torch
import matplotlib.pyplot as plt
from network1 import MyNetwork, load_data
import os
import cv2
import numpy as np


# Evaluate first 10 test samples and plot first 9
def evaluate_and_plot():
    # Load model and data
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    # Get test data using existing load_data function
    _, test_loader = load_data()  # Only need test_loader
    
    # Get first batch of test data
    images, labels = next(iter(test_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = images.to(device), labels.to(device)
    model = model.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.exp(outputs) # converts the model's log-probability outputs into actual probabilities
        
        # Print first 10 results
        for i in range(10):
            probs = probabilities[i].cpu().numpy()
            predicted = torch.argmax(outputs[i]).item()
            actual = labels[i].item()
            
            print(f"\nSample {i+1}:")
            print(f"Output values: {', '.join([f'{p:.2f}' for p in probs])}")
            print(f"Predicted: {predicted}, Actual: {actual}")
            print(f"{'Correct' if predicted == actual else 'Wrong'}")
    
        # Plot first 9 samples in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i in range(9):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Display the image
            ax.imshow(images[i].cpu()[0], cmap='gray', interpolation='none')
            predicted = torch.argmax(outputs[i]).item()
            ax.set_title(f'Prediction: {predicted}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_predictions.png')
        plt.close()

# Test the network on hand writing digits new inputs
def test_custom_images():
    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get mean and std from MNIST dataset for consistent normalization
    _, test_loader = load_data()
    
    # Get file list - only process image files
    img_extensions = ['.png']
    image_files = [
        f for f in os.listdir('.') 
        if os.path.splitext(f)[1].lower() in img_extensions 
        and not f.startswith('processed_')  # Skip previously processed images
    ]
    
    if not image_files:
        print("No image files found in the directory.")
        return

    print(f"Found {len(image_files)} images to process: {image_files}")
    
    predictions = []
    processed_images = []
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        
        # Read image
        img = cv2.imread(img_file) # imread(): returns images as NumPy arrays
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if img.shape != (28, 28):
            print(f"Warning: Image {img_file} is not 28x28, dimensions: {img.shape}")
            # Only resize if absolutely needed
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Calculate mean before any processing
        initial_mean = np.mean(img)
        print(f"Initial mean pixel value: {initial_mean:.2f}")
        
        # Apply adaptive thresholding instead of fixed threshold
        # This works better with varying lighting conditions
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        
        # Check if image needs inversion (MNIST has white digits on black background)
        # Calculate mean pixel value to determine background: 0 = black, 255 = white
        mean_after_threshold = np.mean(img)
        print(f"Mean after thresholding: {mean_after_threshold:.2f}")

        # Likely black digits on white background, so invert
        if mean_after_threshold > 127:
            print("Inverting image (making white digits on black background)")
            img = 255 - img
        
        # Display original processed image
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f"Processed: {img_file}")
        plt.savefig(f"processed_{img_file}")
        plt.close()
            
        # Convert to tensor and normalize
        img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0 # torch.FloatTensor(img): transforms numpy array to PyTorch tensor; .unsqueeze(0).unsqueeze(0): first .unsqueeze(0): adds batch dimension, second .unsqueeze(0): adds channel dimension, Changes shape from [28, 28] → [1, 1, 28, 28]; / 255.0: Scales from integer range [0, 255] to float range [0, 1]
        # Apply same normalization as MNIST dataset
        img_tensor = (img_tensor - 0.1307) / 0.3081
        
        # Move to device
        img_tensor = img_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.exp(outputs)
            predicted = torch.argmax(outputs).item()
            
        # Save results
        predictions.append(predicted)
        processed_images.append(img)
        
        # Print results
        probs = probabilities[0].cpu().numpy()
        print(f"Prediction for {img_file}: {predicted}")
        print(f"Confidence scores: {', '.join([f'{p:.2f}' for p in probs])}")
        
    # Plot all processed images with predictions
    num_images = len(image_files)
    if num_images > 0:
        rows = (num_images + 2) // 3  # Calculate rows needed to display all images in 3 columns grid
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if num_images == 1 else axes
        
        for i, (img, pred, filename) in enumerate(zip(processed_images, predictions, image_files)):
            if i < len(axes):
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"File: {filename}\nPredicted: {pred}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig('custom_predictions.png')
        plt.close()

if __name__ == "__main__":
    evaluate_and_plot()
    test_custom_images()