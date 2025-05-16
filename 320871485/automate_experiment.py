import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from itertools import product
import seaborn as sns

class ConfigurableNetwork(nn.Module):
    """
    Configurable network allowing multiple architecture variations
    """
    def __init__(self, 
                 conv1_filters=10,  # Number of filters in first conv layer
                 conv2_filters=20,  # Number of filters in second conv layer
                 fc1_units=50,      # Number of units in first FC layer
                 dropout_rate=0.25, # Dropout rate
                 fc_dropout=False): # Whether to add dropout after FC layer
        super(ConfigurableNetwork, self).__init__()
        
        # First conv layer with configurable filters
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        
        # Second conv layer with configurable filters
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_dropout_enabled = fc_dropout
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        # Calculate size for first FC layer based on conv filter counts
        # After 2 conv (5x5) and pool (2x2) layers: 28->12->4
        fc1_input_size = conv2_filters * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 10)

    def forward(self, x):
        # First conv block with pooling and ReLU
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block with dropout, pooling and ReLU
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.pool(self.relu(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # First fully connected layer with optional dropout
        x = self.relu(self.fc1(x))
        if self.fc_dropout_enabled:
            x = self.dropout2(x)
        
        # Final layer with log_softmax
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

def load_data(batch_size=128, sample_fraction=0.2):  # Add sample_fraction parameter
    """Load a subset of MNIST dataset"""
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
    
    # Subsample the training data
    if sample_fraction < 1.0:
        num_samples = int(len(train_dataset) * sample_fraction)
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Keep full test set for accurate evaluation
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def train_and_evaluate(model_config, epochs=3, batch_size=128, sample_fraction=0.2):
    """Train and evaluate a model with the given configuration"""
    # Record start time
    start_time = time.time()
    
    # Load subset of data
    train_loader, test_loader = load_data(batch_size, sample_fraction)
    
    # Create model
    model = ConfigurableNetwork(**model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Training metrics
    train_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Record training loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Record test accuracy
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={test_accuracy:.2f}%')
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Return metrics
    return {
        "config": model_config,
        "final_accuracy": test_accuracies[-1],
        "max_accuracy": max(test_accuracies),
        "final_loss": train_losses[-1],
        "training_time": total_time,
        "convergence_speed": test_accuracies[1] / test_accuracies[-1],  # Ratio of early to final accuracy, 
        # This value is then used as the benchmark to compare against when testing different parameter values, to see if changing a parameter improves performance over the baseline. 
        # test_accuracies[-1]: accessing the last value in the test_accuracies list, which contains the accuracy values recorded after each training epoch. Since the accuracies are appended to test_accuracies list sequentially during training, The -1 index gives the accuracy from the final epoch of training,which represents the model's performance after completing all training iterations.
        "train_losses": train_losses,
        "test_accuracies": test_accuracies
    }

def run_experiments():
    """Run a series of experiments with different configurations"""
    # Parameter ranges to explore
    param_ranges = {
        "conv1_filters": [8, 16, 32],      # Remove 24
        "conv2_filters": [16, 48, 64],     # Remove 32
        "fc1_units": [30, 70],             # Remove 50, 100
        "dropout_rate": [0.1, 0.4],        # Remove 0.25
        "fc_dropout": [False]              # Only test without FC dropout
    }
    
    # Storage for results
    results = []
    
    # Linear search strategy for each parameter
    # First, establish a baseline configuration
    baseline_config = {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "fc1_units": 50,
        "dropout_rate": 0.25,
        "fc_dropout": False
    }
    
    # Test baseline
    print("Testing baseline configuration...")
    baseline_result = train_and_evaluate(baseline_config, epochs=3)
    results.append(baseline_result)
    
    # Linear search for each parameter
    for param_name in param_ranges:
        best_value = baseline_config[param_name]
        best_accuracy = baseline_result["final_accuracy"] # baseline_result is a dictionary returned by the train_and_evaluate() function, The key "final_accuracy" contains the model's test accuracy on the last epoch
        
        print(f"\nOptimizing {param_name}...")
        # Loops through each possible value for the current parameter (e.g., if param_name is "conv1_filters", iterates through [8, 16, 32])
        for value in param_ranges[param_name]:
            if value == baseline_config[param_name]:
                continue  # Skip baseline value (already tested)
            
            # Create configuration with this parameter changed
            config = baseline_config.copy()
            config[param_name] = value # Changes only the specific parameter being optimized to the new test value
            
            # Test configuration
            result = train_and_evaluate(config, epochs=3)
            results.append(result)
            
            # Update best value if accuracy improves
            if result["final_accuracy"] > best_accuracy:
                best_value = value
                best_accuracy = result["final_accuracy"]
        
        # Update baseline with best value
        print(f"Best value for {param_name}: {best_value}")
        baseline_config[param_name] = best_value
    
    # Test some random combinations for additional exploration
    print("\nTesting random combinations...")
    num_random_tests = 10
    for _ in range(num_random_tests):
        random_config = {
            param: np.random.choice(param_ranges[param]) 
            for param in param_ranges
        }
        
        result = train_and_evaluate(random_config)
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # extract the individual configuration values from the 'config' dictionary and add them as columns to the DataFrame using pd.concat and apply(pd.Series).
    results_df = pd.concat([results_df.drop('config', axis=1), results_df['config'].apply(pd.Series)], axis=1)

    
    # Save results
    results_df.to_csv("mnist_experiment_results.csv", index=False)
    
    # Generate summary visualizations
    visualize_results(results_df)
    
    return results_df

def visualize_results(results_df):
    """Create visualizations of experiment results"""
    # Create output directory
    os.makedirs("experiment_results", exist_ok=True)
    
    # Plot 1: Parameter effects on accuracy
    plt.figure(figsize=(12, 8))
    
    for i, param in enumerate(["conv1_filters", "conv2_filters", "fc1_units", "dropout_rate"]):
        plt.subplot(2, 2, i+1)
        plt.scatter(results_df[param], results_df["final_accuracy"], alpha=0.7)
        plt.title(f"Effect of {param} on Accuracy")
        plt.xlabel(param)
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("experiment_results/parameter_effects.png", dpi=300)
    
    # Plot 2: Training time vs model complexity
    plt.figure(figsize=(10, 6))
    
    # Create a measure of model complexity (sum of parameters)
    results_df['model_complexity'] = results_df.apply(
        lambda row: row['conv1_filters'] + row['conv2_filters'] + row['fc1_units'], 
        axis=1
    )
    
    plt.scatter(results_df['model_complexity'], results_df['training_time'], alpha=0.7)
    plt.title("Training Time vs Model Complexity")
    plt.xlabel("Model Complexity (Sum of Parameters)")
    plt.ylabel("Training Time (seconds)")
    plt.grid(True, alpha=0.3)
    
    # Fit trendline
    z = np.polyfit(results_df['model_complexity'], results_df['training_time'], 1)
    p = np.poly1d(z)
    plt.plot(results_df['model_complexity'], p(results_df['model_complexity']), "r--")
    
    plt.tight_layout()
    plt.savefig("experiment_results/training_time.png", dpi=300)
    
    # Plot 3: Best performing configurations
    top_configs = results_df.sort_values("final_accuracy", ascending=False).head(5)
    
    fig, axes = plt.subplots(1, len(top_configs), figsize=(15, 5))
    for i, (_, row) in enumerate(top_configs.iterrows()):
        axes[i].plot(row["test_accuracies"])
        axes[i].set_title(f"Config {i+1}: {row['final_accuracy']:.2f}%")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Accuracy (%)")
        axes[i].grid(True, alpha=0.3)
        
        # Add configuration details
        config_text = "\n".join([
            f"Conv1: {int(row['conv1_filters'])}",
            f"Conv2: {int(row['conv2_filters'])}",
            f"FC: {int(row['fc1_units'])}",
            f"Dropout: {row['dropout_rate']:.2f}",
            f"FC Dropout: {row['fc_dropout']}"
        ])
        axes[i].text(0.5, 0.3, config_text, transform=axes[i].transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("experiment_results/best_configurations.png", dpi=300)

if __name__ == "__main__":
    results = run_experiments()
    
    # Print top 5 configurations
    top5 = results.sort_values("final_accuracy", ascending=False).head(5)
    print("\nTop 5 Configurations:")
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"{i+1}. Accuracy: {row['final_accuracy']:.2f}%")
        print(f"   Conv1 filters: {row['conv1_filters']}")
        print(f"   Conv2 filters: {row['conv2_filters']}")
        print(f"   FC units: {row['fc1_units']}")
        print(f"   Dropout rate: {row['dropout_rate']}")
        print(f"   FC dropout: {row['fc_dropout']}")
        print(f"   Training time: {row['training_time']:.2f} seconds")
        print()
