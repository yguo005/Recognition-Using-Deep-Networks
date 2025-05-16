# Project 5: Recognition using Deep Networks


## Development Environment
- **Programming Language:** Python 3.x
- **Key Libraries:**
    - PyTorch (torch): [Specify version, e.g., 1.13.1]
    - Torchvision: [Specify version, e.g., 0.14.1]
    - Matplotlib: [Specify version, e.g., 3.6.2]
    - OpenCV (cv2): [Specify version, e.g., 4.6.0] (primarily for Task 2B - filter2D, and potentially image loading/preprocessing)
    - NumPy: [Specify version, e.g., 1.23.5]
    - ImageMagick (if used for Task 1F preprocessing - mention system-level tool)

---

## Project Overview
This project focuses on building, training, analyzing, and modifying deep neural networks for recognition tasks using PyTorch. Key tasks include:
1.  **MNIST Digit Recognition:** Building and training a Convolutional Neural Network (CNN) to recognize MNIST handwritten digits, saving the model, evaluating it on the test set, and testing it with new handwritten digit inputs.
2.  **Network Examination:** Analyzing the learned filters of the first convolutional layer and their effect on an example image.
3.  **Transfer Learning:** Adapting the pre-trained MNIST digit recognizer to classify Greek letters (alpha, beta, gamma) by freezing most weights and retraining a new final layer.
4.  **Experiment Design:** Planning and executing experiments to evaluate the effect of changing network architecture/hyperparameters on performance for the MNIST (or Fashion MNIST) task.
A proposal for the final project is also a component of this assignment.

## Files Submitted
- `mnist_recognizer.py` 
- `examine_network.py` 
- `transfer_learning_greek.py`
- `experiment_designer.py` 
- `MyNetwork.py` 
- `Project5_Report.pdf`
- `readme.md`
- `results/` 
  - `mnist_model.pth` 


