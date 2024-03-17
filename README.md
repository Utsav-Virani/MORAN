# MORAN: Multimodal Optical Recognition with Attention

MORAN is an advanced optical character recognition (OCR) system tailored to tackle challenging tasks such as reading text in natural scenes. With its ability to handle text in various sizes, fonts, orientations, and lighting conditions, MORAN stands as a state-of-the-art solution in the field.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)

## Installation

To install MORAN and its dependencies, you can use pip with the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## Prerequisites
Ensure that you have a CUDA-enabled GPU for optimal performance.

## Usage

1. **Prepare your datasets**: MORAN expects training, validation, and testing datasets in LMDB format. Ensure your datasets are appropriately formatted and accessible.

2. **Set up configurations**: Customize the configuration parameters in the `main.py` file according to your requirements. You can specify paths to your datasets, adjust batch sizes, learning rates, and other hyperparameters.

3. **Train the model**: Execute the `main.py` script to start training MORAN on your datasets. You can monitor training progress and performance metrics during training.

4. **Evaluate the model**: After training, evaluate the trained model on your validation and testing datasets to assess its accuracy and generalization performance.

## Training

Training MORAN involves optimizing its parameters using backpropagation and gradient descent techniques. Here's a summary of the training process:

- Iterate over batches of images and corresponding ground truth texts.
- Forward pass: Pass the images through the MORAN model to obtain predicted text sequences.
- Compute the loss: Compare the predicted text sequences with the ground truth texts using a suitable loss function (e.g., cross-entropy loss).
- Backward pass: Compute gradients of the loss with respect to the model parameters.
- Update parameters: Use an optimizer (e.g., Adam, Adadelta, RMSprop) to update the model parameters based on the computed gradients.
- Repeat the process for multiple epochs until convergence or satisfactory performance is achieved.
