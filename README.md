MORAN: Multimodal Optical Recognition with Attention

MORAN is a state-of-the-art optical character recognition (OCR) system designed for handling challenging tasks such as reading text in natural scenes, where text might appear in various sizes, fonts, orientations, and lighting conditions. This system utilizes a combination of convolutional and recurrent neural networks with attention mechanisms to accurately transcribe text from images.

Table of Contents
- Installation
- Usage
- Training
- Contributing
- License

Installation

To install MORAN and its dependencies, you can use pip with the provided requirements.txt file.

pip install -r requirements.txt

Please note that MORAN requires a CUDA-enabled GPU for optimal performance.

Usage

After installing the necessary dependencies, you can utilize MORAN for text recognition tasks. Here's a brief overview of how to use it:

Prepare your datasets: MORAN expects training, validation, and testing datasets in LMDB format. Ensure your datasets are appropriately formatted and accessible.
Set up configurations: Customize the configuration parameters in the main.py file according to your requirements. You can specify paths to your datasets, adjust batch sizes, learning rates, and other hyperparameters.
Train the model: Execute the main.py script to start training MORAN on your datasets. You can monitor training progress and performance metrics during training.
Evaluate the model: After training, you can evaluate the trained model on your validation and testing datasets to assess its accuracy and generalization performance.

Training

Training MORAN involves optimizing its parameters using backpropagation and gradient descent techniques. During training, the model learns to recognize text patterns from input images and minimize a predefined loss function.

Here's a summary of the training process:

Iterate over batches of images and corresponding ground truth texts.
Forward pass: Pass the images through the MORAN model to obtain predicted text sequences.
Compute the loss: Compare the predicted text sequences with the ground truth texts using a suitable loss function (e.g., cross-entropy loss).
Backward pass: Compute gradients of the loss with respect to the model parameters.
Update parameters: Use an optimizer (e.g., Adam, Adadelta, RMSprop) to update the model parameters based on the computed gradients.
Repeat the process for multiple epochs until convergence or satisfactory performance is achieved.

Contributing

Contributions to MORAN are welcome! If you have any ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request on the GitHub repository.

License

This project is licensed under the MIT License - see the LICENSE file for details.
