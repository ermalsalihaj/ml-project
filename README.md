# ML Project — Salient Object Detection (SOD)

Xponian Program — Cohort IV
Author: Ermal Salihaj

This repository contains a complete end-to-end implementation of a Salient Object Detection (SOD) system, developed from scratch using PyTorch. The goal of the project is to detect and segment the most visually important object in an image using a custom convolutional neural network architecture.

Dataset and Preprocessing

The project uses the DUTS dataset, which is one of the largest benchmark datasets for saliency detection.
All images were resized to 128×128, normalized to the 0–1 range, and paired with their binary ground-truth masks.
The dataset was split into training, validation, and test sets in a 70/15/15 ratio.
Basic augmentations such as horizontal flipping and brightness adjustment were applied during training to improve generalization.

Model Architecture

The model is a custom encoder–decoder CNN built entirely from scratch.
It includes a four-stage convolutional encoder, a deeper bottleneck layer, and a ConvTranspose2D-based decoder.
Batch Normalization and Dropout were introduced in the improved version of the model to enhance stability and reduce overfitting.
The final output is a one-channel mask generated using a sigmoid activation function.

Training

The training pipeline uses a combination of Binary Cross-Entropy loss and an IoU-based loss term.
The Adam optimizer with a learning rate of 1e-3 was used, along with early stopping to prevent overfitting.
Training was performed on an NVIDIA RTX 4050 Laptop GPU, and the model with the best validation loss was automatically saved as a checkpoint.

Evaluation

Model evaluation is implemented in the evaluate.py script.
The metrics computed on the test set include Intersection over Union (IoU), Precision, Recall, F1-Score, and Mean Absolute Error (MAE).
These metrics provide a comprehensive view of segmentation quality and prediction accuracy.

Results

The improved model, which includes Batch Normalization, Dropout, and an enhanced bottleneck, achieved the following performance on the test set:

IoU: 0.7249
Precision: 0.8410
Recall: 0.8788
F1-Score: 0.8562
MAE: 0.0919

These results demonstrate a clear improvement over the baseline version of the network.

Visualizations

The repository includes scripts to generate model predictions and visualization outputs.
visualize_single.py produces a prediction for a single input image, while visualize_batch.py visualizes multiple images at once.
An interactive demo is also available in the Jupyter notebook demo_notebook.ipynb, allowing easier experimentation and inspection of model outputs.

Final Deliverables

This repository contains the full source code for the project, the notebook demo, the final written report in PDF format, and the presentation slides used for the project review.