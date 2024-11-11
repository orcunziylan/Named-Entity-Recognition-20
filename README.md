# Named Entity Recognition

This project is a part of the NLP course. The task involves categorizing input text data into predefined classes for Named Entity Recognition (NER).

## Overview

The code, written in Python, utilizes popular libraries such as PyTorch, NumPy, Matplotlib, and NLTK. The process involves data preparation, model building, training, and evaluation. Here's a summarized guide:

### Data Preparation

- Load data from CSV files containing sentences and labels.
- Process and prepare data for training, validation, and testing using the `build_dataset` class.
- Apply word lemmatization and n-gram processing (if needed).
- Build vocabulary and label indices.

### Model Building

- Define a Bi-LSTM model using PyTorch's `nn.Module` class.
- The model includes an embedding layer, a Bi-LSTM layer, dropout for regularization, and a linear classifier.
- Optionally, initialize the embedding layer with pre-trained GloVe word embeddings.

### Training and Evaluation

- The `Trainer` class manages model training and evaluation.
- Train the model on the training dataset while monitoring validation loss to prevent overfitting.
- Compute the F1 score during training for performance evaluation.
- Generate confusion matrices for result visualization.

### Experiments

The repository showcases three experiments:

1. Training the Bi-LSTM model without pre-trained embeddings.
2. Training the Bi-LSTM model with pre-trained GloVe embeddings.
3. Training the Bi-LSTM model with lemmatized inputs.

### Results and Visualization

- Plot training and validation loss curves for performance monitoring.
- Visualize confusion matrices with heatmaps to display classification results.
