# FashionMNIST Neural Network Classifier

A fully connected deep neural network for image classification on the FashionMNIST dataset, implemented in PyTorch. This project explores the effects of optimizer choice, weight initialization strategies, and batch normalization on model performance.

## Results Summary

| Configuration                     | Test Accuracy |
|----------------------------------|---------------|
| SGD, lr=0.01, Kaiming (baseline) | 86.73%        |
| RMSprop, lr=0.001                | 88.96%        |
| Adam, lr=0.01                    | 87.47%        |
| **Adam, lr=0.001 (best)**        | **89.57%**    |
| SGD + Batch Norm, lr=0.01        | 86.92%        |

## Project Structure

```
.
├── NNClassifier.ipynb  # Main training notebook
├── NeuralNetwork.pdf   # Full report with results and analysis
└── data/               # FashionMNIST dataset (auto-downloaded)
```

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib
tqdm
```

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm
```

## Usage

Launch the notebook:

```bash
jupyter notebook NNClassifier.ipynb
```

Run cells sequentially, or use **Run All**. The two summary cells at the bottom, which are the test accuracy bar chart and the optimizer comparison overlay, should be run only after all training cells have completed.

## Model Architecture

A flexible fully connected network with configurable depth and width:

- **Input**: 784-dimensional vector (flattened 28×28 image)
- **Hidden layers**: 2 layers × 512 neurons (default)
- **Activation**: ReLU
- **Output**: 10 classes (softmax via CrossEntropyLoss)
- **Optional**: Batch Normalization after each hidden layer

## Key Findings

- **Adam with lr=0.001** achieved the best test accuracy of 89.57%
- **Kaiming initialization** outperformed Random Normal and Xavier for ReLU networks, as it accounts for ReLU's half-unit deactivation
- **Batch normalization** improved accuracy slightly (86.73% → 86.92%) at the cost of added computation
- **Adaptive optimizers** (Adam, RMSprop) benefit from a lower learning rate of 0.001, while SGD performs better at 0.01

## Dataset

FashionMNIST is automatically downloaded on first run into the `data/` directory. Images are normalized with mean=0.2860 and std=0.3530. The training set is split 90/10 into train and validation subsets using stratified sampling.

## Author

Hyun Jun Yoo
