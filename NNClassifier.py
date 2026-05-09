# %%
import numpy as np
import torch
from torch import nn
import tqdm

# %%
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# %%
# Load and normalize the dataset for training and testing
# It will downlad the dataset into data subfolder (change to your data folder name)
torch.manual_seed(1)
np.random.seed(1)
train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.2860,), (0.3530,))
                             ]))

test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.2860,), (0.3530,))
                             ]))


# Create a validation set of 10%
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=0.1,
)

# Generate training and validation subsets based on indices
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)


# set batches sizes
train_batch_size = 512 #Define train batch size
test_batch_size  = 256 #Define test batch size (can be larger than train batch size)


# Define dataloader objects that help to iterate over batches and samples for training, validation and testing
train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                                           
num_train_batches=len(train_batches)
num_val_batches=len(val_batches)
num_test_batches=len(test_batches)


print(num_train_batches)
print(num_val_batches)
print(num_test_batches)


#Sample code to visulaize the first sample in first 16 batches 

# batch_num = 0
# for train_features, train_labels in train_batches:
    
#     if batch_num == 16:
#         break    # break here
    
#     batch_num = batch_num +1
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
    
#     img = train_features[0].squeeze()
#     label = train_labels[0]
#     plt.imshow(img, cmap="gray")
#     plt.show()
#     print(f"Label: {label}")



# Sample code to plot N^2 images from the dataset
def plot_images(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
  
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_images(train_dataset.data[:64], 8, "First 64 Training Images" )

    

# %%
#Fully Connected Neural Network
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_neurons=256, batch_norm=False):
        super(Network, self).__init__()
        
        layers = [input_dim] + [hidden_neurons] * hidden_layers + [output_dim]
        self.fc_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(hidden_neurons) for j in range(hidden_layers)]
            )
    
    def forward(self, x):
        for i, layer in enumerate(self.fc_layers[:-1]):  # all but last
            x = layer(x)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.relu(x)
        return self.fc_layers[-1](x)  # no activation on output

# %%
#Function to try different optimizers. It supports SGD, RMSprop and Adam. 
def train_and_evaluate(optimizer_name, lr, init_method='kaiming', batch_norm=False, epochs=40, hidden_layers=2, hidden_neurons=512):
    torch.manual_seed(1)
    np.random.seed(1)
    
    model = Network(input_dim=784, output_dim=10, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, batch_norm=batch_norm)
    
    # Apply weight initialization. 
    for layer in model.fc_layers:
        if init_method == 'xavier':
            nn.init.xavier_normal_(layer.weight)
        elif init_method == 'random_normal':
            nn.init.normal_(layer.weight)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
    
    train_loss_list = np.zeros((epochs,))
    validation_accuracy_list = np.zeros((epochs,))
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    for epoch in tqdm.trange(epochs, desc=f"{optimizer_name} lr={lr} init={init_method} bn={batch_norm}"):
        val_acc = 0.0
        train_loss = 0.0
        
        for train_features, train_labels in train_batches:
            model.train()
            train_features = train_features.reshape(-1, 28*28)
            optimizer.zero_grad()
            output = model(train_features)
            loss = loss_func(output, train_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss_list[epoch] = train_loss / num_train_batches
        
        for val_features, val_labels in val_batches:
            with torch.no_grad():
                model.eval()
                val_features = val_features.reshape(-1, 28*28)
                outputs = model(val_features)
                predicted = torch.argmax(outputs, dim=1)
                val_acc += (predicted == val_labels).float().mean().item()
        
        validation_accuracy_list[epoch] = val_acc / num_val_batches * 100
        print("Epoch: " + str(epoch) + "; Train Loss: " + str(round(train_loss_list[epoch], 4)) +
              "; Validation Accuracy: " + str(round(validation_accuracy_list[epoch], 2)) + '%')
    
    # Test accuracy
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for test_features, test_labels in test_batches:
            model.eval()
            test_features = test_features.reshape(-1, 28*28)
            outputs = model(test_features)
            predicted = torch.argmax(outputs, dim=1)
            test_correct += (predicted == test_labels).sum().item()
            test_total += test_labels.size(0)
    
    test_acc = test_correct / test_total * 100
    print(f"Test Accuracy: {round(test_acc, 2)}%")

    # Plot training loss and validation accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(range(epochs), train_loss_list)
    ax1.set_title(f'Training Loss: {optimizer_name}, lr={lr}, init={init_method}, bn={batch_norm}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(range(epochs), validation_accuracy_list)
    ax2.set_title(f'Validation Accuracy: {optimizer_name}, lr={lr}, init={init_method}, bn={batch_norm}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()
    
    return model, train_loss_list, validation_accuracy_list, test_acc

# %%
#SGD with different learning rates
model_sgd, train_loss_sgd, val_acc_sgd, test_acc_sgd = train_and_evaluate("SGD", lr=0.01)
model_sgd, train_loss_sgd, val_acc_sgd, test_acc_sgd = train_and_evaluate("SGD", lr=0.001)

# %%
#RMSprop with different learning rates
model_rmsprop_01, train_loss_rmsprop_01, val_acc_rmsprop_01, test_acc_rmsprop_01 = train_and_evaluate('RMSprop', lr=0.01)
model_rmsprop_001, train_loss_rmsprop_001, val_acc_rmsprop_001, test_acc_rmsprop_001 = train_and_evaluate('RMSprop', lr=0.001)

# %%
#Adam with different learning rates
model_adam_01, train_loss_adam_01, val_acc_adam_01, test_acc_adam_01 = train_and_evaluate('Adam', lr=0.01)
model_adam_001, train_loss_adam_001, val_acc_adam_001, test_acc_adam_001 = train_and_evaluate('Adam', lr=0.001)

# %%
#Random Normal with different learning rates
init_random, tl_random, val_acc_random, test_acc_random = train_and_evaluate('SGD', lr=0.01, init_method='random_normal')
init_random1, tl_random1, val_acc_random1, test_acc_random1 = train_and_evaluate('SGD', lr=0.001, init_method='random_normal')


# %%
#Xavier with different learning rates
init_xavier, tl_xavier, val_acc_xavier, test_acc_xavier = train_and_evaluate('SGD', lr=0.01, init_method='xavier')
init_xavier1, tl_xavier1, val_acc_xavier1, test_acc_xavier1 = train_and_evaluate('SGD', lr=0.001, init_method='xavier')

# %%
#Batch Norm with different learning rates
model_sgd_bn, train_loss_sgd_bn, val_acc_sgd_bn, test_acc_sgd_bn = train_and_evaluate("SGD", lr=0.01, batch_norm=True)
model_adam_bn, train_loss_adam_bn, val_acc_adam_bn, test_acc_adam_bn = train_and_evaluate("Adam", lr=0.001, batch_norm=True)



