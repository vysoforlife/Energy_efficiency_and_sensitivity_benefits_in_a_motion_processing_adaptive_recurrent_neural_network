import gzip
import pickle
import statistics
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, font_manager
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from scipy.stats import pearsonr

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
torch.manual_seed(0)
# Parameters
im_size = 32
num_of_frames = 9
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27

infile = gzip.GzipFile('../data/train_im32_till10_full_2.1', 'rb')
obj = infile.read()
temp_storage = pickle.loads(obj)
infile.close()
images = temp_storage['images']
labels = temp_storage['labels']


# Dataset class provides image and labels
class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode, toTrain, labl=labels, fts=images):
        """Linear relation between input and output"""
        self.num_samples = num_samples  # number of generated samples
        self.labl = torch.FloatTensor(labl)
        self.toTrain = toTrain
        # self.fts = torch.tensor(fts).unsqueeze(1)
        self.ftT = torch.FloatTensor(fts)
        # torch.div(self.ftT, 255)
        lab_lst = []  # store each generated sample in a list
        feature_list = []
        secFeat = []
        thirFeat = []
        self.indVal = 0
        if toTrain == 0:
            self.indVal = self.labl.size(0) - num_samples

        print('Preparing Cluster Inputs')
        for sno in range(num_samples):
            secFeat = []
            for iter in range(timesteps):
                feature_list = [self.ftT[sno + self.indVal, iter, :, :], self.ftT[sno + self.indVal, iter + 1, :, :]]
                tempT = torch.stack(feature_list, dim=0)
                secFeat.append(tempT)
            thirFeat.append(torch.stack(secFeat, dim=0))
        self.fts = torch.stack(thirFeat, dim=0)

        # generate linear functions one by one

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idxa):
        """General implementation, but we only have one sample."""
        if (self.toTrain == 1):
            return self.fts[idxa, :, :, :, :].to(device), self.labl[idxa].to(device)
        else:
            return self.fts[idxa, :, :, :, :].to(device), self.labl[self.indVal + idxa].to(device)


def targets_to_reg(targets, num_steps):
    """
    Converts labels to have repeated values over timesteps.

    Args:
        targets (Tensor): Original labels.
        num_steps (int): Number of timesteps.

    Returns:
        Tensor: Repeated labels over timesteps.
    """
    return targets.unsqueeze(1).repeat(1, num_steps, 1)


# Dataset and DataLoader
train_data = RegressionDataset(num_of_frames, num_samples=60000, mode="linear", toTrain=1)
val_data = RegressionDataset(num_of_frames, num_samples=4000, mode="linear", toTrain=0)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)


class SimpleRNN_n(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True, batch_first=False):
        super(SimpleRNN_n, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first

        # Initialize weights for input to hidden connection
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))

        # Initialize weights for hidden to hidden connection
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Initialize bias terms
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        # Initialize the weights
        self.reset_parameters()

        # Select the nonlinearity
        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError("Unknown nonlinearity. Supported: 'tanh', 'relu'")

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert batch-first to seq-first

        seq_len, batch_size, _ = x.size()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            hx = self.activation(
                x_t @ self.weight_ih.t() + self.bias_ih +
                hx @ self.weight_hh.t() + self.bias_hh
            )
            output.append(hx)

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)  # Convert seq-first back to batch-first

        return output, hx


class Model_n(nn.Module):
    def __init__(self):
        super(Model_n, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.conv = nn.Conv2d(2, n_kernels, kernel_size=(kernel_size, kernel_size), stride=(stride_size, stride_size))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(input_size=act_size * act_size * n_kernels, hidden_size=rnn_units, nonlinearity='relu',
                               batch_first=True).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        # x = torch.cat([x[:, :, 0], x[:, -1:, 1]], dim=1)
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn

num_models = 10
criterion = nn.MSELoss().to(device)
# Create a list to store all 10 models
models = [Model_n().to(device) for _ in range(num_models)]
optimizers = [Adam(model.parameters(), lr=0.00005) for model in models]

num_epochs = 100

# Training
loss_histories = [[] for _ in range(num_models)]
epoch_losses = [[] for _ in range(num_models)]

print("Starting training of 10 models...")

total_epochs = num_models * num_epochs  # Total number of epochs across all models
with tqdm.tqdm(total=total_epochs, desc="Training Progress") as pbar:
    for model_idx, (model, optimizer) in enumerate(zip(models, optimizers), 1):
        for epoch in range(num_epochs):
            model.train()
            epoch_model_losses = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                targets = targets_to_reg(labels, num_of_frames)

                outputs, conv, _, rnn_out = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_model_losses.append(loss.item())

            avg_loss = statistics.mean(epoch_model_losses)
            loss_histories[model_idx - 1].append(avg_loss)
            epoch_losses[model_idx - 1].append(avg_loss)

            pbar.set_postfix(model=f"{model_idx}/10", epoch=f"{epoch + 1}/{num_epochs}", loss=f"{avg_loss:.3e}")
            pbar.update(1)

        # Save the model
        torch.save(model, f'../models/motionnet_fu2_100epochs_{model_idx}.pt')

print("\nFinished training all 10 models.")

# Testing and visualization
print("Starting evaluation of all models...")

# Initialize lists to store predictions and targets
sc_x = [[[] for _ in range(num_of_frames)] for _ in range(num_models)]
sc_y = [[[] for _ in range(num_of_frames)] for _ in range(num_models)]
sc_xT = []
sc_yT = []
corr_x = [[] for _ in range(num_models)]
corr_y = [[] for _ in range(num_models)]
corr_combined = [[] for _ in range(num_models)]  # To store combined correlations

for model_idx, model in enumerate(models):
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            targets = targets_to_reg(labels, num_of_frames)

            outputs, _, _, _ = model(inputs)
            for fr in range(num_of_frames):
                sc_x[model_idx][fr].append(outputs[:, fr, 0].cpu().numpy())
                sc_y[model_idx][fr].append(outputs[:, fr, 1].cpu().numpy())

            if model_idx == 0:  # Only need to do this once
                sc_xT.append(targets[:, 0, 0].cpu().numpy())
                sc_yT.append(targets[:, 0, 1].cpu().numpy())

        # Concatenate all batches
        actual_x = np.concatenate(sc_xT)
        actual_y = np.concatenate(sc_yT)

        for fr in range(num_of_frames):
            predicted_x = np.concatenate(sc_x[model_idx][fr])
            predicted_y = np.concatenate(sc_y[model_idx][fr])

            # Calculate Pearson correlation for x and y
            corr_x_val, _ = pearsonr(actual_x, predicted_x)
            corr_y_val, _ = pearsonr(actual_y, predicted_y)
            corr_x[model_idx].append(corr_x_val)
            corr_y[model_idx].append(corr_y_val)

            # Calculate combined correlation as the average of x and y correlations
            combined_corr = (corr_x_val + corr_y_val) / 2
            corr_combined[model_idx].append(combined_corr)

# Calculate average correlations and standard deviations
avg_corr_x = np.mean(corr_x, axis=0)
avg_corr_y = np.mean(corr_y, axis=0)
avg_corr_combined = np.mean(corr_combined, axis=0)

std_corr_x = np.std(corr_x, axis=0)
std_corr_y = np.std(corr_y, axis=0)
std_corr_combined = np.std(corr_combined, axis=0)

# Line Plot for Average Correlations with Error Bars
plt.figure(figsize=(12, 6))
frames = range(1, num_of_frames + 1)

plt.errorbar(frames, avg_corr_x, yerr=std_corr_x, label='Average Correlation x', capsize=5, marker='o')
plt.errorbar(frames, avg_corr_y, yerr=std_corr_y, label='Average Correlation y', capsize=5, marker='s')
plt.errorbar(frames, avg_corr_combined, yerr=std_corr_combined, label='Average Combined Correlation', capsize=5, marker='^')

plt.xlabel('Frame')
plt.ylabel('Average Pearson Correlation')
plt.title('Average Pearson Correlation between Predicted and Actual Values over Time')
plt.xticks(frames)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Compute average training loss over epochs
import numpy as np

# Convert loss_histories to numpy array
loss_histories_np = np.array(loss_histories)  # shape (num_models, num_epochs)

mean_losses = np.mean(loss_histories_np, axis=0)  # shape (num_epochs,)
std_losses = np.std(loss_histories_np, axis=0)

epochs = range(1, num_epochs + 1)

# Plot Average Training Loss with Error Bars
plt.figure(figsize=(30, 15))
plt.errorbar(epochs, mean_losses, yerr=std_losses, capsize=5, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.title('Average Training Loss over Epochs')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()