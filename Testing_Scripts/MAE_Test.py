import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from torch.utils.data import DataLoader
import gzip
import pickle
import math

from tqdm import tqdm

device = torch.device('cuda')
print(torch.cuda.is_available())
print(torch.version.cuda)
print(f"Using device: {device}")
torch.manual_seed(0)

# Load the dataset
infile = gzip.GzipFile('../Training_Data/train_im32_till10_full.1', 'rb')
obj = infile.read()
temp_storage = pickle.loads(obj)
infile.close()
images = temp_storage['images']
labels = temp_storage['labels']


# Dataset class
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, timesteps, num_samples, mode, toTrain, labl=labels, fts=images):
        self.num_samples = num_samples
        self.labl = torch.FloatTensor(labl)
        self.toTrain = toTrain
        self.ftT = torch.FloatTensor(fts)
        torch.div(self.ftT, 255)
        self.indVal = 0
        if toTrain == 0:
            self.indVal = self.labl.size(0) - num_samples

        print('Preparing Cluster Inputs')
        thirFeat = []
        for sno in range(num_samples):
            secFeat = []
            for iter in range(timesteps - 2):  # Reduce by 2 to account for the additional frames
                feature_list = [self.ftT[sno + self.indVal, iter, :, :], self.ftT[sno + self.indVal, iter + 1, :, :]]
                tempT = torch.stack(feature_list, dim=0)
                secFeat.append(tempT)
            # Add two more frames at the end
            for _ in range(2):
                feature_list = [self.ftT[sno + self.indVal, -1, :, :], self.ftT[sno + self.indVal, -1, :, :]]
                tempT = torch.stack(feature_list, dim=0)
                secFeat.append(tempT)
            thirFeat.append(torch.stack(secFeat, dim=0))
        self.fts = torch.stack(thirFeat, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idxa):
        if (self.toTrain == 1):
            return self.fts[idxa, :, :, :, :].to(device), self.labl[idxa].to(device)
        else:
            return self.fts[idxa, :, :, :, :].to(device), self.labl[self.indVal + idxa].to(device)


# Model definition
# Base RNN class for the first model
class SimpleRNN_n(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True, batch_first=False):
        super(SimpleRNN_n, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

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
            x = x.transpose(0, 1)

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
            output = output.transpose(0, 1)

        return output, hx


# Adaptive RNN class for the second model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True, batch_first=False,
                 adaptation_rate=0.2, recovery_rate=0.1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

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
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        adaptation = torch.zeros_like(hx)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            pre_activation = x_t @ self.weight_ih.t() + self.bias_ih + hx @ self.weight_hh.t() + self.bias_hh
            adapted_activation = torch.max(torch.zeros_like(pre_activation), pre_activation - adaptation)
            adaptation += self.adaptation_rate * adapted_activation
            adaptation -= self.recovery_rate * adaptation
            hx = self.activation(adapted_activation)
            output.append(hx)

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hx


class AdaptiveLayer(nn.Module):
    def __init__(self, num_features, adaptation_rate=0.1, recovery_rate=0.1):
        super(AdaptiveLayer, self).__init__()
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate
        self.adaptation = nn.Parameter(torch.zeros(1, num_features), requires_grad=False)

    def forward(self, x):
        batch_size, num_frames, num_features = x.size()
        adaptation = self.adaptation.expand(batch_size, -1)

        adapted_output = []
        for i in range(num_frames):
            frame = x[:, i, :]
            output = torch.clamp(frame - adaptation, min=0)
            adaptation = adaptation + self.adaptation_rate * output
            adaptation = adaptation * (1 - self.recovery_rate)
            adapted_output.append(output)

        self.adaptation.data = adaptation.mean(dim=0, keepdim=True)
        return torch.stack(adapted_output, dim=1)


# First model (Normal)
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
        self.rnn = SimpleRNN_n(input_size=act_size * act_size * n_kernels, hidden_size=rnn_units,
                               nonlinearity='relu', batch_first=True).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn


# Second model (Adaptive)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.conv = nn.Conv2d(2, n_kernels, kernel_size=(kernel_size, kernel_size), stride=(stride_size, stride_size))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.adaptive_layer = AdaptiveLayer(num_features=act_size * act_size * n_kernels,
                                            adaptation_rate=0.1,
                                            recovery_rate=0.1)
        self.rnn = SimpleRNN(input_size=act_size * act_size * n_kernels,
                             hidden_size=rnn_units,
                             nonlinearity='relu',
                             batch_first=True,
                             adaptation_rate=0.2,
                             recovery_rate=0.1).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_adapted = self.adaptive_layer(out_flat)
        out_rnn, _ = self.rnn(out_adapted)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_adapted, out_rnn

# Parameters
im_size = 32
num_of_frames = 11  # Previously 9
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27

print("Creating test dataset...")
test_data = RegressionDataset(num_of_frames, num_samples=64000, mode="linear", toTrain=0)
print("Test dataset created.")

# Load trained models
num_models = 10
models1 = []
models2 = []
for i in range(1, num_models + 1):
    model1 = torch.load(f'../Trained_Models/motionnet_full_30epochs_{i}.pt', map_location=device)
    model1.eval()
    model1.to(device)
    models1.append(model1)

    model2 = torch.load(f'../Trained_Models/adaptnet_full_30epochs_{i}_04.pt', map_location=device)
    model2.eval()
    model2.to(device)
    models2.append(model2)


def categorize_sequences(labels, num_categories=20, min_val=-3.8, max_val=3.8):
    categories = [[] for _ in range(num_categories)]
    bin_edges = np.linspace(min_val, max_val, num_categories + 1)

    for idx, label in enumerate(labels):
        x_val = label[0].item()
        if min_val <= x_val <= max_val:
            category = np.digitize(x_val, bin_edges) - 1
            if category == num_categories:
                category = num_categories - 1
            categories[category].append(idx)

    return categories, bin_edges


def freeze_sequence(sequence, freeze_frame):
    # Add two more frames at the end
    extended_sequence = torch.cat([sequence, sequence[:, -1:].repeat(1, 2, 1, 1, 1)], dim=1)

    if freeze_frame == -1:
        # Make the first two frames stationary
        extended_sequence[:, :2, :, :, :] = extended_sequence[:, 2:3, 0:1, :, :].repeat(1, 2, 2, 1, 1)
        return extended_sequence
    elif freeze_frame == 0:
        # Make all frames stationary based on the first frame of the third timestep
        return extended_sequence[:, 2:3, 0:1, :, :].repeat(1, extended_sequence.size(1), 2, 1, 1)
    else:
        frozen_seq = extended_sequence.clone()
        # Make the first two frames stationary
        frozen_seq[:, :2, :, :, :] = extended_sequence[:, 2:3, 0:1, :, :].repeat(1, 2, 2, 1, 1)
        # Apply freezing from the specified frame onwards
        freeze_content = extended_sequence[:, freeze_frame - 1:freeze_frame,
                         1:2]  # Channel 1 of the frame before freezing
        for i in range(max(freeze_frame, 3), extended_sequence.size(1)):
            frozen_seq[:, i:i + 1] = freeze_content
        return frozen_seq


def calculate_category_response_over_frames(category_indices, models1, models2, test_data, freeze_frame):
    frame_responses1 = []
    frame_errors1 = []
    frame_responses2 = []
    frame_errors2 = []

    # Prepare batch data for the category
    sequences = []
    for idx in category_indices:
        sequence, _ = test_data[idx]
        sequences.append(sequence)

    sequences = torch.stack(sequences).to(device)
    frozen_sequences = freeze_sequence(sequences, freeze_frame)

    for frame in range(num_of_frames):
        model_responses1 = []
        model_responses2 = []
        for model1, model2 in zip(models1, models2):
            model1.eval()
            model2.eval()
            with torch.no_grad():
                predictions1, _, _, _ = model1(frozen_sequences)
                predictions2, _, _, _, _ = model2(frozen_sequences)
                x_responses1 = predictions1[:, frame, 0]
                x_responses2 = predictions2[:, frame, 0]
                model_responses1.append(x_responses1.mean().item())
                model_responses2.append(x_responses2.mean().item())

        frame_responses1.append(np.mean(model_responses1))
        frame_errors1.append(np.std(model_responses1))
        frame_responses2.append(np.mean(model_responses2))
        frame_errors2.append(np.std(model_responses2))

    return frame_responses1, frame_errors1, frame_responses2, frame_errors2


# Get all labels
all_labels = [test_data[i][1] for i in range(len(test_data))]
categories, bin_edges = categorize_sequences(all_labels)

# Print category information
print("Sequence Categories:")
print("-------------------")
for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    count = len(categories[i])
    print(f"Category {i + 1}: {lower:.2f} to {upper:.2f} - {count} sequences")
print("-------------------")

# Main loop
while True:
    try:
        category = int(input("Enter a category number (1-20) to analyze, or 0 to quit: "))
        if category == 0:
            break

        if 1 <= category <= 20:
            category_indices = categories[category - 1]
            if len(category_indices) == 0:
                print("No sequences in this category. Please choose another.")
                continue

            freeze_frame = int(input("Enter a frame number to freeze (0-11), or -1 for no freezing: "))
            if -1 <= freeze_frame <= 11:
                print(f"Calculating responses for category {category} with freeze frame {freeze_frame}...")
                frame_responses1, frame_errors1, frame_responses2, frame_errors2 = calculate_category_response_over_frames(
                    category_indices, models1, models2,
                    test_data, freeze_frame)

                font_path = '../misc/fonts/Roboto-Regular.ttf'  # Update this path
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Roboto'

                plt.rcParams.update({
                    'font.size': 35,
                    'axes.titlesize': 35,
                    'axes.labelsize': 35,
                    'xtick.labelsize': 35,
                    'ytick.labelsize': 35,
                    'axes.linewidth': 1
                })

                # Plot the results
                plt.figure(figsize=(22, 10))  # Increased figure width to accommodate additional frames
                frames = range(1, num_of_frames + 1)

                # Font configuration
                font_path = '../misc/fonts/Roboto-Regular.ttf'  # Update this path
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Roboto'

                plt.rcParams.update({
                    'font.size': 30,
                    'axes.titlesize': 30,
                    'axes.labelsize': 30,
                    'xtick.labelsize': 30,
                    'ytick.labelsize': 30,
                    'axes.linewidth': 1
                })

                # Plot for AdaptNet
                plt.plot(frames, frame_responses1, '-o', linewidth=2, label='AdaptNet')
                plt.fill_between(frames,
                                 np.array(frame_responses1) - np.array(frame_errors1),
                                 np.array(frame_responses1) + np.array(frame_errors1),
                                 alpha=0.3)

                # Plot for Motionnet-R
                plt.plot(frames, frame_responses2, '-s', linewidth=2, label='Motionnet-R')
                plt.fill_between(frames,
                                 np.array(frame_responses2) - np.array(frame_errors2),
                                 np.array(frame_responses2) + np.array(frame_errors2),
                                 alpha=0.3)

                plt.axhline(y=0, color='black', linestyle='--', label='Reference Line')

                plt.xlabel('Frame')
                plt.ylabel('Average X Response\n(pixels per frame)')
                plt.xticks(range(1, num_of_frames + 1))  # Ensure all frame numbers are shown on x-axis

                # Get the range of values for the current category
                category_min = bin_edges[category - 1]
                category_max = bin_edges[category]

                plt.legend()

                # Remove the top and right spines
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Remove the grid
                plt.grid(False)

                # Increase tick size
                ax.tick_params(axis='both', which='major', width=2, length=20)

                plt.tight_layout()
                plt.savefig('waterfall_single_ver2.svg', format='svg')
                plt.show()
            else:
                print("Invalid frame number. Please enter a number between -1 and 9.")
        else:
            print("Invalid category number. Please enter a number between 1 and 20.")

    except ValueError:
        print("Invalid input. Please enter valid numbers.")

print("Analysis complete.")