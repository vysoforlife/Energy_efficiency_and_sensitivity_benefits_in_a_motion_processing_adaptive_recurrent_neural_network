import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import gzip
import pickle
import math
from tqdm import tqdm
from math import pi
import random
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(0)

# Parameters
im_size = 32
num_of_frames = 11  # Total frames in the test sequences
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27  # Activation size after convolution

# Load the test dataset
# Adjust the path to your dataset accordingly
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

print("Creating test dataset...")
test_data = RegressionDataset(num_of_frames, num_samples=64000, mode="linear", toTrain=0)
print("Test dataset created.")

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
                             adaptation_rate=0.4,
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
        return out_fc, out_conv, out_flat, out_rnn

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

# Generate moving dot sequences to determine preferred directions
def generate_moving_dot_sequences(num_directions, num_speeds, dot_radius=3, num_frames=10, im_height=32, im_width=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    speeds = torch.linspace(0, 5, num_speeds, device=device)
    angles = torch.linspace(0, 2 * pi, num_directions + 1, device=device)[:-1]
    speeds, angles = torch.meshgrid(speeds, angles, indexing='ij')
    speeds = speeds.flatten()
    angles = angles.flatten()
    batch_size = num_speeds * num_directions

    x_speeds = speeds * torch.cos(angles)
    y_speeds = speeds * torch.sin(angles)

    scale_factor = 4
    hr_size = int(dot_radius * 2 * scale_factor)
    y_coords, x_coords = torch.meshgrid(torch.arange(hr_size, device=device), torch.arange(hr_size, device=device),
                                        indexing='ij')
    y_coords, x_coords = y_coords.float(), x_coords.float()
    center = hr_size / 2 - 0.5
    dist = torch.sqrt(((x_coords - center) / scale_factor) ** 2 + ((y_coords - center) / scale_factor) ** 2)
    hr_dot = torch.clamp(dot_radius - dist, 0, 1) * 255

    all_sequences = torch.zeros(batch_size, num_frames, im_height, im_width, device=device)

    x = torch.full((batch_size,), float(im_width) / 2, device=device)
    y = torch.full((batch_size,), float(im_height) / 2, device=device)

    for t in range(num_frames):
        x = (x + x_speeds) % im_width
        y = (y + y_speeds) % im_height

        canvas = torch.zeros(batch_size, im_height * scale_factor, im_width * scale_factor, device=device)

        x_start = (x - dot_radius) * scale_factor
        y_start = (y - dot_radius) * scale_factor

        for i in range(hr_size):
            for j in range(hr_size):
                canvas_y = (y_start.long() + i) % (im_height * scale_factor)
                canvas_x = (x_start.long() + j) % (im_width * scale_factor)
                canvas[torch.arange(batch_size), canvas_y, canvas_x] = hr_dot[i, j]

        all_sequences[:, t] = F.avg_pool2d(canvas.unsqueeze(1), scale_factor, stride=scale_factor).squeeze(1)

    network_input = torch.zeros(batch_size, num_frames - 1, 2, im_height, im_width, device=device)
    for i in range(num_frames - 1):
        network_input[:, i, 0] = all_sequences[:, i]
        network_input[:, i, 1] = all_sequences[:, i + 1]

    angles_deg = (angles * 180 / pi) % 360
    labels = torch.stack([x_speeds, y_speeds, angles_deg, speeds], dim=1)

    return network_input.cpu(), labels.cpu(), speeds.cpu()

def analyze_rnn_responses(model, sequences):
    model.eval()
    with torch.no_grad():
        _, _, _, rnn_output = model(sequences.to(device))
    return rnn_output.mean(dim=1).cpu().numpy()

def calculate_preferred_direction_speed(rnn_responses, labels):
    num_neurons = rnn_responses.shape[1]
    preferred_directions = []
    preferred_speeds = []

    for neuron in range(num_neurons):
        neuron_responses = rnn_responses[:, neuron]
        max_response_idx = np.argmax(neuron_responses)
        preferred_directions.append(labels[max_response_idx, 2].item())
        preferred_speeds.append(labels[max_response_idx, 3].item())

    return preferred_directions, preferred_speeds

def circular_mean(angles):
    angles_rad = np.deg2rad(angles)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean_rad) % 360

def categorize_neurons(all_preferred_directions, n_categories=8):
    # Updated category edges with 22.5 degree shift
    category_edges = np.linspace(-22.5, 360 - 22.5, n_categories + 1)
    categories = np.digitize(all_preferred_directions, category_edges) - 1
    return categories

# Generate sequences and determine preferred directions for each model
num_directions = 24
num_speeds = 20
dot_radius = 3
num_frames_moving_dot = 10  # Should be more than 1 to get movement
im_height = 32
im_width = 32

batch_sequences, labels, speeds = generate_moving_dot_sequences(
    num_directions, num_speeds, dot_radius, num_frames_moving_dot, im_height, im_width
)

# Function to get neuron categories for a list of models
def get_neuron_categories(models, batch_sequences, labels):
    categories_list = []
    for model in models:
        rnn_responses = analyze_rnn_responses(model, batch_sequences)
        preferred_directions, preferred_speeds = calculate_preferred_direction_speed(rnn_responses, labels)
        categories = categorize_neurons(preferred_directions)
        categories_list.append(categories)
    return categories_list

categories_list1 = get_neuron_categories(models1, batch_sequences, labels)
categories_list2 = get_neuron_categories(models2, batch_sequences, labels)
n_categories = 8

# Get all labels from the test dataset
all_labels = [test_data[i][1] for i in range(len(test_data))]

def categorize_sequences_xy(labels, num_categories=20, min_val_x=-3.8, max_val_x=3.8, min_val_y=-3.8, max_val_y=3.8):
    categories_x = [[] for _ in range(num_categories)]
    categories_y = [[] for _ in range(num_categories)]

    bin_edges_x = np.linspace(min_val_x, max_val_x, num_categories + 1)
    bin_edges_y = np.linspace(min_val_y, max_val_y, num_categories + 1)

    for idx, label in enumerate(labels):
        x_val = label[0].item()
        y_val = label[1].item()

        if min_val_x <= x_val <= max_val_x:
            category_x = np.digitize(x_val, bin_edges_x) - 1
            if category_x == num_categories:
                category_x = num_categories - 1
            categories_x[category_x].append(idx)

        if min_val_y <= y_val <= max_val_y:
            category_y = np.digitize(y_val, bin_edges_y) - 1
            if category_y == num_categories:
                category_y = num_categories - 1
            categories_y[category_y].append(idx)

    return categories_x, categories_y, bin_edges_x, bin_edges_y

categories_x, categories_y, bin_edges_x, bin_edges_y = categorize_sequences_xy(all_labels)

# Print category information
print("Sequence Categories:")
print("-------------------")
print("X Categories:")
for i, (lower, upper) in enumerate(zip(bin_edges_x[:-1], bin_edges_x[1:])):
    count = len(categories_x[i])
    print(f"Category {i + 1}: {lower:.2f} to {upper:.2f} - {count} sequences")

print("\nY Categories:")
for i, (lower, upper) in enumerate(zip(bin_edges_y[:-1], bin_edges_y[1:])):
    count = len(categories_y[i])
    print(f"Category {i + 1}: {lower:.2f} to {upper:.2f} - {count} sequences")
print("-------------------")

# Extract sequences from the test dataset
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
        freeze_content = extended_sequence[:, freeze_frame - 1:freeze_frame, 1:2]  # Channel 1 of the frame before freezing
        for i in range(max(freeze_frame, 3), extended_sequence.size(1)):
            frozen_seq[:, i:i + 1] = freeze_content
        return frozen_seq

# Function to process models and get category responses
def get_category_responses(models, categories_list, n_categories, sequences):
    num_frames = sequences.size(1)
    all_model_responses = np.zeros((len(models), num_frames, n_categories))

    for i, model in enumerate(models):
        categories = categories_list[i]
        with torch.no_grad():
            _, _, _, rnn_outputs = model(sequences)

            for frame in range(num_frames):
                frame_outputs = rnn_outputs[:, frame, :]  # [batch_size, hidden_size]
                for category in range(n_categories):
                    # Get indices of neurons in this category
                    neuron_indices = np.where(categories == category)[0]
                    if len(neuron_indices) == 0:
                        continue

                    # Get responses of these neurons
                    neuron_responses = frame_outputs[:, neuron_indices]  # [batch_size, num_neurons_in_category]

                    # First average across neurons in this category
                    category_response = neuron_responses.mean(dim=1)  # [batch_size]

                    # Then average across batch
                    model_response = category_response.mean().item()

                    all_model_responses[i, frame, category] = model_response

    # Calculate mean and standard error of the mean across models
    avg_responses = all_model_responses.mean(axis=0)  # [num_frames, n_categories]
    std_responses = all_model_responses.std(axis=0) / np.sqrt(len(models))  # Standard error of the mean

    return avg_responses, std_responses

# Plot the responses
def plot_category_responses(responses1, std1, responses2, std2, n_categories, num_frames):
    # Font configuration
    font_path = '../misc/fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams.update({
        'font.size': 30,
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 1
    })

    direction_ranges = [
        f"(337.5°-22.5°)",
        f"(22.5°-67.5°)",
        f"(67.5°-112.5°)",
        f"(112.5°-157.5°)",
        f"(157.5°-202.5°)",
        f"(202.5°-247.5°)",
        f"(247.5°-292.5°)",
        f"(292.5°-337.5°)"
    ]

    cols = 4
    rows = (num_frames + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axs = axs.flatten()

    x = range(n_categories)
    for frame in range(num_frames):
        ax = axs[frame]

        # Plot MotionNet-R with error shading
        ax.plot(x, responses1[frame], marker='o', linewidth=2, label='MotionNet-R', color='blue')
        ax.fill_between(x,
                        responses1[frame] - 2 * std1[frame],  # 95% confidence interval
                        responses1[frame] + 2 * std1[frame],
                        color='blue', alpha=0.2)

        # Plot AdaptNet with error shading
        ax.plot(x, responses2[frame], marker='s', linewidth=2, label='AdaptNet', color='orange')
        ax.fill_between(x,
                        responses2[frame] - 2 * std2[frame],  # 95% confidence interval
                        responses2[frame] + 2 * std2[frame],
                        color='orange', alpha=0.2)

        ax.set_title(f'Frame {frame + 1}')
        ax.set_xlabel('Direction Category')
        ax.set_ylabel('Average Response')
        ax.set_xticks(x)
        ax.set_xticklabels(direction_ranges, rotation=45, ha='right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', width=2, length=10)

        # Dynamically set y-axis limits including error regions
        ymax = max(np.max(responses1[frame] + 2 * std1[frame]),
                   np.max(responses2[frame] + 2 * std2[frame]))
        ymin = min(np.min(responses1[frame] - 2 * std1[frame]),
                   np.min(responses2[frame] - 2 * std2[frame]))
        yrange = ymax - ymin
        ax.set_ylim(max(0, ymin - 0.1 * yrange), ymax + 0.1 * yrange)

    # Remove any remaining empty subplots
    for ax in axs[num_frames:]:
        fig.delaxes(ax)


    plt.tight_layout()

    # Add legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=20)

    plt.savefig('../Saved_Images/RNN_Activity.svg', format='svg')

    plt.show()

# Generate and show the waterfall plot
def calculate_category_response_over_frames(category_indices, models1, models2, test_data, freeze_frame, dimension='x'):
    frame_responses1 = []
    frame_errors1 = []
    frame_responses2 = []
    frame_errors2 = []

    # Set dimension index (0 for x, 1 for y)
    dim_idx = 0 if dimension == 'x' else 1

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
                predictions2, _, _, _ = model2(frozen_sequences)
                # Use the appropriate dimension
                responses1 = predictions1[:, frame, dim_idx]
                responses2 = predictions2[:, frame, dim_idx]
                model_responses1.append(responses1.mean().item())
                model_responses2.append(responses2.mean().item())

        frame_responses1.append(np.mean(model_responses1))
        frame_errors1.append(np.std(model_responses1))
        frame_responses2.append(np.mean(model_responses2))
        frame_errors2.append(np.std(model_responses2))

    return frame_responses1, frame_errors1, frame_responses2, frame_errors2

# Plot waterfall
def plot_waterfall(frame_responses1, frame_errors1, frame_responses2, frame_errors2, dimension='x'):
    plt.figure(figsize=(14, 8))
    frames = range(1, num_of_frames + 1)

    # Plot for AdaptNet
    plt.plot(frames, frame_responses1, '-o', linewidth=2, label='AdaptNet')
    plt.fill_between(frames,
                     np.array(frame_responses1) - np.array(frame_errors1),
                     np.array(frame_responses1) + np.array(frame_errors1),
                     alpha=0.3)

    # Plot for MotionNet-R
    plt.plot(frames, frame_responses2, '-s', linewidth=2, label='MotionNet-R')
    plt.fill_between(frames,
                     np.array(frame_responses2) - np.array(frame_errors2),
                     np.array(frame_responses2) + np.array(frame_errors2),
                     alpha=0.3)

    plt.axhline(y=0, color='black', linestyle='--', label='Reference Line')

    plt.xlabel('Frame')
    if dimension == 'both':
        plt.ylabel('Average Response\n(pixels per frame)')
    else:
        plt.ylabel(f'Average {dimension.upper()}-Response\n(pixels per frame)')
    plt.xticks(range(1, num_of_frames + 1))

    plt.legend()

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)
    ax.tick_params(axis='both', which='major', width=2, length=20)

    plt.tight_layout()
    plt.savefig('../Saved_Images/MAE_Output.svg', format='svg')
    plt.show()

# Main execution flow
dimension = input("Choose dimension to analyze (x/y/both): ").lower()
if dimension not in ['x', 'y', 'both']:
    print("Invalid dimension. Please choose 'x', 'y', or 'both'.")
    exit()

category_number = int(input("Enter a category number (1-20) to analyze: "))
if not (1 <= category_number <= 20):
    print("Invalid category number. Please enter a number between 1 and 20.")
    exit()

freeze_frame = int(input("Enter a frame number to freeze (0-11), or -1 for no freezing: "))
if not (-1 <= freeze_frame <= 11):
    print("Invalid frame number. Please enter a number between -1 and 11.")
    exit()

# Get the appropriate category indices based on dimension
if dimension == 'x':
    category_indices = categories_x[category_number - 1]
elif dimension == 'y':
    category_indices = categories_y[category_number - 1]
else:  # dimension == 'both'
    category_indices_x = categories_x[category_number - 1]
    category_indices_y = categories_y[category_number - 1]
    if len(category_indices_x) == 0 or len(category_indices_y) == 0:
        print("No sequences in one of the categories.")
        exit()
    # Combine indices, ensuring uniqueness
    category_indices = list(set(category_indices_x + category_indices_y))

if len(category_indices) == 0:
    print("No sequences in this category.")
    exit()

# Prepare batch data for the category
sequences = []
labels_in_category = []
for idx in category_indices:
    sequence, label = test_data[idx]
    sequences.append(sequence)
    labels_in_category.append(label)

sequences = torch.stack(sequences).to(device)
frozen_sequences = freeze_sequence(sequences, freeze_frame)

# Get category responses for both models with error information
if dimension == 'both':
    # Process X sequences
    sequences_x = []
    for idx in category_indices_x:
        sequence, label = test_data[idx]
        sequences_x.append(sequence)
    sequences_x = torch.stack(sequences_x).to(device)
    frozen_sequences_x = freeze_sequence(sequences_x, freeze_frame)

    # Process Y sequences
    sequences_y = []
    for idx in category_indices_y:
        sequence, label = test_data[idx]
        sequences_y.append(sequence)
    sequences_y = torch.stack(sequences_y).to(device)
    frozen_sequences_y = freeze_sequence(sequences_y, freeze_frame)

    # Get category responses for both models with error information for both x and y
    responses1_x, std1_x = get_category_responses(models1, categories_list1, n_categories, frozen_sequences_x)
    responses2_x, std2_x = get_category_responses(models2, categories_list2, n_categories, frozen_sequences_x)

    responses1_y, std1_y = get_category_responses(models1, categories_list1, n_categories, frozen_sequences_y)
    responses2_y, std2_y = get_category_responses(models2, categories_list2, n_categories, frozen_sequences_y)

    # Average the responses
    responses1 = (responses1_x + responses1_y) / 2
    std1 = (std1_x + std1_y) / 2

    responses2 = (responses2_x + responses2_y) / 2
    std2 = (std2_x + std2_y) / 2

    # Plot the responses with error shading
    plot_category_responses(responses1, std1, responses2, std2, n_categories, num_of_frames)

    # Get waterfall data with dimension parameter
    frame_responses1_x, frame_errors1_x, frame_responses2_x, frame_errors2_x = calculate_category_response_over_frames(
        category_indices_x, models1, models2, test_data, freeze_frame, dimension='x'
    )

    frame_responses1_y, frame_errors1_y, frame_responses2_y, frame_errors2_y = calculate_category_response_over_frames(
        category_indices_y, models1, models2, test_data, freeze_frame, dimension='y'
    )

    # Average the frame responses
    frame_responses1 = [(fr1 + fr2) / 2 for fr1, fr2 in zip(frame_responses1_x, frame_responses1_y)]
    frame_errors1 = [(fe1 + fe2) / 2 for fe1, fe2 in zip(frame_errors1_x, frame_errors1_y)]
    frame_responses2 = [(fr1 + fr2) / 2 for fr1, fr2 in zip(frame_responses2_x, frame_responses2_y)]
    frame_errors2 = [(fe1 + fe2) / 2 for fe1, fe2 in zip(frame_errors2_x, frame_errors2_y)]

    # Plot waterfall with dimension 'both'
    plot_waterfall(frame_responses1, frame_errors1, frame_responses2, frame_errors2, dimension='both')
else:
    # Get category responses for both models with error information
    responses1, std1 = get_category_responses(models1, categories_list1, n_categories, frozen_sequences)
    responses2, std2 = get_category_responses(models2, categories_list2, n_categories, frozen_sequences)

    # Plot the responses with error shading
    plot_category_responses(responses1, std1, responses2, std2, n_categories, num_of_frames)

    # Get waterfall data with dimension parameter
    frame_responses1, frame_errors1, frame_responses2, frame_errors2 = calculate_category_response_over_frames(
        category_indices, models1, models2, test_data, freeze_frame, dimension
    )

    # Plot waterfall with dimension parameter
    plot_waterfall(frame_responses1, frame_errors1, frame_responses2, frame_errors2, dimension)
