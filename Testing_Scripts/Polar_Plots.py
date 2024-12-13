import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch import nn
from scipy import interpolate

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################
# CONSTANTS & PARAMETERS
############################################
IM_SIZE = 32            # Image size (height and width)
NUM_FRAMES = 9          # Number of frames used in stimulus generation
N_KERNELS = 16          # Number of Conv kernels
RNN_UNITS = 16          # Number of units in the RNN layer
ACT_SIZE = 27           # Derived from the convolution output size (27x27)
LIN_UNITS = 16          # Units in the first linear layer of mnet

# Additional model parameters used when building models
im_size = 32
num_of_frames = 40
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27
lin_units = 16

############################################
# MODEL DEFINITIONS
############################################

class mnet(nn.Module):
    """
    A simple feedforward model that processes each frame independently:
    - Conv2D(2 -> n_kernels)
    - Flatten+ReLU
    - Linear -> ReLU
    - Linear -> output(2)

    Returns per-frame outputs stacked along time dimension.
    """
    def __init__(self):
        super(mnet, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.conv = nn.Conv2d(2, n_kernels, kernel_size=(kernel_size, kernel_size), stride=(stride_size, stride_size))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=n_kernels * act_size * act_size, out_features=lin_units),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(lin_units, 2)
        )

        self.apply(init_weights)

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels=2, height=32, width=32)
        batch_size, num_frames, _, _, _ = x.size()

        # Process each frame independently
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_fc1 = torch.stack([self.fc1(out_flat[:, i]) for i in range(num_frames)], dim=1)
        out_fc2 = torch.stack([self.fc2(out_fc1[:, i]) for i in range(num_frames)], dim=1)

        return out_fc2, out_conv, out_flat, out_fc1


class SimpleRNN_n(nn.Module):
    """
    A simple RNN implementation using ReLU as the nonlinearity:
    h_t = ReLU(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
    """
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN_n, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight_ih, -0.1, 0.1)
        nn.init.uniform_(self.weight_hh, -0.1, 0.1)
        nn.init.uniform_(self.bias_ih, -0.1, 0.1)
        nn.init.uniform_(self.bias_hh, -0.1, 0.1)

    def forward(self, x, hx=None):
        # x shape: (seq_len, batch_size, input_size)
        if hx is None:
            hx = torch.zeros(x.size(1), self.hidden_size, device=x.device)

        output = []
        for t in range(x.size(0)):
            hx = torch.relu(
                x[t] @ self.weight_ih.t() + self.bias_ih +
                hx @ self.weight_hh.t() + self.bias_hh
            )
            output.append(hx)
        return torch.stack(output, dim=0), hx


class Model_n(nn.Module):
    """
    Model_n:
    - Convolution
    - Flatten+ReLU
    - SimpleRNN_n
    - FC -> 2 outputs
    """
    def __init__(self):
        super(Model_n, self).__init__()
        self.conv = nn.Conv2d(2, N_KERNELS, kernel_size=6, stride=1)
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(27 * 27 * N_KERNELS, RNN_UNITS)
        self.fc = nn.Linear(RNN_UNITS, 2)

    def forward(self, x):
        # x shape: (batch_size, num_frames, 2, 32, 32)
        batch_size, num_frames, _, _, _ = x.size()

        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat.transpose(0, 1))
        out_fc = torch.stack([self.fc(out_rnn[i]) for i in range(num_frames)], dim=1)

        return out_fc, out_conv, out_flat, out_rnn.transpose(0, 1)


class SimpleRNN(nn.Module):
    """
    A custom Simple RNN with adaptation and recovery:
    adapted_activation = max(0, pre_activation - adaptation)

    adaptation dynamics:
      adaptation += adaptation_rate * adapted_activation
      adaptation -= recovery_rate * adaptation

    Nonlinearity: 'relu' or 'tanh'
    """
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True, batch_first=False, adaptation_rate=0.2,
                 recovery_rate=0.1):
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
        # If batch_first: (batch, seq, feature) -> (seq, batch, feature)
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


class Model(nn.Module):
    """
    Model:
    - Conv2D -> Flatten+ReLU
    - Custom SimpleRNN (with adaptation)
    - FC -> 2 outputs
    """
    def __init__(self):
        super(Model, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.conv = nn.Conv2d(2, n_kernels, kernel_size=(kernel_size, kernel_size), stride=(stride_size, stride_size))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN(
            input_size=act_size * act_size * n_kernels,
            hidden_size=rnn_units,
            nonlinearity='relu',
            batch_first=True,
            adaptation_rate=0.4,
            recovery_rate=0.1
        ).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        # x shape: (batch_size, num_frames, 2, 32, 32)
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)

        return out_fc, out_conv, out_flat, out_rnn


############################################
# STIMULUS GENERATION
############################################

def generate_grating(width, height, frequency, direction, num_frames, speed, is_plaid=False):
    """
    Generate a single grating (sine or plaid).
    Returns frames of shape: (num_frames+1, height, width)
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    frames = []
    for i in range(num_frames + 1):
        phase_shift = i * 2 * np.pi * speed / num_frames
        if is_plaid:
            angle_rad1 = np.deg2rad(direction - 67.5)
            angle_rad2 = np.deg2rad(direction + 67.5)
            grating1 = np.sin(2 * np.pi * frequency * (-x * np.cos(angle_rad1) + y * np.sin(angle_rad1) + phase_shift))
            grating2 = np.sin(2 * np.pi * frequency * (-x * np.cos(angle_rad2) + y * np.sin(angle_rad2) + phase_shift))
            grating = (grating1 + grating2) / 2
        else:
            angle_rad = np.deg2rad(direction)
            grating = np.sin(2 * np.pi * frequency * (-x * np.cos(angle_rad) + y * np.sin(angle_rad) + phase_shift))
        frames.append((grating + 1) / 2)
    return np.array(frames)


def generate_grating_sequences(width, height, directions, spatial_frequencies, temporal_frequencies, num_frames, stimulus_type='plaid'):
    """
    Generate sequences for all SF, TF, and directions.
    Shape: (n_sf, n_tf, n_dir, num_frames, 2, width, height)
    """
    n_sf, n_tf, n_dir = len(spatial_frequencies), len(temporal_frequencies), len(directions)
    sequences = np.zeros((n_sf, n_tf, n_dir, num_frames, 2, width, height))
    for i, sf in enumerate(spatial_frequencies):
        for j, tf in enumerate(temporal_frequencies):
            for k, direction in enumerate(directions):
                frames = generate_grating(width, height, sf, direction, num_frames, tf, is_plaid=(stimulus_type == 'plaid'))
                # Store pairs of consecutive frames in channel 0 and 1
                sequences[i, j, k, :, 0] = frames[:-1]
                sequences[i, j, k, :, 1] = frames[1:]
    return sequences


############################################
# OPTIMAL FREQUENCY SELECTION
############################################

def find_optimal_sf_tf(models, sequences, num_units, num_kernels, num_frames):
    """
    Find optimal SF/TF for RNN units and Conv kernels.
    For each model, find which SF/TF pair produces the max mean response across directions.
    """
    n_sf, n_tf, n_dir = sequences.shape[:3]
    optimal_sf_tf_rnn_list = []
    optimal_sf_tf_conv_list = []

    for model in models:
        optimal_sf_tf_rnn = np.zeros((num_units, 2), dtype=int)
        optimal_sf_tf_conv = np.zeros((num_kernels, 2), dtype=int)
        max_responses_rnn = np.zeros(num_units)
        max_responses_conv = np.zeros(num_kernels)

        for i in range(n_sf):
            for j in range(n_tf):
                rnn_responses = np.zeros((num_units, n_dir))
                conv_responses = np.zeros((num_kernels, n_dir))

                for k in range(n_dir):
                    sample_input = torch.tensor(sequences[i, j, k], dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, out_conv, _, rnn_outputs = model(sample_input)

                    rnn_responses[:, k] = rnn_outputs[0].mean(dim=0).cpu().numpy()
                    conv_responses[:, k] = out_conv[0].mean(dim=(0, 2, 3)).cpu().numpy()

                # Update optimal for RNN units
                for unit in range(num_units):
                    if rnn_responses[unit].max() > max_responses_rnn[unit]:
                        max_responses_rnn[unit] = rnn_responses[unit].max()
                        optimal_sf_tf_rnn[unit] = [i, j]

                # Update optimal for Conv kernels
                for kernel in range(num_kernels):
                    if conv_responses[kernel].max() > max_responses_conv[kernel]:
                        max_responses_conv[kernel] = conv_responses[kernel].max()
                        optimal_sf_tf_conv[kernel] = [i, j]

        optimal_sf_tf_rnn_list.append(optimal_sf_tf_rnn)
        optimal_sf_tf_conv_list.append(optimal_sf_tf_conv)

    return optimal_sf_tf_rnn_list, optimal_sf_tf_conv_list


def plot_polar_responses(frame, angles, activations_rnn_plaid_list, activations_rnn_sine_list,
                         activations_conv_plaid_list, activations_conv_sine_list,
                         rotation_rnn_degrees=180, rotation_conv_degrees=180):
    """
    Plot averaged RNN and Conv responses on polar plots, with normalization and shaded error bars.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})

    num_models = len(activations_rnn_plaid_list)
    num_angles = len(angles)

    # Arrays to store normalized activations
    normalized_rnn_plaid = np.zeros((num_models, num_angles))
    normalized_rnn_sine = np.zeros((num_models, num_angles))
    normalized_conv_plaid = np.zeros((num_models, num_angles))
    normalized_conv_sine = np.zeros((num_models, num_angles))

    # Normalize responses model-wise
    for idx in range(num_models):
        activations_rnn_plaid = activations_rnn_plaid_list[idx]
        activations_rnn_sine = activations_rnn_sine_list[idx]
        activations_conv_plaid = activations_conv_plaid_list[idx]
        activations_conv_sine = activations_conv_sine_list[idx]

        avg_rnn_plaid = activations_rnn_plaid.mean(axis=1)
        avg_rnn_sine = activations_rnn_sine.mean(axis=1)
        avg_conv_plaid = activations_conv_plaid.mean(axis=1)
        avg_conv_sine = activations_conv_sine.mean(axis=1)

        # Normalize individually
        norm_rnn_plaid = (avg_rnn_plaid - avg_rnn_plaid.min()) / (avg_rnn_plaid.max() - avg_rnn_plaid.min()) if avg_rnn_plaid.max() > avg_rnn_plaid.min() else np.zeros_like(avg_rnn_plaid)
        norm_rnn_sine = (avg_rnn_sine - avg_rnn_sine.min()) / (avg_rnn_sine.max() - avg_rnn_sine.min()) if avg_rnn_sine.max() > avg_rnn_sine.min() else np.zeros_like(avg_rnn_sine)
        norm_conv_plaid = (avg_conv_plaid - avg_conv_plaid.min()) / (avg_conv_plaid.max() - avg_conv_plaid.min()) if avg_conv_plaid.max() > avg_conv_plaid.min() else np.zeros_like(avg_conv_plaid)
        norm_conv_sine = (avg_conv_sine - avg_conv_sine.min()) / (avg_conv_sine.max() - avg_conv_sine.min()) if avg_conv_sine.max() > avg_conv_sine.min() else np.zeros_like(avg_conv_sine)

        normalized_rnn_plaid[idx] = norm_rnn_plaid
        normalized_rnn_sine[idx] = norm_rnn_sine
        normalized_conv_plaid[idx] = norm_conv_plaid
        normalized_conv_sine[idx] = norm_conv_sine

    # Mean and SEM across models
    mean_rnn_plaid = normalized_rnn_plaid.mean(axis=0)
    sem_rnn_plaid = normalized_rnn_plaid.std(axis=0) / np.sqrt(num_models)
    mean_rnn_sine = normalized_rnn_sine.mean(axis=0)
    sem_rnn_sine = normalized_rnn_sine.std(axis=0) / np.sqrt(num_models)

    mean_conv_plaid = normalized_conv_plaid.mean(axis=0)
    sem_conv_plaid = normalized_conv_plaid.std(axis=0) / np.sqrt(num_models)
    mean_conv_sine = normalized_conv_sine.mean(axis=0)
    sem_conv_sine = normalized_conv_sine.std(axis=0) / np.sqrt(num_models)

    # Rotate RNN angles
    rotated_angles_rnn = (angles + rotation_rnn_degrees) % 360
    rotated_angles_rnn_rad = np.deg2rad(rotated_angles_rnn)
    rotated_angles_rnn_rad = np.append(rotated_angles_rnn_rad, rotated_angles_rnn_rad[0])
    rotated_mean_rnn_plaid = np.append(mean_rnn_plaid, mean_rnn_plaid[0])
    rotated_mean_rnn_sine = np.append(mean_rnn_sine, mean_rnn_sine[0])
    rotated_sem_rnn_plaid = np.append(sem_rnn_plaid, sem_rnn_plaid[0])
    rotated_sem_rnn_sine = np.append(sem_rnn_sine, sem_rnn_sine[0])

    # Rotate Conv angles
    rotated_angles_conv = (angles + rotation_conv_degrees) % 360
    rotated_angles_conv_rad = np.deg2rad(rotated_angles_conv)
    rotated_angles_conv_rad = np.append(rotated_angles_conv_rad, rotated_angles_conv_rad[0])
    rotated_mean_conv_plaid = np.append(mean_conv_plaid, mean_conv_plaid[0])
    rotated_mean_conv_sine = np.append(mean_conv_sine, mean_conv_sine[0])
    rotated_sem_conv_plaid = np.append(sem_conv_plaid, sem_conv_plaid[0])
    rotated_sem_conv_sine = np.append(sem_conv_sine, sem_conv_sine[0])

    font_path = '../misc/fonts/Roboto-Regular.ttf'
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

    # Plot RNN averaged responses
    axs[0].plot(rotated_angles_rnn_rad, rotated_mean_rnn_plaid, color='blue', label='Plaid', marker='s', markersize=8)
    axs[0].fill_between(rotated_angles_rnn_rad,
                        rotated_mean_rnn_plaid - rotated_sem_rnn_plaid,
                        rotated_mean_rnn_plaid + rotated_sem_rnn_plaid,
                        color='blue', alpha=0.3)
    axs[0].plot(rotated_angles_rnn_rad, rotated_mean_rnn_sine, color='orange', label='Sine', marker='s', markersize=8)
    axs[0].fill_between(rotated_angles_rnn_rad,
                        rotated_mean_rnn_sine - rotated_sem_rnn_sine,
                        rotated_mean_rnn_sine + rotated_sem_rnn_sine,
                        color='orange', alpha=0.3)
    axs[0].set_ylim([0, 0.85])
    axs[0].set_yticklabels([])
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)

    # Plot Conv averaged responses
    axs[1].plot(rotated_angles_conv_rad, rotated_mean_conv_plaid, color='red', label='Plaid', marker='s', markersize=8)
    axs[1].fill_between(rotated_angles_conv_rad,
                        rotated_mean_conv_plaid - rotated_sem_conv_plaid,
                        rotated_mean_conv_plaid + rotated_sem_conv_plaid,
                        color='red', alpha=0.3)
    axs[1].plot(rotated_angles_conv_rad, rotated_mean_conv_sine, color='green', label='Sine', marker='s', markersize=8)
    axs[1].fill_between(rotated_angles_conv_rad,
                        rotated_mean_conv_sine - rotated_sem_conv_sine,
                        rotated_mean_conv_sine + rotated_sem_conv_sine,
                        color='green', alpha=0.3)
    axs[1].set_ylim([0, 1])
    axs[1].set_yticklabels([])
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig('../Saved_Images/polar_ver2.svg', format='svg')
    plt.show()


def plot_optimal_responses(models, angles, num_units, num_kernels, num_frames,
                           optimal_sf_tf_rnn_list, optimal_sf_tf_conv_list,
                           plaid_sequences, sine_sequences,
                           rotation_rnn_degrees=180, rotation_conv_degrees=180):
    """
    Collect activations from all models at their optimally-responsive SF/TF parameters, then plot the averaged responses.
    Also computes and prints the average peak sine values.
    """
    peak_sine_values_rnn = []
    peak_sine_values_conv = []

    for frame in range(num_frames):
        activations_rnn_plaid_list = []
        activations_conv_plaid_list = []
        activations_rnn_sine_list = []
        activations_conv_sine_list = []

        for model_idx, model in enumerate(models):
            optimal_sf_tf_rnn = optimal_sf_tf_rnn_list[model_idx]
            optimal_sf_tf_conv = optimal_sf_tf_conv_list[model_idx]

            activations_rnn_plaid = np.zeros((len(angles), num_units))
            activations_conv_plaid = np.zeros((len(angles), num_kernels))
            activations_rnn_sine = np.zeros((len(angles), num_units))
            activations_conv_sine = np.zeros((len(angles), num_kernels))

            # RNN unit activations at optimal SF/TF
            for unit in range(num_units):
                sf_idx, tf_idx = optimal_sf_tf_rnn[unit]
                for angle_idx in range(len(angles)):
                    plaid_sequence = plaid_sequences[sf_idx, tf_idx, angle_idx]
                    sine_sequence = sine_sequences[sf_idx, tf_idx, angle_idx]
                    plaid_input = torch.tensor(plaid_sequence, dtype=torch.float32).unsqueeze(0).to(device)
                    sine_input = torch.tensor(sine_sequence, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        _, _, _, rnn_outputs_plaid = model(plaid_input)
                        _, _, _, rnn_outputs_sine = model(sine_input)

                    activations_rnn_plaid[angle_idx, unit] = rnn_outputs_plaid[0, frame, unit].item()
                    activations_rnn_sine[angle_idx, unit] = rnn_outputs_sine[0, frame, unit].item()

            # Conv kernel activations at optimal SF/TF
            for kernel in range(num_kernels):
                sf_idx, tf_idx = optimal_sf_tf_conv[kernel]
                for angle_idx in range(len(angles)):
                    plaid_sequence = plaid_sequences[sf_idx, tf_idx, angle_idx]
                    sine_sequence = sine_sequences[sf_idx, tf_idx, angle_idx]
                    plaid_input = torch.tensor(plaid_sequence, dtype=torch.float32).unsqueeze(0).to(device)
                    sine_input = torch.tensor(sine_sequence, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        _, out_conv_plaid, _, _ = model(plaid_input)
                        _, out_conv_sine, _, _ = model(sine_input)

                    activations_conv_plaid[angle_idx, kernel] = out_conv_plaid[0, frame, kernel].mean().item()
                    activations_conv_sine[angle_idx, kernel] = out_conv_sine[0, frame, kernel].mean().item()

            activations_rnn_plaid_list.append(activations_rnn_plaid)
            activations_rnn_sine_list.append(activations_rnn_sine)
            activations_conv_plaid_list.append(activations_conv_plaid)
            activations_conv_sine_list.append(activations_conv_sine)

            # Peak sine responses for this model and frame
            peak_rnn_sine = activations_rnn_sine.max()
            peak_conv_sine = activations_conv_sine.max()

            peak_sine_values_rnn.append(peak_rnn_sine)
            peak_sine_values_conv.append(peak_conv_sine)

    # Average peak sine values across models and frames
    avg_peak_sine_rnn = np.mean(peak_sine_values_rnn)
    avg_peak_sine_conv = np.mean(peak_sine_values_conv)

    print(f"Average peak sine value for RNN (before normalization): {avg_peak_sine_rnn}")
    print(f"Average peak sine value for Conv (before normalization): {avg_peak_sine_conv}")

    # Plot averaged responses for the first frame as example
    frame_to_plot = 0
    plot_polar_responses(
        frame_to_plot,
        angles,
        activations_rnn_plaid_list,
        activations_rnn_sine_list,
        activations_conv_plaid_list,
        activations_conv_sine_list,
        rotation_rnn_degrees=rotation_rnn_degrees,
        rotation_conv_degrees=rotation_conv_degrees
    )


def plot_sequences(sine_sequences, plaid_sequences, direction_idx, sf_idx, tf_idx):
    """
    Plot a sample set of sine and plaid sequences for visualization purposes.
    """
    sine_seq = sine_sequences[sf_idx, tf_idx, direction_idx]
    plaid_seq = plaid_sequences[sf_idx, tf_idx, direction_idx]
    num_frames = sine_seq.shape[0]

    fig, axs = plt.subplots(4, num_frames, figsize=(20, 10))
    fig.suptitle(f'Sine and Plaid Sequences for 0 degrees\nSF index: {sf_idx}, TF index: {tf_idx}', fontsize=16)

    for frame in range(num_frames):
        # Sine Ch 0
        axs[0, frame].imshow(sine_seq[frame, 0], cmap='gray', vmin=0, vmax=1)
        axs[0, frame].axis('off')
        if frame == 0:
            axs[0, frame].set_ylabel('Sine Ch 0')

        # Sine Ch 1
        axs[1, frame].imshow(sine_seq[frame, 1], cmap='gray', vmin=0, vmax=1)
        axs[1, frame].axis('off')
        if frame == 0:
            axs[1, frame].set_ylabel('Sine Ch 1')

        # Plaid Ch 0
        axs[2, frame].imshow(plaid_seq[frame, 0], cmap='gray', vmin=0, vmax=1)
        axs[2, frame].axis('off')
        if frame == 0:
            axs[2, frame].set_ylabel('Plaid Ch 0')

        # Plaid Ch 1
        axs[3, frame].imshow(plaid_seq[frame, 1], cmap='gray', vmin=0, vmax=1)
        axs[3, frame].axis('off')
        if frame == 0:
            axs[3, frame].set_ylabel('Plaid Ch 1')

        axs[3, frame].set_xlabel(f'Frame {frame}')

    plt.tight_layout()
    plt.show()


def main():
    # Direction angles
    directions = np.linspace(0, 360, 12, endpoint=False)
    # Spatial and Temporal frequencies
    spatial_frequencies = np.linspace(0.02, 0.12, 25)
    temporal_frequencies = np.linspace(0.5, 6, 25)

    # Rotation variables (degrees)
    rotation_rnn_degrees = 90   # Adjust this value to rotate the RNN plot if desired
    rotation_conv_degrees = 220 # Adjust this value to rotate the Conv plot if desired

    # Load 10 models from specified paths
    models = []
    for i in range(1, 11):
        model_path = f'../Trained_Models/motionnet_full_30epochs_{i}.pt'
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping.")
            continue
        model = torch.load(model_path, map_location=device)
        model.eval()  # Set to evaluation mode
        models.append(model)

    if not models:
        print("No models loaded. Please check the model paths.")
        return

    # Generate sine and plaid sequences
    sine_sequences = generate_grating_sequences(IM_SIZE, IM_SIZE, directions, spatial_frequencies, temporal_frequencies, NUM_FRAMES, 'sine')
    plaid_sequences = generate_grating_sequences(IM_SIZE, IM_SIZE, directions, spatial_frequencies, temporal_frequencies, NUM_FRAMES, 'plaid')

    # Example indices for visualization
    direction_idx = np.argmin(np.abs(directions - 0))
    sf_idx = len(spatial_frequencies) // 2
    tf_idx = len(temporal_frequencies) // 2

    # Plot example sequences
    plot_sequences(sine_sequences, plaid_sequences, direction_idx, sf_idx, tf_idx)

    print("Generating combined plots for each frame:")

    # Find optimal SF/TF settings for each model
    optimal_sf_tf_rnn_list, optimal_sf_tf_conv_list = find_optimal_sf_tf(
        models, sine_sequences, RNN_UNITS, N_KERNELS, NUM_FRAMES
    )

    # Plot responses
    plot_optimal_responses(
        models,
        directions,
        RNN_UNITS,
        N_KERNELS,
        NUM_FRAMES,
        optimal_sf_tf_rnn_list,
        optimal_sf_tf_conv_list,
        plaid_sequences,
        sine_sequences,
        rotation_rnn_degrees=rotation_rnn_degrees,
        rotation_conv_degrees=rotation_conv_degrees
    )


if __name__ == "__main__":
    main()
