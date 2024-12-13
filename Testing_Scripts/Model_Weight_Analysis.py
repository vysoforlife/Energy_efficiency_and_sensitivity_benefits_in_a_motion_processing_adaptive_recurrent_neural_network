import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from scipy.fft import fft2, fftshift
import math
from matplotlib import font_manager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IM_SIZE, NUM_FRAMES, BATCH_SIZE = 32, 9, 32
N_KERNELS, KERNEL_SIZE, STRIDE_SIZE = 16, 6, 1
RNN_UNITS, ACT_SIZE, NOISE_LEVEL = 16, 27, 0.3
# Parameters
im_size = 32
num_of_frames = 9
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27
noise_level = 0.5

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


class Model_n(nn.Module):
    def __init__(self):
        super(Model_n, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.conv = nn.Conv2d(2, N_KERNELS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), stride=(STRIDE_SIZE, STRIDE_SIZE))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(input_size=ACT_SIZE * ACT_SIZE * N_KERNELS, hidden_size=RNN_UNITS, nonlinearity='relu',
                               batch_first=True).to(device)
        self.fc = nn.Linear(RNN_UNITS, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True, batch_first=False, adaptation_rate=0.2,
                 recovery_rate=0.1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate

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

        # Initialize adaptation
        adaptation = torch.zeros_like(hx)

        output = []
        for t in range(seq_len):
            x_t = x[t]

            # Compute pre-activation
            pre_activation = x_t @ self.weight_ih.t() + self.bias_ih + hx @ self.weight_hh.t() + self.bias_hh

            # Apply adaptation
            adapted_activation = torch.max(torch.zeros_like(pre_activation), pre_activation - adaptation)

            # Update adaptation
            adaptation += self.adaptation_rate * adapted_activation
            adaptation -= self.recovery_rate * adaptation

            # Apply nonlinearity
            hx = self.activation(adapted_activation)

            output.append(hx)

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)  # Convert seq-first back to batch-first

        return output, hx


class AdaptiveLayer(nn.Module):
    def __init__(self, num_features, adaptation_rate=0.3, recovery_rate=0.1):
        super(AdaptiveLayer, self).__init__()
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate
        self.adaptation = nn.Parameter(torch.zeros(1, num_features), requires_grad=False)

    def forward(self, x):
        # x shape: [batch_size, num_frames, num_features]
        batch_size, num_frames, num_features = x.size()

        # Expand adaptation to match batch size
        adaptation = self.adaptation.expand(batch_size, -1)

        adapted_output = []
        for i in range(num_frames):
            frame = x[:, i, :]
            output = torch.clamp(frame - adaptation, min=0)
            adaptation = adaptation + self.adaptation_rate * output
            adaptation = adaptation * (1 - self.recovery_rate)
            adapted_output.append(output)

        # Update the stored adaptation with the mean across the batch
        self.adaptation.data = adaptation.mean(dim=0, keepdim=True)

        return torch.stack(adapted_output, dim=1)


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

        # Add the AdaptiveLayer here
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

        # Apply the AdaptiveLayer to the flattened output
        out_adapted = self.adaptive_layer(out_flat)

        # Use the adapted output for the RNN
        out_rnn, _ = self.rnn(out_adapted)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)

        return out_fc, out_conv, out_flat, out_adapted, out_rnn

def generate_grating(width, height, frequency, direction, num_frames, speed, is_plaid=False):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    angle_rad = np.deg2rad(direction)
    frames = []

    for i in range(num_frames + 1):
        phase_shift = i * 2 * np.pi * speed / num_frames
        grating1 = np.sin(2 * np.pi * frequency * (x * np.cos(angle_rad) + y * np.sin(angle_rad) + phase_shift))

        if is_plaid:
            angle_rad2 = np.deg2rad((direction + 90) % 360)
            grating2 = np.sin(2 * np.pi * frequency * (x * np.cos(angle_rad2) + y * np.sin(angle_rad2) + phase_shift))
            grating = (grating1 + grating2) / 2
        else:
            grating = grating1

        grating = ((grating + 1) / 2 * 255).astype(np.uint8)
        frames.append(grating)

    return np.array(frames)


def generate_grating_sequences(width, height, directions, spatial_frequencies, temporal_frequencies, num_frames,
                               stimulus_type='plaid'):
    n_sf, n_tf, n_dir = len(spatial_frequencies), len(temporal_frequencies), len(directions)
    sequences = np.zeros((n_sf, n_tf, n_dir, num_frames, 2, width, height))
    for i, sf in enumerate(spatial_frequencies):
        for j, tf in enumerate(temporal_frequencies):
            for k, direction in enumerate(directions):
                frames = generate_grating(width, height, sf, direction, num_frames, tf,
                                          is_plaid=(stimulus_type == 'plaid'))
                sequences[i, j, k, :, 0] = frames[:-1]
                sequences[i, j, k, :, 1] = frames[1:]
    return sequences

def analyze_weights(model, sequences, directions, num_frames, analysis_type):
    device = next(model.parameters()).device
    num_units = model.rnn.hidden_size if analysis_type == 'rnn' else N_KERNELS

    if analysis_type == 'rnn':
        preferred_directions = []
        for frame in [0, 1]:
            frame_preferences = []
            for dir_idx, direction in enumerate(directions):
                sample_input = torch.tensor(sequences[:, :, dir_idx], dtype=torch.float32).squeeze(0).to(device)
                with torch.no_grad():
                    if isinstance(model, Model_n):
                        _, _, _, rnn_outputs = model(sample_input)
                    elif isinstance(model, Model):
                        _, _, _, _, rnn_outputs = model(sample_input)
                    else:
                        raise ValueError(f"Unknown model type: {type(model)}")
                frame_preferences.append(rnn_outputs[0, frame].cpu().numpy())
            preferred_directions.append(np.argmax(np.array(frame_preferences), axis=0))

        weights = model.rnn.weight_hh.cpu().detach().numpy()
    else:  # conv_rnn
        conv_preferences = []
        rnn_preferences = []

        for dir_idx, direction in enumerate(directions):
            sample_input = torch.tensor(sequences[:, :, dir_idx], dtype=torch.float32).squeeze(0).to(device)

            with torch.no_grad():
                if isinstance(model, Model_n):
                    _, out_conv, _, rnn_outputs = model(sample_input)
                elif isinstance(model, Model):
                    _, out_conv, _, _, rnn_outputs = model(sample_input)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")

            conv_responses = out_conv[0, 0].mean(dim=(1, 2)).cpu().numpy()
            rnn_responses = rnn_outputs[0, 1].cpu().numpy()

            conv_preferences.append(conv_responses)
            rnn_preferences.append(rnn_responses)

        conv_preferences = np.array(conv_preferences)
        rnn_preferences = np.array(rnn_preferences)

        preferred_directions = [np.argmax(conv_preferences, axis=0), np.argmax(rnn_preferences, axis=0)]
        weights = model.rnn.weight_ih.cpu().detach().numpy()

    dir_weights = np.zeros((len(directions), len(directions)))

    for i in range(len(directions)):
        for j in range(len(directions)):
            dir_i_neurons = np.where(preferred_directions[0] == i)[0]
            dir_j_neurons = np.where(preferred_directions[1] == j)[0]
            if len(dir_i_neurons) > 0 and len(dir_j_neurons) > 0:
                if analysis_type == 'rnn':
                    weights_subset = weights[dir_j_neurons][:, dir_i_neurons]
                else:  # conv_rnn
                    conv_inputs_per_unit = weights.shape[1] // num_units
                    conv_inputs = np.concatenate(
                        [np.arange(unit * conv_inputs_per_unit, (unit + 1) * conv_inputs_per_unit) for unit in
                         dir_i_neurons])
                    weights_subset = weights[dir_j_neurons][:, conv_inputs]
                dir_weights[i, j] = np.mean(weights_subset)

    angular_diffs = []
    for i in range(len(directions)):
        for j in range(len(directions)):
            diff = (directions[j] - directions[i]) % 360
            if diff > 180:
                diff -= 360
            angular_diffs.append(diff)

    avg_weights = [dir_weights[i, j] for i in range(len(directions)) for j in range(len(directions))]

    unique_diffs = np.unique(angular_diffs)
    final_avg_weights = [np.mean([w for d, w in zip(angular_diffs, avg_weights) if d == diff]) for diff in unique_diffs]

    # Add the 180-degree value to -180 degrees
    if -180 not in unique_diffs:
        unique_diffs = np.concatenate(([-180], unique_diffs))
        final_avg_weights = np.concatenate(([final_avg_weights[-1]], final_avg_weights))

    return unique_diffs, final_avg_weights

def analyze_and_plot_weights(models_type1, models_type2, im_size, num_frames, analysis_type='rnn'):
    directions = np.linspace(0, 360, 8, endpoint=False)
    spatial_frequencies = [0.07]
    temporal_frequencies = [4.75]

    sequences = generate_grating_sequences(im_size, im_size, directions, spatial_frequencies, temporal_frequencies,
                                           num_frames, stimulus_type='sine')

    all_diffs_type1, all_weights_type1 = [], []
    all_diffs_type2, all_weights_type2 = [], []

    for models, all_diffs, all_weights in [(models_type1, all_diffs_type1, all_weights_type1),
                                           (models_type2, all_diffs_type2, all_weights_type2)]:
        for i, model in enumerate(models):
            print(f"Analyzing model {i + 1}")
            diffs, weights = analyze_weights(model, sequences, directions, num_frames, analysis_type)
            all_diffs.append(diffs)
            all_weights.append(weights)

    avg_weights_type1 = np.mean(all_weights_type1, axis=0)
    std_weights_type1 = np.std(all_weights_type1, axis=0)
    avg_weights_type2 = np.mean(all_weights_type2, axis=0)
    std_weights_type2 = np.std(all_weights_type2, axis=0)

    return (all_diffs_type1[0], avg_weights_type1, std_weights_type1), (
    all_diffs_type2[0], avg_weights_type2, std_weights_type2)

def analyze_rnn_to_linear_weights(models_type1, models_type2, im_size, num_frames=9):
    device = next(models_type1[0].parameters()).device
    num_models = len(models_type1)
    num_units = models_type1[0].rnn.hidden_size
    num_dir_categories = 8

    spatial_frequency = 0.07
    temporal_frequency = 4.75
    directions = np.linspace(0, 360, 24, endpoint=False)

    results = []

    for models in [models_type1, models_type2]:
        weights_x = np.zeros((num_models, num_dir_categories))
        weights_y = np.zeros((num_models, num_dir_categories))

        for model_idx, model in enumerate(models):
            print(f"Analyzing model {model_idx + 1}")

            rnn_preferences = []
            for direction in directions:
                sequence = generate_grating_sequences(im_size, im_size, [direction], [spatial_frequency],
                                                      [temporal_frequency], num_frames, 'sine')
                sample_input = torch.tensor(sequence, dtype=torch.float32).squeeze(0).squeeze(0).to(device)

                with torch.no_grad():
                    if isinstance(model, Model_n):
                        _, _, _, rnn_outputs = model(sample_input)
                    elif isinstance(model, Model):
                        _, _, _, _, rnn_outputs = model(sample_input)
                    else:
                        raise ValueError(f"Unknown model type: {type(model)}")

                rnn_responses = rnn_outputs[0].mean(dim=0).cpu().numpy()
                rnn_preferences.append(rnn_responses)

            rnn_preferences = np.array(rnn_preferences)
            preferred_directions = np.argmax(rnn_preferences, axis=0)

            rnn_to_linear_weights = model.fc.weight.cpu().detach().numpy()

            for dir_cat in range(num_dir_categories):
                dir_neurons = np.where((preferred_directions >= dir_cat * len(directions) // num_dir_categories) &
                                       (preferred_directions < (dir_cat + 1) * len(
                                           directions) // num_dir_categories))[0]

                if len(dir_neurons) > 0:
                    weights_x[model_idx, dir_cat] = np.mean(rnn_to_linear_weights[0, dir_neurons])
                    weights_y[model_idx, dir_cat] = np.mean(rnn_to_linear_weights[1, dir_neurons])

        avg_weights_x = np.mean(weights_x, axis=0)
        avg_weights_y = np.mean(weights_y, axis=0)
        std_weights_x = np.std(weights_x, axis=0)
        std_weights_y = np.std(weights_y, axis=0)

        results.append((avg_weights_x, avg_weights_y, std_weights_x, std_weights_y))

    return results


def plot_all_analyses(rnn_results, conv_rnn_results, rnn_to_linear_results):
    # Font configuration
    font_path = '../data/fonts/Roboto-Regular.ttf'  # Update this path
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams.update({
        'font.size': 35,
        'axes.titlesize': 35,
        'axes.labelsize': 35,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'axes.linewidth': 1  # Increase axis line thickness
    })

    fig, axs = plt.subplots(2, 2, figsize=(32, 18))  # More horizontal aspect ratio

    # Define colors for consistency
    colors = ['blue', 'orange']

    # Function to set up common axis properties
    def setup_axis(ax, xlabel, ylabel, title, center_y_axis=False, bottom_graph=False):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if bottom_graph:
            ax.set_xticks(np.arange(0, 361, 45))
            ax.set_xlim(0, 360)
        else:
            ax.set_xticks(np.arange(-180, 181, 45))
            ax.set_xlim(-180, 180)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=2, length=20)

        if center_y_axis:
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('axes', 0))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_label_coords(-0.05, 0.5)

    # RNN Weight Flow Analysis
    rnn_type1, rnn_type2 = rnn_results
    for i, (rnn_type, color, label) in enumerate(zip([rnn_type1, rnn_type2], colors, ['Motionnet-R', 'AdaptNet'])):
        axs[0, 0].plot(rnn_type[0], rnn_type[1], color=color, label=label, linewidth=2)
        axs[0, 0].fill_between(rnn_type[0], rnn_type[1] - rnn_type[2], rnn_type[1] + rnn_type[2], alpha=0.3,
                               color=color)
    setup_axis(axs[0, 0], 'Angular Difference (degrees)', 'Average Weight (arb. units)', '', center_y_axis=True)

    # Conv-to-RNN Weight Flow Analysis
    conv_rnn_type1, conv_rnn_type2 = conv_rnn_results
    for i, (conv_rnn_type, color, label) in enumerate(
            zip([conv_rnn_type1, conv_rnn_type2], colors, ['Motionnet-R', 'AdaptNet'])):
        axs[0, 1].plot(conv_rnn_type[0], conv_rnn_type[1], color=color, label=label, linewidth=2)
        axs[0, 1].fill_between(conv_rnn_type[0], conv_rnn_type[1] - conv_rnn_type[2],
                               conv_rnn_type[1] + conv_rnn_type[2], alpha=0.3, color=color)
    setup_axis(axs[0, 1], 'Angular Difference (degrees)', 'Average Weight (arb. units)', '', center_y_axis=True)

    # RNN to Linear Layer Weights (X output)
    results_type1, results_type2 = rnn_to_linear_results
    avg_weights_x1, avg_weights_y1, std_weights_x1, std_weights_y1 = results_type1
    avg_weights_x2, avg_weights_y2, std_weights_x2, std_weights_y2 = results_type2
    directions_0_360 = np.linspace(0, 360, len(avg_weights_x1) + 1, endpoint=True)

    for i, (avg_weights_x, std_weights_x, color, label) in enumerate(
            zip([avg_weights_x1, avg_weights_x2], [std_weights_x1, std_weights_x2], colors,
                ['Motionnet-R', 'AdaptNet'])):
        axs[1, 0].plot(directions_0_360, np.append(avg_weights_x, avg_weights_x[0]), color=color, label=label,
                       linewidth=2)
        axs[1, 0].fill_between(directions_0_360,
                               np.append(avg_weights_x - std_weights_x, avg_weights_x[0] - std_weights_x[0]),
                               np.append(avg_weights_x + std_weights_x, avg_weights_x[0] + std_weights_x[0]), alpha=0.3,
                               color=color)
    setup_axis(axs[1, 0], 'Direction (degrees)', 'Average Weight (arb. units)', '', bottom_graph=True)

    # RNN to Linear Layer Weights (Y output)
    for i, (avg_weights_y, std_weights_y, color, label) in enumerate(
            zip([avg_weights_y1, avg_weights_y2], [std_weights_y1, std_weights_y2], colors,
                ['Motionnet-R', 'AdaptNet'])):
        axs[1, 1].plot(directions_0_360, np.append(avg_weights_y, avg_weights_y[0]), color=color, label=label,
                       linewidth=2)
        axs[1, 1].fill_between(directions_0_360,
                               np.append(avg_weights_y - std_weights_y, avg_weights_y[0] - std_weights_y[0]),
                               np.append(avg_weights_y + std_weights_y, avg_weights_y[0] + std_weights_y[0]), alpha=0.3,
                               color=color)
    setup_axis(axs[1, 1], 'Direction (degrees)', 'Average Weight (arb. units)', '', bottom_graph=True)

    # Add a single legend for all subplots at the bottom center
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=25)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.2)  # Adjust margins and spacing
    plt.savefig('flowdiag_ver2.svg', format='svg')
    plt.show()

def main():
    directions = np.linspace(0, 360, 24, endpoint=False)
    spatial_frequencies = np.linspace(0.02, 0.12, 25)
    temporal_frequencies = np.linspace(0.5, 6, 25)

    models_type1 = [torch.load(f'../models/motionnet_fu2_30epochs_{i}.pt', map_location=device) for i in
                    range(1, 11)]
    models_type2 = [torch.load(f'../models/adaptnet_fu2_30epochs_{i}_04.pt', map_location=device) for i in
                    range(1, 11)]

    # Ensure models are instances of the correct classes
    for model in models_type1:
        if not isinstance(model, Model_n):
            raise TypeError(f"Expected Model_n, but got {type(model)}")

    for model in models_type2:
        if not isinstance(model, Model):
            raise TypeError(f"Expected Model, but got {type(model)}")

    sine_sequences = generate_grating_sequences(IM_SIZE, IM_SIZE, directions, spatial_frequencies,
                                                temporal_frequencies,
                                                NUM_FRAMES, 'sine')
    plaid_sequences = generate_grating_sequences(IM_SIZE, IM_SIZE, directions, spatial_frequencies,
                                                 temporal_frequencies, NUM_FRAMES, 'plaid')

    print("Analyzing RNN weights for all models:")
    try:
        rnn_results = analyze_and_plot_weights(models_type1, models_type2, IM_SIZE, NUM_FRAMES, analysis_type='rnn')
        np.savez('rnn_weight_analysis_results.npz',
                 type1_diffs=rnn_results[0][0], type1_avg=rnn_results[0][1], type1_std=rnn_results[0][2],
                 type2_diffs=rnn_results[1][0], type2_avg=rnn_results[1][1], type2_std=rnn_results[1][2])
    except Exception as e:
        print(f"Error in analyze_and_plot_weights (RNN): {e}")
        print("Model structure:")
        print(models_type1[0])
        raise

    print("Analyzing Conv-to-RNN weights for all models:")
    try:
        conv_rnn_results = analyze_and_plot_weights(models_type1, models_type2, IM_SIZE, NUM_FRAMES,
                                                    analysis_type='conv_rnn')
        np.savez('conv_rnn_weight_analysis_results.npz',
                 type1_diffs=conv_rnn_results[0][0], type1_avg=conv_rnn_results[0][1],
                 type1_std=conv_rnn_results[0][2],
                 type2_diffs=conv_rnn_results[1][0], type2_avg=conv_rnn_results[1][1],
                 type2_std=conv_rnn_results[1][2])
    except Exception as e:
        print(f"Error in analyze_and_plot_weights (Conv-RNN): {e}")
        print("Model structure:")
        print(models_type1[0])
        raise

    print("Analyzing RNN to linear layer weights:")
    try:
        rnn_to_linear_results = analyze_rnn_to_linear_weights(models_type1, models_type2, IM_SIZE, NUM_FRAMES)
        np.savez('rnn_to_linear_weight_analysis_results.npz',
                 type1_x=rnn_to_linear_results[0][0], type1_y=rnn_to_linear_results[0][1],
                 type1_std_x=rnn_to_linear_results[0][2], type1_std_y=rnn_to_linear_results[0][3],
                 type2_x=rnn_to_linear_results[1][0], type2_y=rnn_to_linear_results[1][1],
                 type2_std_x=rnn_to_linear_results[1][2], type2_std_y=rnn_to_linear_results[1][3])
    except Exception as e:
        print(f"Error in analyze_rnn_to_linear_weights: {e}")
        print("Model structure:")
        print(models_type1[0])
        raise

    # Plot all analyses in a single figure
    plot_all_analyses(rnn_results, conv_rnn_results, rnn_to_linear_results)

if __name__ == "__main__":
    main()