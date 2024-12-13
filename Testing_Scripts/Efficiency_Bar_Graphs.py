import math
import random
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch import nn
import itertools
from tqdm import tqdm
import matplotlib.font_manager as font_manager
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def perform_t_test(data1, data2, label):
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    print(f"\nIndependent t-test results for {label}:")
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "(not significant)"

    print(f"Significance: {significance}")

    if p_value < 0.05:
        print("The difference is statistically significant")
    else:
        print("The difference is not statistically significant")

def generate_moving_dot_sequence(initial_x_speed, initial_y_speed, dot_radius, num_frames, im_height, im_width,
                                 change_mode=False, num_changes=0, timesteps_between_changes=0, change_speeds=None):
    sequence = torch.zeros(num_frames, im_height, im_width)

    x = im_width // 2
    y = im_height // 2

    x_speed = initial_x_speed
    y_speed = initial_y_speed

    change_points = []
    if change_mode and num_changes > 0:
        change_points = [i * timesteps_between_changes for i in range(1, num_changes + 1)]
        if change_points[-1] >= num_frames:
            raise ValueError("Too many changes for the given number of frames")

    change_index = 0
    for t in range(num_frames):
        if change_mode and change_index < len(change_points) and t == change_points[change_index]:
            x_speed, y_speed = change_speeds[change_index]
            change_index += 1

        x = (x + x_speed) % im_width
        y = (y + y_speed) % im_height

        yy, xx = torch.meshgrid(torch.arange(im_height), torch.arange(im_width))

        dx = torch.min(torch.abs(xx - x), im_width - torch.abs(xx - x))
        dy = torch.min(torch.abs(yy - y), im_height - torch.abs(yy - y))

        d = torch.sqrt(dx ** 2 + dy ** 2)

        dot = (d <= dot_radius).float() * 255

        sequence[t] = dot

    network_input = torch.zeros(num_frames - 1, 2, im_height, im_width)
    for i in range(num_frames - 1):
        network_input[i, 0] = sequence[i]
        network_input[i, 1] = sequence[i + 1]

    network_input = network_input.unsqueeze(0)

    return network_input, change_points


# Example usage:
x_speed = 2
y_speed = 1
dot_radius = 3
num_frames = 40
im_height = 32
im_width = 32

change_mode = True
num_changes = 3
timesteps_between_changes = 10
change_speeds = [(-1, 2), (3, -1), (0, 3)]

input_sequence, change_points = generate_moving_dot_sequence(
    x_speed, y_speed, dot_radius, num_frames, im_height, im_width,
    change_mode, num_changes, timesteps_between_changes, change_speeds
)

print(f"Change points: {change_points}")

input_sequence = input_sequence.to(device)

im_size = 32
num_of_frames = 40
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27


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
        return out_fc, out_conv, out_flat, out_rnn


def analyze_responsiveness_and_efficiency(predictions, true_speeds, change_points, window_size=3, threshold=0.1):
    individual_results = []
    overall_results = {
        'avg_response_time': 0,
        'avg_settling_time': 0,
        'avg_overshoot': 0,
        'avg_mse': 0,
        'efficiency_ratio': 0,
        'snr': 0,
        'prediction_energy': 0,
        'normalized_prediction_energy': 0,
        'change_efficiency': 0,
        'total_prediction_magnitude': 0
    }

    true_speeds_aligned = true_speeds[1:]

    total_prediction_magnitude = np.sum(np.abs(predictions))
    prediction_energy = np.sum(predictions ** 2)
    num_frames = len(predictions)

    correct_changes = 0
    total_changes = 0

    for i, change_point in enumerate(change_points):
        old_speed = true_speeds[change_point - 1]
        new_speed = true_speeds[change_point]

        pred_window = predictions[change_point - 1:change_point - 1 + window_size]
        true_window = true_speeds_aligned[change_point - 1:change_point - 1 + window_size]

        min_length = min(len(pred_window), len(true_window))
        pred_window = pred_window[:min_length]
        true_window = true_window[:min_length]

        response_time = next((i for i, p in enumerate(pred_window) if abs(p - new_speed) <= threshold), min_length)

        settling_time = next((i for i in range(len(pred_window))
                              if all(abs(p - new_speed) <= threshold for p in pred_window[i:])), min_length)

        overshoot = max(abs(p - new_speed) for p in pred_window)

        mse = np.mean((pred_window - true_window) ** 2)

        if i > 0:
            pred_change = predictions[change_point - 1] - predictions[change_point - 2]
            true_change = new_speed - old_speed
            if np.sign(pred_change) == np.sign(true_change):
                correct_changes += 1
            total_changes += 1

        result = {
            'change_point': change_point,
            'old_speed': old_speed,
            'new_speed': new_speed,
            'response_time': response_time,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'mse': mse
        }
        individual_results.append(result)

        overall_results['avg_response_time'] += response_time
        overall_results['avg_settling_time'] += settling_time
        overall_results['avg_overshoot'] += overshoot
        overall_results['avg_mse'] += mse

    num_changes = len(change_points)
    for key in ['avg_response_time', 'avg_settling_time', 'avg_overshoot', 'avg_mse']:
        overall_results[key] /= num_changes

    overall_results['efficiency_ratio'] = overall_results['avg_response_time'] / total_prediction_magnitude
    overall_results['snr'] = np.var(true_speeds_aligned) / np.var(predictions - true_speeds_aligned)
    overall_results['prediction_energy'] = prediction_energy
    overall_results['normalized_prediction_energy'] = prediction_energy / num_frames
    overall_results['change_efficiency'] = correct_changes / total_changes if total_changes > 0 else 0
    overall_results['total_prediction_magnitude'] = total_prediction_magnitude

    return individual_results, overall_results


def generate_change_combinations(num_changes, speed_range):
    speeds = list(range(speed_range[0], speed_range[1] + 1))
    return list(itertools.product(speeds, repeat=num_changes + 1))


def calculate_total_change_magnitude(speeds):
    return sum(abs(speeds[i + 1] - speeds[i]) for i in range(len(speeds) - 1))


def generate_moving_dot_sequence_batch(initial_speeds, change_speeds, dot_radius, num_frames, im_height, im_width,
                                       timesteps_between_changes):
    batch_size = len(initial_speeds)
    network_input = torch.zeros(batch_size, num_frames - 1, 2, im_height, im_width, device=device)

    x = torch.tensor([speed[0] for speed in initial_speeds], device=device).float()
    y = torch.tensor([speed[1] for speed in initial_speeds], device=device).float()

    speeds = torch.tensor(initial_speeds, device=device).float()

    yy, xx = torch.meshgrid(torch.arange(im_height, device=device), torch.arange(im_width, device=device))
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)

    for t in range(num_frames):
        if t > 0 and t % timesteps_between_changes == 0:
            change_idx = t // timesteps_between_changes - 1
            if change_idx < len(change_speeds[0]):
                speeds = torch.tensor([cs[change_idx] for cs in change_speeds], device=device).float()

        x = (x + speeds[:, 0]) % im_width
        y = (y + speeds[:, 1]) % im_height

        dx = torch.min(torch.abs(xx - x.unsqueeze(1).unsqueeze(2)),
                       im_width - torch.abs(xx - x.unsqueeze(1).unsqueeze(2)))
        dy = torch.min(torch.abs(yy - y.unsqueeze(1).unsqueeze(2)),
                       im_height - torch.abs(yy - y.unsqueeze(1).unsqueeze(2)))

        d = torch.sqrt(dx ** 2 + dy ** 2)
        dot = (d <= dot_radius).float() * 255

        if t < num_frames - 1:
            network_input[:, t, 0] = dot
        if t > 0:
            network_input[:, t - 1, 1] = dot

    return network_input


def batch_forward(model, x):
    batch_size, num_frames, _, _, _ = x.size()
    out_conv = model.conv(x.view(-1, 2, 32, 32)).view(batch_size, num_frames, -1)
    out_flat = model.flatten(out_conv)
    out_rnn, _ = model.rnn(out_flat)
    out_fc = model.fc(out_rnn)
    return out_fc


def run_simulation_batch(models, initial_speeds, change_speeds, num_frames, im_height, im_width, dot_radius,
                         timesteps_between_changes):
    input_sequence = generate_moving_dot_sequence_batch(
        initial_speeds, change_speeds, dot_radius, num_frames, im_height, im_width, timesteps_between_changes
    )

    all_predictions = []
    for model in models:
        with torch.no_grad():
            predictions = batch_forward(model, input_sequence)
        all_predictions.append(predictions.cpu())

    avg_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    total_prediction_magnitudes_x = torch.sum(torch.abs(avg_predictions[:, :, 0]), dim=1).numpy()
    total_prediction_magnitudes_y = torch.sum(torch.abs(avg_predictions[:, :, 1]), dim=1).numpy()

    return total_prediction_magnitudes_x, total_prediction_magnitudes_y


def calculate_mse(predictions, true_speeds, change_points, window_size):
    mse_values = []
    for change_point in change_points:
        pred_window = predictions[change_point:change_point + window_size]
        true_window = true_speeds[change_point:change_point + window_size]
        mse = torch.mean((pred_window - true_window) ** 2).item()
        mse_values.append(mse)
    return np.mean(mse_values)


def plot_sequence(sequence, predictions_set1, predictions_set2, true_speeds, title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(title)

    axes[0].imshow(sequence[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title('Input Sequence (First Frame)')
    axes[0].axis('off')

    t = np.arange(len(predictions_set1))
    axes[1].plot(t, predictions_set1[:, 0], 'b-', label='Motionnet-R X')
    axes[1].plot(t, predictions_set1[:, 1], 'g-', label='Motionnet-R Y')
    axes[1].plot(t, predictions_set2[:, 0], 'r--', label='AdaptNet X')
    axes[1].plot(t, predictions_set2[:, 1], 'm--', label='AdaptNet Y')
    axes[1].plot(t, true_speeds[:, 0], 'k-', label='True X')
    axes[1].plot(t, true_speeds[:, 1], 'k--', label='True Y')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Speed')
    axes[1].legend()
    axes[1].set_title('Speed Predictions')

    x, y = 16, 16
    ax = axes[2]
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_aspect('equal')
    ax.set_title('Trajectory')

    colors = ['b', 'r', 'g']
    for i, speeds in enumerate([true_speeds, predictions_set1, predictions_set2]):
        x, y = 16, 16
        path_x, path_y = [x], [y]
        for dx, dy in speeds:
            x = (x + dx) % 32
            y = (y + dy) % 32
            path_x.append(x)
            path_y.append(y)
        ax.plot(path_x, path_y, f'{colors[i]}-', label=['True', 'Motionnet-R', 'AdaptNet'][i])
        ax.add_patch(Circle((path_x[-1], path_y[-1]), 0.5, color=colors[i]))

    ax.legend()
    plt.tight_layout()
    plt.show()


def remove_borders(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager



# Plot comparison graphs with Roboto font applied
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager



# Plot comparison graphs with Roboto font applied
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager



def plot_comparison_graphs(linear_data, whole_data):
    rnn_linear_set1, rnn_linear_set2, mse_linear_set1, mse_linear_set2 = linear_data
    rnn_whole_set1, rnn_whole_set2, mse_whole_set1, mse_whole_set2 = whole_data

    rnn_means = [np.mean(rnn_linear_set1), np.mean(rnn_linear_set2), np.mean(rnn_whole_set1), np.mean(rnn_whole_set2)]
    mse_means = [np.mean(mse_linear_set1), np.mean(mse_linear_set2), np.mean(mse_whole_set1), np.mean(mse_whole_set2)]

    rnn_errors = [np.std(rnn_linear_set1) / np.sqrt(10), np.std(rnn_linear_set2) / np.sqrt(10),
                  np.std(rnn_whole_set1) / np.sqrt(10), np.std(rnn_whole_set2) / np.sqrt(10)]
    mse_errors = [np.std(mse_linear_set1) / np.sqrt(10), np.std(mse_linear_set2) / np.sqrt(10),
                  np.std(mse_whole_set1) / np.sqrt(10), np.std(mse_whole_set2) / np.sqrt(10)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))  # Set figure size to 30x15
    # fig.suptitle('Comparison of RNN Responses and MSE for Change and Constant Cases', fontsize=35)

    x = np.arange(2)
    width = 0.35
    # Ensure the font is available
    font_path = '../../data/fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 50  # Set all font sizes to 35
    plt.rcParams['xtick.labelsize'] = 50  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 50  # Y-axis tick labels
    plt.rcParams['axes.titlesize'] = 50  # Title size
    plt.rcParams['axes.labelsize'] = 50  # X and Y labels
    plt.rcParams['legend.fontsize'] = 50  # Legend font size

    rects1 = ax1.bar(x - width / 2, [rnn_means[0], rnn_means[2]], width, label='Motionnet-R',
                     yerr=[rnn_errors[0], rnn_errors[2]], capsize=5)
    rects2 = ax1.bar(x + width / 2, [rnn_means[1], rnn_means[3]], width, label='AdaptNet',
                     yerr=[rnn_errors[1], rnn_errors[3]], capsize=5)

    ax1.set_ylabel('Average RNN Response\n(arb. units)', fontsize=50)
    ax1.set_title('RNN Responses', fontsize=50)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['With Change\n***', 'Constant\n***'], fontsize=50)
    ax1.tick_params(axis='y', labelsize=50)  # Set y-axis tick label size
    ax1.legend(fontsize=50)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', width=2, length=20)  # Add this line

    rects3 = ax2.bar(x - width / 2, [mse_means[0], mse_means[2]], width, label='Motionnet-R',
                     yerr=[mse_errors[0], mse_errors[2]], capsize=5)
    rects4 = ax2.bar(x + width / 2, [mse_means[1], mse_means[3]], width, label='AdaptNet',
                     yerr=[mse_errors[1], mse_errors[3]], capsize=5)
    ax2.set_ylabel('Average MSE\n(pixels per frame)', fontsize=50)
    ax2.set_title('Mean Squared Error', fontsize=35)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['With Change\n***', 'Constant\n***'], fontsize=50)
    ax2.tick_params(axis='y', labelsize=50)  # Set y-axis tick label size
    ax2.legend(fontsize=50)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', width=2, length=20)  # Add this line

    plt.tight_layout()
    plt.savefig('MSEpower_ver2.svg', format='svg')
    plt.show()



def main():
    x_speed = 3
    y_speed = -3
    dot_radius = 3
    num_frames = 40
    im_height = 32
    im_width = 32
    window_size = 3

    change_mode = True
    num_changes = 3
    timesteps_between_changes = 10
    change_speeds = [(-1, 2), (3, -1), (0, 3), (0, 3)]

    input_sequence, change_points = generate_moving_dot_sequence(
        x_speed, y_speed, dot_radius, num_frames, im_height, im_width,
        change_mode, num_changes, timesteps_between_changes, change_speeds
    )

    print(f"Change points: {change_points}")

    input_sequence = input_sequence.to(device)

    num_models = 10
    models_set1 = []
    models_set2 = []
    for i in range(1, num_models + 1):
        model1 = torch.load(f'../../models/motionnet_full_30epochs_{i}.pt', map_location=device)
        model1.eval()
        model1.to(device)
        models_set1.append(model1)

        model2 = torch.load(f'../../models/adaptnet_full_30epochs_{i}.pt', map_location=device)
        model2.eval()
        model2.to(device)
        models_set2.append(model2)

    speed_range = (-2, 2)
    x_combinations = list(itertools.product(range(speed_range[0], speed_range[1] + 1), repeat=num_changes + 1))
    y_combinations = list(itertools.product(range(speed_range[0], speed_range[1] + 1), repeat=num_changes + 1))

    results_x_set1, results_y_set1 = {}, {}
    results_x_set2, results_y_set2 = {}, {}
    mse_x_set1, mse_y_set1 = {}, {}
    mse_x_set2, mse_y_set2 = {}, {}

    mse_x_increasing_set1, mse_y_increasing_set1 = [], []
    mse_x_increasing_set2, mse_y_increasing_set2 = [], []
    mse_x_decreasing_set1, mse_y_decreasing_set1 = [], []
    mse_x_decreasing_set2, mse_y_decreasing_set2 = [], []
    rnn_lin_set1, rnn_lin_set2 = [], []
    whole_mse_constant_set1, whole_mse_constant_set2 = [], []
    whole_mse_same_speed_set1 = []
    whole_mse_same_speed_set2 = []
    total_rnn_responses_same_speed_set1 = []
    total_rnn_responses_same_speed_set2 = []

    batch_size = 32

    total_combinations = len(x_combinations) * len(y_combinations)
    progress_bar = tqdm(total=total_combinations, desc="Processing combinations")

    try:
        for i in range(0, len(x_combinations), batch_size):
            for j in range(0, len(y_combinations), batch_size):
                x_batch = x_combinations[i:i + batch_size]
                y_batch = y_combinations[j:j + batch_size]

                batch_inputs = []
                total_change_magnitudes = []
                true_speeds_x = []
                true_speeds_y = []

                for x_speeds, y_speeds in zip(x_batch, y_batch):
                    initial_speed = (x_speeds[0], y_speeds[0])
                    change_speeds = list(zip(x_speeds[1:], y_speeds[1:]))

                    input_sequence, _ = generate_moving_dot_sequence(
                        initial_speed[0], initial_speed[1], dot_radius, num_frames, im_height, im_width,
                        change_mode=True, num_changes=len(change_speeds),
                        timesteps_between_changes=timesteps_between_changes,
                        change_speeds=change_speeds
                    )
                    batch_inputs.append(input_sequence)

                    total_change_magnitude = sum(
                        abs(x_speeds[k + 1] - x_speeds[k]) + abs(y_speeds[k + 1] - y_speeds[k]) for k in
                        range(num_changes))
                    total_change_magnitudes.append(total_change_magnitude)

                    true_x = np.array(
                        [x_speeds[0]] + [speed for speed in x_speeds[1:] for _ in range(timesteps_between_changes)])[
                             :num_frames]
                    true_y = np.array(
                        [y_speeds[0]] + [speed for speed in y_speeds[1:] for _ in range(timesteps_between_changes)])[
                             :num_frames]
                    true_speeds_x.append(true_x)
                    true_speeds_y.append(true_y)

                batch_inputs = torch.cat(batch_inputs, dim=0).to(device)

                def process_batch(models):
                    all_predictions = []
                    all_mse_x = []
                    all_mse_y = []
                    all_rnn_outs = []
                    all_whole_mse_x = []
                    all_whole_mse_y = []
                    all_total_rnn_responses = []
                    for model in models:
                        with torch.no_grad():
                            predictions, _, _, rnn_outs = model(batch_inputs)
                        all_predictions.append(predictions.cpu())
                        all_rnn_outs.append(rnn_outs.cpu())

                        mse_x = [calculate_mse(predictions[k, :, 0], torch.tensor(true_speeds_x[k], device=device),
                                               range(0, num_frames, timesteps_between_changes)[1:], window_size)
                                 for k in range(len(true_speeds_x))]
                        mse_y = [calculate_mse(predictions[k, :, 1], torch.tensor(true_speeds_y[k], device=device),
                                               range(0, num_frames, timesteps_between_changes)[1:], window_size)
                                 for k in range(len(true_speeds_y))]
                        all_mse_x.append(mse_x)
                        all_mse_y.append(mse_y)

                        whole_mse_x = []
                        whole_mse_y = []
                        total_rnn_responses = []
                        for k in range(len(true_speeds_x)):
                            pred_len = predictions.shape[1]
                            true_len = len(true_speeds_x[k])
                            min_len = min(pred_len, true_len)

                            mse_x = torch.mean((predictions[k, :min_len, 0] - torch.tensor(true_speeds_x[k][:min_len],
                                                                                           device=device)) ** 2).item()
                            mse_y = torch.mean((predictions[k, :min_len, 1] - torch.tensor(true_speeds_y[k][:min_len],
                                                                                           device=device)) ** 2).item()

                            whole_mse_x.append(mse_x)
                            whole_mse_y.append(mse_y)

                            total_rnn_response = torch.sum(torch.abs(rnn_outs[k, :min_len])).item()
                            total_rnn_responses.append(total_rnn_response)

                        all_whole_mse_x.append(whole_mse_x)
                        all_whole_mse_y.append(whole_mse_y)
                        all_total_rnn_responses.append(total_rnn_responses)

                    all_predictions = torch.stack(all_predictions)
                    all_rnn_outs = torch.sum(torch.sum(torch.stack(all_rnn_outs), dim=3), dim=2)
                    total_prediction_magnitudes_x = torch.sum(torch.abs(all_predictions[:, :, :, 0]), dim=2).numpy()
                    total_prediction_magnitudes_y = torch.sum(torch.abs(all_predictions[:, :, :, 1]), dim=2).numpy()

                    return total_prediction_magnitudes_x, total_prediction_magnitudes_y, np.array(all_mse_x), np.array(
                        all_mse_y), all_rnn_outs, np.array(all_whole_mse_x), np.array(all_whole_mse_y), np.array(
                        all_total_rnn_responses)

                tpm_x_set1, tpm_y_set1, mse_x_set1_batch, mse_y_set1_batch, rnn_outs_set1, whole_mse_x_set1_batch, whole_mse_y_set1_batch, total_rnn_responses_set1_batch = process_batch(
                    models_set1)
                tpm_x_set2, tpm_y_set2, mse_x_set2_batch, mse_y_set2_batch, rnn_outs_set2, whole_mse_x_set2_batch, whole_mse_y_set2_batch, total_rnn_responses_set2_batch = process_batch(
                    models_set2)

                for idx, (x_speeds, y_speeds) in enumerate(zip(x_batch, y_batch)):
                    tcm = total_change_magnitudes[idx]

                    if tcm in results_x_set1:
                        results_x_set1[tcm].append(tpm_x_set1[:, idx])
                        results_y_set1[tcm].append(tpm_y_set1[:, idx])
                        results_x_set2[tcm].append(tpm_x_set2[:, idx])
                        results_y_set2[tcm].append(tpm_y_set2[:, idx])
                        mse_x_set1[tcm].append(mse_x_set1_batch[:, idx])
                        mse_y_set1[tcm].append(mse_y_set1_batch[:, idx])
                        mse_x_set2[tcm].append(mse_x_set2_batch[:, idx])
                        mse_y_set2[tcm].append(mse_y_set2_batch[:, idx])
                    else:
                        results_x_set1[tcm] = [tpm_x_set1[:, idx]]
                        results_y_set1[tcm] = [tpm_y_set1[:, idx]]
                        results_x_set2[tcm] = [tpm_x_set2[:, idx]]
                        results_y_set2[tcm] = [tpm_y_set2[:, idx]]
                        mse_x_set1[tcm] = [mse_x_set1_batch[:, idx]]
                        mse_y_set1[tcm] = [mse_y_set1_batch[:, idx]]
                        mse_x_set2[tcm] = [mse_x_set2_batch[:, idx]]
                        mse_y_set2[tcm] = [mse_y_set2_batch[:, idx]]

                    if all(x_speeds[i] == y_speeds[i] for i in range(len(x_speeds))) and len(set(x_speeds)) == 1:
                        whole_mse_same_speed_set1.append(
                            whole_mse_x_set1_batch[:, idx] + whole_mse_y_set1_batch[:, idx])
                        whole_mse_same_speed_set2.append(
                            whole_mse_x_set2_batch[:, idx] + whole_mse_y_set2_batch[:, idx])
                        total_rnn_responses_same_speed_set1.append(total_rnn_responses_set1_batch[:, idx])
                        total_rnn_responses_same_speed_set2.append(total_rnn_responses_set2_batch[:, idx])

                    if all(x_speeds[i] < x_speeds[i + 1] for i in range(len(x_speeds) - 1)):
                        mse_x_increasing_set1.append(mse_x_set1_batch[:, idx])
                        mse_x_increasing_set2.append(mse_x_set2_batch[:, idx])
                        rnn_lin_set1.append(rnn_outs_set1[:, idx])
                        rnn_lin_set2.append(rnn_outs_set2[:, idx])

                    if all(x_speeds[i] > x_speeds[i + 1] for i in range(len(x_speeds) - 1)):
                        mse_x_decreasing_set1.append(mse_x_set1_batch[:, idx])
                        mse_x_decreasing_set2.append(mse_x_set2_batch[:, idx])
                        rnn_lin_set1.append(rnn_outs_set1[:, idx])
                        rnn_lin_set2.append(rnn_outs_set2[:, idx])

                    if all(y_speeds[i] < y_speeds[i + 1] for i in range(len(y_speeds) - 1)):
                        mse_y_increasing_set1.append(mse_y_set1_batch[:, idx])
                        mse_y_increasing_set2.append(mse_y_set2_batch[:, idx])
                        rnn_lin_set1.append(rnn_outs_set1[:, idx])
                        rnn_lin_set2.append(rnn_outs_set2[:, idx])

                    if all(y_speeds[i] > y_speeds[i + 1] for i in range(len(y_speeds) - 1)):
                        mse_y_decreasing_set1.append(mse_y_set1_batch[:, idx])
                        mse_y_decreasing_set2.append(mse_y_set2_batch[:, idx])
                        rnn_lin_set1.append(rnn_outs_set1[:, idx])
                        rnn_lin_set2.append(rnn_outs_set2[:, idx])

                progress_bar.update(len(x_batch) * len(y_batch))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        progress_bar.close()

    print("Comprehensive analysis completed.")

    mse_x_linear_set1 = mse_x_increasing_set1 + mse_x_decreasing_set1
    mse_y_linear_set1 = mse_y_increasing_set1 + mse_y_decreasing_set1
    mse_x_linear_set2 = mse_x_increasing_set2 + mse_x_decreasing_set2
    mse_y_linear_set2 = mse_y_increasing_set2 + mse_y_decreasing_set2

    avg_mse_linear_set1 = np.mean(mse_x_linear_set1 + mse_y_linear_set1)
    avg_mse_linear_set2 = np.mean(mse_x_linear_set2 + mse_y_linear_set2)

    print("\nAverage combined MSE (X and Y) for cases with With Change:")
    print(f"Motionnet-R: {avg_mse_linear_set1:.4f}")
    print(f"AdaptNet: {avg_mse_linear_set2:.4f}")

    mse_difference = avg_mse_linear_set1 - avg_mse_linear_set2
    print(f"\nDifference in MSE (Motionnet-R - AdaptNet): {mse_difference:.4f}")

    if avg_mse_linear_set1 != 0:
        percentage_improvement = (mse_difference / avg_mse_linear_set1) * 100
        print(f"Percentage improvement of AdaptNet over Motionnet-R: {percentage_improvement:.2f}%")
    else:
        print("Cannot calculate percentage improvement as Motionnet-R MSE is zero.")

    avg_rnn_lin_set1 = torch.mean(torch.stack(rnn_lin_set1))
    avg_rnn_lin_set2 = torch.mean(torch.stack(rnn_lin_set2))

    print("\nAverage RNN responses for With Change:")
    print(f"Motionnet-R: {avg_rnn_lin_set1:.4f}")
    print(f"AdaptNet: {avg_rnn_lin_set2:.4f}")

    rnn_difference = avg_rnn_lin_set1 - avg_rnn_lin_set2
    print(f"\nDifference in RNN responses (Motionnet-R - AdaptNet): {rnn_difference:.4f}")

    if avg_rnn_lin_set1 != 0:
        percentage_difference = (rnn_difference / avg_rnn_lin_set1) * 100
        print(f"Percentage difference of AdaptNet compared to Motionnet-R: {percentage_difference:.2f}%")
    else:
        print("Cannot calculate percentage difference as Motionnet-R response is zero.")

    if whole_mse_same_speed_set1 and whole_mse_same_speed_set2:
        avg_whole_mse_same_speed_set1 = np.mean(whole_mse_same_speed_set1)
        avg_whole_mse_same_speed_set2 = np.mean(whole_mse_same_speed_set2)
        avg_total_rnn_responses_same_speed_set1 = np.mean(total_rnn_responses_same_speed_set1)
        avg_total_rnn_responses_same_speed_set2 = np.mean(total_rnn_responses_same_speed_set2)

        print("\nAverage whole sequence MSE for cases with Constant speed in all directions:")
        print(f"Motionnet-R: {avg_whole_mse_same_speed_set1:.4f}")
        print(f"AdaptNet: {avg_whole_mse_same_speed_set2:.4f}")

        print("\nAverage total RNN responses for cases with Constant speed in all directions:")
        print(f"Motionnet-R: {avg_total_rnn_responses_same_speed_set1:.4f}")
        print(f"AdaptNet: {avg_total_rnn_responses_same_speed_set2:.4f}")

        whole_mse_difference = avg_whole_mse_same_speed_set1 - avg_whole_mse_same_speed_set2
        total_rnn_responses_difference = avg_total_rnn_responses_same_speed_set1 - avg_total_rnn_responses_same_speed_set2

        print(f"\nDifference in whole sequence MSE (Motionnet-R - AdaptNet): {whole_mse_difference:.4f}")
        print(f"Difference in total RNN responses (Motionnet-R - AdaptNet): {total_rnn_responses_difference:.4f}")

        if avg_whole_mse_same_speed_set1 != 0:
            mse_percentage_difference = (whole_mse_difference / avg_whole_mse_same_speed_set1) * 100
            print(f"Percentage difference of AdaptNet compared to Motionnet-R (MSE): {mse_percentage_difference:.2f}%")
        else:
            print("Cannot calculate percentage difference for MSE as Motionnet-R whole sequence MSE is zero.")

        if avg_total_rnn_responses_same_speed_set1 != 0:
            rnn_percentage_difference = (total_rnn_responses_difference / avg_total_rnn_responses_same_speed_set1) * 100
            print(
                f"Percentage difference of AdaptNet compared to Motionnet-R (RNN responses): {rnn_percentage_difference:.2f}%")
        else:
            print("Cannot calculate percentage difference for RNN responses as Motionnet-R total response is zero.")
    else:
        print("\nNo cases found with Constant speed in all directions.")

    linear_data = (
        rnn_lin_set1, rnn_lin_set2, (mse_x_linear_set1 + mse_y_linear_set1), (mse_x_linear_set2 + mse_y_linear_set2))
    whole_data = (total_rnn_responses_same_speed_set1, total_rnn_responses_same_speed_set2, whole_mse_same_speed_set1,
                  whole_mse_same_speed_set2)

    plot_comparison_graphs(linear_data, whole_data)

    def calc_avg_se(data):
        avg = {k: np.mean(np.concatenate(v)) for k, v in data.items()}
        se = {k: np.std(np.concatenate(v)) / np.sqrt(len(v) * 10) for k, v in data.items()}
        return avg, se

    avg_x_set1, se_x_set1 = calc_avg_se(results_x_set1)
    avg_y_set1, se_y_set1 = calc_avg_se(results_y_set1)
    avg_x_set2, se_x_set2 = calc_avg_se(results_x_set2)
    avg_y_set2, se_y_set2 = calc_avg_se(results_y_set2)

    avg_mse_x_set1, se_mse_x_set1 = calc_avg_se(mse_x_set1)
    avg_mse_y_set1, se_mse_y_set1 = calc_avg_se(mse_y_set1)
    avg_mse_x_set2, se_mse_x_set2 = calc_avg_se(mse_x_set2)
    avg_mse_y_set2, se_mse_y_set2 = calc_avg_se(mse_y_set2)

    # Perform t-tests
    print("\nPerforming t-tests:")

    # Change case
    mse_change_set1 = np.concatenate(mse_x_linear_set1 + mse_y_linear_set1)
    mse_change_set2 = np.concatenate(mse_x_linear_set2 + mse_y_linear_set2)
    perform_t_test(mse_change_set1, mse_change_set2, "MSE (Change case)")

    power_change_set1 = np.concatenate(rnn_lin_set1)
    power_change_set2 = np.concatenate(rnn_lin_set2)
    perform_t_test(power_change_set1, power_change_set2, "Power (Change case)")

    # Constant case
    mse_constant_set1 = np.concatenate(whole_mse_same_speed_set1)
    mse_constant_set2 = np.concatenate(whole_mse_same_speed_set2)
    perform_t_test(mse_constant_set1, mse_constant_set2, "MSE (Constant case)")

    power_constant_set1 = np.concatenate(total_rnn_responses_same_speed_set1)
    power_constant_set2 = np.concatenate(total_rnn_responses_same_speed_set2)
    perform_t_test(power_constant_set1, power_constant_set2, "Power (Constant case)")

    font_path = '../../data/fonts/Roboto-Regular.ttf'  # Update this path
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

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Change Magnitude vs Prediction Magnitude and MSE (Window Size: {window_size})', fontsize=16)

    change_magnitudes_x = sorted(set(avg_x_set1.keys()) | set(avg_x_set2.keys()))
    prediction_magnitudes_x_set1 = [avg_x_set1.get(k, 0) for k in change_magnitudes_x]
    prediction_magnitudes_x_set2 = [avg_x_set2.get(k, 0) for k in change_magnitudes_x]
    error_x_set1 = [se_x_set1.get(k, 0) for k in change_magnitudes_x]
    error_x_set2 = [se_x_set2.get(k, 0) for k in change_magnitudes_x]

    ax1.errorbar(change_magnitudes_x, prediction_magnitudes_x_set1, yerr=error_x_set1, fmt='o-', color='blue',
                 label='Motionnet-R', capsize=5)
    ax1.errorbar(change_magnitudes_x, prediction_magnitudes_x_set2, yerr=error_x_set2, fmt='s-', color='red',
                 label='AdaptNet', capsize=5)
    ax1.set_xlabel('Total Change Magnitude')
    ax1.set_ylabel('Average Total Prediction Magnitude')
    ax1.set_title('X Prediction')
    ax1.grid(True)
    ax1.legend()
    remove_borders(ax1)

    change_magnitudes_y = sorted(set(avg_y_set1.keys()) | set(avg_y_set2.keys()))
    prediction_magnitudes_y_set1 = [avg_y_set1.get(k, 0) for k in change_magnitudes_y]
    prediction_magnitudes_y_set2 = [avg_y_set2.get(k, 0) for k in change_magnitudes_y]
    error_y_set1 = [se_y_set1.get(k, 0) for k in change_magnitudes_y]
    error_y_set2 = [se_y_set2.get(k, 0) for k in change_magnitudes_y]

    ax2.errorbar(change_magnitudes_y, prediction_magnitudes_y_set1, yerr=error_y_set1, fmt='o-', color='green',
                 label='Motionnet-R', capsize=5)
    ax2.errorbar(change_magnitudes_y, prediction_magnitudes_y_set2, yerr=error_y_set2, fmt='s-', color='purple',
                 label='AdaptNet', capsize=5)
    ax2.set_xlabel('Total Change Magnitude')
    ax2.set_ylabel('Average Total Prediction Magnitude')
    ax2.set_title('Y Prediction')
    ax2.grid(True)
    ax2.legend()
    remove_borders(ax2)

    mse_x_set1 = [avg_mse_x_set1.get(k, 0) for k in change_magnitudes_x]
    mse_x_set2 = [avg_mse_x_set2.get(k, 0) for k in change_magnitudes_x]
    error_mse_x_set1 = [se_mse_x_set1.get(k, 0) for k in change_magnitudes_x]
    error_mse_x_set2 = [se_mse_x_set2.get(k, 0) for k in change_magnitudes_x]

    ax3.errorbar(change_magnitudes_x, mse_x_set1, yerr=error_mse_x_set1, fmt='o-', color='blue', label='Motionnet-R',
                 capsize=5)
    ax3.errorbar(change_magnitudes_x, mse_x_set2, yerr=error_mse_x_set2, fmt='s-', color='red', label='AdaptNet',
                 capsize=5)
    ax3.set_xlabel('Total Change Magnitude')
    ax3.set_ylabel('Average MSE')
    ax3.set_title('X MSE')
    ax3.grid(True)
    ax3.legend()
    remove_borders(ax3)

    mse_y_set1 = [avg_mse_y_set1.get(k, 0) for k in change_magnitudes_y]
    mse_y_set2 = [avg_mse_y_set2.get(k, 0) for k in change_magnitudes_y]
    error_mse_y_set1 = [se_mse_y_set1.get(k, 0) for k in change_magnitudes_y]
    error_mse_y_set2 = [se_mse_y_set2.get(k, 0) for k in change_magnitudes_y]

    ax4.errorbar(change_magnitudes_y, mse_y_set1, yerr=error_mse_y_set1, fmt='o-', color='green', label='Motionnet-R',
                 capsize=5)
    ax4.errorbar(change_magnitudes_y, mse_y_set2, yerr=error_mse_y_set2, fmt='s-', color='purple', label='AdaptNet',
                 capsize=5)
    ax4.set_xlabel('Total Change Magnitude')
    ax4.set_ylabel('Average MSE')
    ax4.set_title('Y MSE')
    ax4.grid(True)
    ax4.legend()
    remove_borders(ax4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
