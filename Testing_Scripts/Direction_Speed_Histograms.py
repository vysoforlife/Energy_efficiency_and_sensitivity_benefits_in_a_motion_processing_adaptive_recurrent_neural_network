import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from math import pi, cos, sin
import math
from torch import nn
from tqdm import tqdm
from scipy.stats import ttest_rel, binomtest, chisquare

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_moving_dot_sequences(num_directions, num_speeds, dot_radius=3, num_frames=10, im_height=32, im_width=32,
                                  fixed_speed=None, fixed_angle=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fixed_speed is not None and fixed_angle is not None:
        speeds = torch.tensor([fixed_speed], device=device)
        angles = torch.tensor([fixed_angle], device=device)
        batch_size = 1
    else:
        speeds = torch.linspace(0, 5, num_speeds, device=device)
        angles = torch.linspace(0, 2 * pi, num_directions + 1, device=device)[:-1]
        speeds, angles = torch.meshgrid(speeds, angles, indexing='ij')
        speeds = speeds.flatten()
        angles = angles.flatten()
        batch_size = num_speeds * num_directions

    if fixed_speed is not None and fixed_speed == 0:
        x_speeds = torch.zeros(1, device=device)
        y_speeds = torch.zeros(1, device=device)
    else:
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
        _, _, _, rnn_output = model(sequences)
    return rnn_output.mean(dim=1).cpu().numpy()

def analyze_conv_responses(model, sequences):
    model.eval()
    with torch.no_grad():
        _, conv_output, _, _ = model(sequences)
    return conv_output.mean(dim=(1, 3, 4)).cpu().numpy()

def calculate_preferred_direction_speed(responses, labels):
    num_units = responses.shape[1]
    preferred_directions = []
    preferred_speeds = []

    for unit in range(num_units):
        unit_responses = responses[:, unit]
        max_response_idx = np.argmax(unit_responses)
        preferred_directions.append(labels[max_response_idx, 2].item())
        preferred_speeds.append(labels[max_response_idx, 3].item())

    return preferred_directions, preferred_speeds

from matplotlib import font_manager

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import pi, cos, sin
import math
from torch import nn
from tqdm import tqdm
from matplotlib import font_manager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Your existing functions remain the same
# (generate_moving_dot_sequences, analyze_rnn_responses, analyze_conv_responses,
#  calculate_preferred_direction_speed, circular_mean, categorize_neurons,
#  generate_single_sequence, process_sequence, SimpleRNN_n, Model_n)

# Update the plot_histograms function to plot_combined_histograms


def plot_combined_histograms(conv_directions, conv_speeds, rnn_directions, rnn_speeds):
    font_path = '../misc/fonts/Roboto-Regular.ttf'  # Update this path
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'

    plt.rcParams.update({
        'font.size': 35,
        'axes.titlesize': 35,
        'axes.labelsize': 35,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35
    })

    fig, axs = plt.subplots(2, 2, figsize=(20, 18))  # Increased figure height

    handles, labels = plot_histogram(axs[0, 0], conv_directions, 'Conv Kernels Preferred Directions', 'Direction')
    plot_histogram(axs[0, 1], conv_speeds, 'Conv Kernels Preferred Speeds', 'Speed (deg/s)')
    plot_histogram(axs[1, 0], rnn_directions, 'RNN Units Preferred Directions', 'Direction')
    plot_histogram(axs[1, 1], rnn_speeds, 'RNN Units Preferred Speeds', 'Speed (deg/s)')

    # Create a single legend for the entire figure
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02),
               fontsize=30, frameon=False)

    plt.tight_layout()
    # Adjust the bottom margin to make room for the legend
    plt.subplots_adjust(bottom=0.15)

    # Save the figure with a larger bounding box to include the legend
    plt.savefig('../Saved_Images/Dir_Speed_Histograms.svg', format='svg', bbox_inches='tight', pad_inches=0.5)
    plt.show()


def plot_histogram(ax, data, title, xlabel):
    if xlabel == 'Direction':
        # Direction plotting remains the same
        bins = np.linspace(0, 360, 9)
        width = (bins[1] - bins[0]) * 0.45
        for i, model_data in enumerate(data):
            counts, _ = np.histogram(model_data, bins=bins)
            ax.bar(bins[:-1] + i * width, counts, width=width, align='edge',
                   color='blue' if i == 0 else 'orange', alpha=0.7,
                   label='motionnet-R' if i == 0 else 'AdaptNet')

        ax.set_xticks(np.linspace(22.5, 337.5, 8))
        ax.set_xticklabels([f"{int(x)}°" for x in np.linspace(0, 315, 8)])
        ax.set_xlim(0, 360)
    else:
        # Speed plotting with improved tick handling
        bins = np.logspace(np.log10(1), np.log10(7), 15)
        width_factor = 0.45

        for i, model_data in enumerate(data):
            valid_data = [x for x in model_data if 1 <= x <= 7]
            counts, edges = np.histogram(valid_data, bins=bins)
            counts = counts / len(model_data)

            bar_lefts = edges[:-1]
            bar_widths = (edges[1:] - edges[:-1]) * width_factor

            ax.bar(bar_lefts + i * bar_widths, counts,
                   width=bar_widths, align='edge',
                   color='blue' if i == 0 else 'orange', alpha=0.7,
                   label='motionnet-R' if i == 0 else 'AdaptNet')

        ax.set_xscale('log')  # Set logarithmic scale
        ax.set_xlim(1, 7)  # Set range from 1 to 7

        # Create custom tick locations and labels
        tick_locations = [1, 2, 3, 4, 5, 6, 7]
        tick_labels = ['1', '2', '3', '4', '5', '6', '7']

        # Apply the custom ticks and formatting
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)

        # Force matplotlib to use these exact tick locations and labels
        ax.xaxis.set_major_locator(plt.FixedLocator(tick_locations))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(tick_labels))

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel('Proportion of units')

    if xlabel == 'Speed (pixels/frame)':
        ax.set_ylim(0, 0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    return ax.get_legend_handles_labels()

def circular_mean(angles):
    angles_rad = np.deg2rad(angles)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean_rad) % 360

def categorize_neurons(all_preferred_directions, n_categories=8):
    category_edges = np.linspace(0, 360, n_categories + 1)
    categories = np.digitize(all_preferred_directions, category_edges) - 1
    return categories

def generate_single_sequence(x_speed, y_speed, dot_radius=3, num_frames=8, im_height=32, im_width=32):
    speed = np.sqrt(x_speed**2 + y_speed**2)
    angle = np.arctan2(y_speed, x_speed) if speed > 0 else 0
    return generate_moving_dot_sequences(1, 1, dot_radius, num_frames, im_height, im_width,
                                         fixed_speed=speed, fixed_angle=angle)

def process_sequence(models, sequence, categories, n_categories):
    all_responses = []
    for model in models:
        model.eval()
        with torch.no_grad():
            _, _, _, rnn_output = model(sequence.to(device))
        all_responses.append(rnn_output.cpu().numpy())

    all_responses = np.concatenate(all_responses, axis=2)

    category_responses = []
    for frame in range(all_responses.shape[1]):
        frame_responses = all_responses[0, frame]
        cat_resp = [frame_responses[categories == i].mean() for i in range(n_categories)]
        category_responses.append(cat_resp)

    return np.array(category_responses)

def plot_category_responses(category_responses, n_categories):
    num_frames = category_responses.shape[0]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.ravel()

    x = range(n_categories)
    direction_ranges = [f"({i * 360 // n_categories}°-{(i + 1) * 360 // n_categories}°)" for i in range(n_categories)]

    for frame in range(num_frames):
        ax = axs[frame]
        ax.plot(x, category_responses[frame], marker='o')
        ax.set_title(f'Frame {frame + 1}')
        ax.set_xlabel('Direction Category')
        ax.set_ylabel('Average Response')
        ax.set_xticks(x)
        ax.set_xticklabels(direction_ranges, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)

    for i in range(num_frames, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

# Parameters
im_size = 32
num_of_frames = 9  # Previously 9
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27
lin_units = 16

class mnet(nn.Module):
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
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_fc1 = torch.stack([self.fc1(out_flat[:, i]) for i in range(num_frames)], dim=1)
        out_fc2 = torch.stack([self.fc2(out_fc1[:, i]) for i in range(num_frames)], dim=1)
        return out_fc2, out_conv, out_flat, out_fc1

class SimpleRNN_n(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True, batch_first=False):
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

        self.conv = nn.Conv2d(2, 16, kernel_size=(6, 6), stride=(1, 1))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(input_size=27 * 27 * 16, hidden_size=16, nonlinearity='relu',
                               batch_first=True).to(device)
        self.fc = nn.Linear(16, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn

# Model definition
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

        return out_fc, out_conv, out_flat, out_rnn

def main():
    num_directions = 24
    num_speeds = 20
    dot_radius = 3
    num_frames = 8
    im_height = 32
    im_width = 32
    num_models = 10

    batch_sequences, labels, speeds = generate_moving_dot_sequences(
        num_directions, num_speeds, dot_radius, num_frames, im_height, im_width
    )

    # Initialize arrays for both directions and speeds
    all_preferred_directions_rnn = [[], []]
    all_preferred_directions_conv = [[], []]
    all_preferred_speeds_rnn = [[], []]  # Add this line
    all_preferred_speeds_conv = [[], []]  # Add this line

    model_paths = [
        '../Trained_Models/motionnet_full_30epochs_{}.pt',
        '../Trained_Models/adaptnet_full_30epochs_{}_04.pt'
    ]

    for model_type in range(2):
        for i in range(1, num_models + 1):
            model_path = model_paths[model_type].format(i)
            try:
                model = torch.load(model_path, map_location=device)
                model.eval()
                model.to(device)
                print(f"Analyzing model type {model_type + 1}, model {i}")

                # Analyze RNN units
                rnn_responses = analyze_rnn_responses(model, batch_sequences.to(device))
                preferred_directions_rnn, preferred_speeds_rnn = calculate_preferred_direction_speed(rnn_responses, labels)
                all_preferred_directions_rnn[model_type].extend(preferred_directions_rnn)
                all_preferred_speeds_rnn[model_type].extend(preferred_speeds_rnn)  # Add this line

                # Analyze Conv units
                conv_responses = analyze_conv_responses(model, batch_sequences.to(device))
                preferred_directions_conv, preferred_speeds_conv = calculate_preferred_direction_speed(conv_responses, labels)
                all_preferred_directions_conv[model_type].extend(preferred_directions_conv)
                all_preferred_speeds_conv[model_type].extend(preferred_speeds_conv)  # Add this line

            except FileNotFoundError:
                print(f"Model file not found: {model_path}")

    # Now we can plot with both directions and speeds
    plot_combined_histograms(
        all_preferred_directions_conv,
        all_preferred_speeds_conv,
        all_preferred_directions_rnn,
        all_preferred_speeds_rnn
    )

if __name__ == "__main__":
    main()