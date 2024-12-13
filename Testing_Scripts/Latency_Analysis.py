import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import pi
import math

from matplotlib import font_manager
from torch import nn
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_moving_dot_sequences(num_directions, num_speeds, dot_radius=3, num_frames=20, im_height=32, im_width=32,
                                  fixed_speed=None, fixed_angle=None):
    """
    Generates a batch of moving dot sequences with specified parameters.
    """
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

def generate_single_sequence_with_reverse(x_speed, y_speed, reverse_frame, dot_radius=3, num_frames=20, im_height=32,
                                          im_width=32):
    """
    Generates a single sequence of moving dots with a reversal at the specified frame.
    """
    x_speeds = [x_speed] * num_frames
    y_speeds = [y_speed] * num_frames

    # Reverse speeds after reverse_frame
    for t in range(reverse_frame, num_frames):
        x_speeds[t] = -x_speed
        y_speeds[t] = -y_speed

    # Initialize positions
    x_positions = []
    y_positions = []

    x = float(im_width) / 2
    y = float(im_height) / 2

    for t in range(num_frames):
        x = (x + x_speeds[t]) % im_width
        y = (y + y_speeds[t]) % im_height
        x_positions.append(x)
        y_positions.append(y)

    # Generate the sequence
    scale_factor = 4
    hr_size = int(dot_radius * 2 * scale_factor)
    y_coords, x_coords = torch.meshgrid(torch.arange(hr_size, device=device), torch.arange(hr_size, device=device),
                                        indexing='ij')
    y_coords, x_coords = y_coords.float(), x_coords.float()
    center = hr_size / 2 - 0.5
    dist = torch.sqrt(((x_coords - center) / scale_factor) ** 2 + ((y_coords - center) / scale_factor) ** 2)
    hr_dot = torch.clamp(dot_radius - dist, 0, 1) * 255

    all_sequences = torch.zeros(1, num_frames, im_height, im_width, device=device)

    for t in range(num_frames):
        x_t = x_positions[t]
        y_t = y_positions[t]

        canvas = torch.zeros(1, im_height * scale_factor, im_width * scale_factor, device=device)

        x_start = (x_t - dot_radius) * scale_factor
        y_start = (y_t - dot_radius) * scale_factor

        for i in range(hr_size):
            for j in range(hr_size):
                canvas_y = (int(y_start) + i) % (im_height * scale_factor)
                canvas_x = (int(x_start) + j) % (im_width * scale_factor)
                canvas[0, canvas_y, canvas_x] = hr_dot[i, j]

        all_sequences[0, t] = F.avg_pool2d(canvas.unsqueeze(1), scale_factor, stride=scale_factor).squeeze(1)

    network_input = torch.zeros(1, num_frames - 1, 2, im_height, im_width, device=device)
    for i in range(num_frames - 1):
        network_input[0, i, 0] = all_sequences[0, i]
        network_input[0, i, 1] = all_sequences[0, i + 1]

    return network_input

def circular_mean(angles):
    """
    Calculates the circular mean of angles in degrees.
    """
    angles_rad = np.deg2rad(angles)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean_rad) % 360

def categorize_neurons(all_preferred_directions, n_categories=8):
    """
    Categorizes neurons based on their preferred directions.
    """
    category_edges = np.linspace(-22.5, 360 - 22.5, n_categories + 1)
    categories = np.digitize(all_preferred_directions, category_edges) - 1
    # Handle edge case where angle is exactly 360 - 22.5
    categories = np.clip(categories, 0, n_categories - 1)
    return categories

# Parameters
im_size = 32
num_of_frames = 20  # Updated to 20 frames
batch_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27

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

        self.conv = nn.Conv2d(2, n_kernels, kernel_size=(kernel_size, kernel_size), stride=(stride_size, stride_size))
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(input_size=act_size * act_size * n_kernels, hidden_size=rnn_units, nonlinearity='relu',
                               batch_first=True).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True, batch_first=False, adaptation_rate=0.2,
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
def analyze_rnn_responses(model, sequences):
    """
    Analyzes RNN responses by passing sequences through the model.
    """
    model.eval()
    with torch.no_grad():
        _, _, _, rnn_output = model(sequences)
    return rnn_output.mean(dim=1).cpu().numpy()

def calculate_preferred_direction_speed(rnn_responses, labels):
    """
    Calculates preferred directions and speeds for neurons based on RNN responses.
    """
    num_neurons = rnn_responses.shape[1]
    preferred_directions = []
    preferred_speeds = []

    for neuron in range(num_neurons):
        neuron_responses = rnn_responses[:, neuron]
        max_response_idx = np.argmax(neuron_responses)
        preferred_directions.append(labels[max_response_idx, 2].item())
        preferred_speeds.append(labels[max_response_idx, 3].item())

    return preferred_directions, preferred_speeds

def process_single_sequence(motionnet_models, adaptnet_models, sequence, categories, n_categories):
    """
    Processes a single sequence through both Motionnet-R and AdaptNet models.
    """
    motionnet_responses = []
    adaptnet_responses = []

    for model in motionnet_models:
        model.eval()
        with torch.no_grad():
            _, _, _, rnn_output = model(sequence.to(device))
        motionnet_responses.append(rnn_output.cpu().numpy())

    for model in adaptnet_models:
        model.eval()
        with torch.no_grad():
            _, _, _, rnn_output = model(sequence.to(device))
        adaptnet_responses.append(rnn_output.cpu().numpy())

    motionnet_responses = np.concatenate(motionnet_responses, axis=2)
    adaptnet_responses = np.concatenate(adaptnet_responses, axis=2)

    motionnet_category_responses = []
    motionnet_category_errors = []
    adaptnet_category_responses = []
    adaptnet_category_errors = []

    for frame in range(motionnet_responses.shape[1]):
        motionnet_frame_responses = motionnet_responses[0, frame]
        adaptnet_frame_responses = adaptnet_responses[0, frame]

        motionnet_cat_resp = [motionnet_frame_responses[categories == i].mean() for i in range(n_categories)]
        motionnet_cat_err = [motionnet_frame_responses[categories == i].std() / np.sqrt(np.sum(categories == i)) if np.sum(categories == i) > 0 else 0 for i in range(n_categories)]
        adaptnet_cat_resp = [adaptnet_frame_responses[categories == i].mean() for i in range(n_categories)]
        adaptnet_cat_err = [adaptnet_frame_responses[categories == i].std() / np.sqrt(np.sum(categories == i)) if np.sum(categories == i) > 0 else 0 for i in range(n_categories)]

        motionnet_category_responses.append(motionnet_cat_resp)
        motionnet_category_errors.append(motionnet_cat_err)
        adaptnet_category_responses.append(adaptnet_cat_resp)
        adaptnet_category_errors.append(adaptnet_cat_err)

    return (np.array(motionnet_category_responses), np.array(motionnet_category_errors),
            np.array(adaptnet_category_responses), np.array(adaptnet_category_errors))

def main():
    """
    Main function to execute the analysis.
    """
    # Sequence and Model Parameters
    num_directions = 24
    num_speeds = 20
    dot_radius = 3
    num_frames = 20  # Updated to 20 frames
    im_height = 32
    im_width = 32
    num_models = 10

    # Generate moving dot sequences
    batch_sequences, labels, speeds = generate_moving_dot_sequences(
        num_directions, num_speeds, dot_radius, num_frames, im_height, im_width
    )

    print(f"Generated sequences shape: {batch_sequences.shape}")  # (batch_size, num_frames-1, 2, 32, 32)
    print(f"Labels shape: {labels.shape}")  # (batch_size, 4)
    print(f"Speeds: {speeds}")

    # Initialize lists to store preferred directions and speeds
    all_preferred_directions = []
    all_preferred_speeds = []
    motionnet_models = []
    adaptnet_models = []

    # Load models
    for i in range(1, num_models + 1):
        motionnet_path = f'../Trained_Models/motionnet_full_30epochs_{i}.pt'
        adaptnet_path = f'../Trained_Models/adaptnet_full_30epochs_{i}_04.pt'

        try:
            # Load Motionnet-R model
            motionnet_model = torch.load(motionnet_path, map_location=device)
            motionnet_model.eval()
            motionnet_model.to(device)
            motionnet_models.append(motionnet_model)

            # Load AdaptNet model
            adaptnet_model = torch.load(adaptnet_path, map_location=device)
            adaptnet_model.eval()
            adaptnet_model.to(device)
            adaptnet_models.append(adaptnet_model)

            print(f"Analyzing model pair {i}")

            # Analyze RNN responses
            rnn_responses = analyze_rnn_responses(motionnet_model, batch_sequences.to(device))
            preferred_directions, preferred_speeds = calculate_preferred_direction_speed(rnn_responses, labels)

            all_preferred_directions.extend(preferred_directions)
            all_preferred_speeds.extend(preferred_speeds)

        except FileNotFoundError as e:
            print(f"Model file not found: {e}")

    # Statistics on preferred directions and speeds
    print("\nPreferred Direction Statistics:")
    print(f"Mean: {np.mean(all_preferred_directions):.2f} degrees")
    print(f"Circular Mean: {circular_mean(all_preferred_directions):.2f} degrees")
    print(f"Std Dev: {np.std(all_preferred_directions):.2f} degrees")

    print("\nPreferred Speed Statistics:")
    print(f"Mean: {np.mean(all_preferred_speeds):.2f}")
    print(f"Std Dev: {np.std(all_preferred_speeds):.2f}")

    # Categorize neurons based on preferred directions
    n_categories = 8
    categories = categorize_neurons(all_preferred_directions, n_categories)

    # Define direction ranges for labeling
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

    # Define the x_speed values and fixed parameters
    x_speed_values = [-3, -2, -1, 1, 2, 3]
    y_speed = 0

    # Lists to store averages and standard deviations for plotting
    reverse_frames_list = list(range(3, 16))  # From 5 to 10 inclusive
    motionnet_avg_frames_to_peak_list = []
    adaptnet_avg_frames_to_peak_list = []
    motionnet_std_frames_to_peak_list = []
    adaptnet_std_frames_to_peak_list = []

    for reverse_frame in reverse_frames_list:
        print(f"\nProcessing sequences with reverse_frame={reverse_frame}")
        # To store frames to reach within 10% of highest peak for summary
        motionnet_frames_to_peak = []
        adaptnet_frames_to_peak = []

        for x_speed in x_speed_values:
            input_label = f"({x_speed}, {y_speed})"
            print(f"\nProcessing sequence with x_speed={x_speed}, y_speed={y_speed}, reverse_frame={reverse_frame}")

            # Generate the sequence with reversal at reverse_frame
            sequence = generate_single_sequence_with_reverse(x_speed, y_speed, reverse_frame - 1, num_frames=num_frames)

            # Process the sequence through models
            motionnet_response, motionnet_error, adaptnet_response, adaptnet_error = process_single_sequence(
                motionnet_models, adaptnet_models, sequence, categories, n_categories)

            # Initialize lists to store peak values
            motionnet_peaks = []
            adaptnet_peaks = []

            # Store max (peak) of average responses for each frame
            for frame in range(motionnet_response.shape[0]):
                # Motionnet Peak
                motionnet_peak = np.max(motionnet_response[frame])
                # AdaptNet Peak
                adaptnet_peak = np.max(adaptnet_response[frame])

                # Store peaks
                motionnet_peaks.append(motionnet_peak)
                adaptnet_peaks.append(adaptnet_peak)

            # Identify the highest peak after the change point
            # Frames are 0-indexed; change occurs at reverse_frame-1 (frame number reverse_frame)
            post_change_peaks_motionnet = motionnet_peaks[reverse_frame -1:]
            post_change_peaks_adaptnet = adaptnet_peaks[reverse_frame -1:]

            highest_peak_motionnet = max(post_change_peaks_motionnet) if post_change_peaks_motionnet else None
            highest_peak_adaptnet = max(post_change_peaks_adaptnet) if post_change_peaks_adaptnet else None

            # Define 10% threshold
            threshold_motionnet = 0.9 * highest_peak_motionnet if highest_peak_motionnet else None
            threshold_adaptnet = 0.9 * highest_peak_adaptnet if highest_peak_adaptnet else None

            # Find first frame reaching within 10% of highest peak
            frames_to_threshold_motionnet = None
            if threshold_motionnet is not None:
                for idx, val in enumerate(post_change_peaks_motionnet):
                    if val >= threshold_motionnet:
                        frames_to_threshold_motionnet = idx + 1  # frames after reversal
                        break

            frames_to_threshold_adaptnet = None
            if threshold_adaptnet is not None:
                for idx, val in enumerate(post_change_peaks_adaptnet):
                    if val >= threshold_adaptnet:
                        frames_to_threshold_adaptnet = idx + 1
                        break

            # Append to lists for variance calculation
            if frames_to_threshold_motionnet is not None:
                motionnet_frames_to_peak.append(frames_to_threshold_motionnet)
            if frames_to_threshold_adaptnet is not None:
                adaptnet_frames_to_peak.append(frames_to_threshold_adaptnet)

        # After processing all x_speed_values for the current reverse_frame
        # Calculate average frames to reach within 10% of highest peak
        avg_motionnet_frames_to_peak = np.mean(motionnet_frames_to_peak) if motionnet_frames_to_peak else None
        avg_adaptnet_frames_to_peak = np.mean(adaptnet_frames_to_peak) if adaptnet_frames_to_peak else None

        std_motionnet_frames_to_peak = np.std(motionnet_frames_to_peak) if motionnet_frames_to_peak else None
        std_adaptnet_frames_to_peak = np.std(adaptnet_frames_to_peak) if adaptnet_frames_to_peak else None

        motionnet_avg_frames_to_peak_list.append(avg_motionnet_frames_to_peak)
        adaptnet_avg_frames_to_peak_list.append(avg_adaptnet_frames_to_peak)
        motionnet_std_frames_to_peak_list.append(std_motionnet_frames_to_peak)
        adaptnet_std_frames_to_peak_list.append(std_adaptnet_frames_to_peak)

        # Ensure the font is available
        font_path = '../misc/fonts/Roboto-Regular.ttf'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Roboto'
        plt.rcParams['font.size'] = 50
        plt.rcParams['xtick.labelsize'] = 50
        plt.rcParams['ytick.labelsize'] = 50
        plt.rcParams['axes.titlesize'] = 50
        plt.rcParams['axes.labelsize'] = 50
        plt.rcParams['legend.fontsize'] = 50

        print(f"\nAverage Number of Frames to Reach Within 10% of Highest Peak After Change Point for reverse_frame={reverse_frame}:")
        if avg_motionnet_frames_to_peak is not None:
            print(f"  Motionnet-R: {avg_motionnet_frames_to_peak:.2f} frames (±{std_motionnet_frames_to_peak:.2f})")
        else:
            print("  Motionnet-R: No peaks reached within 10% after the change point.")

        if avg_adaptnet_frames_to_peak is not None:
            print(f"  AdaptNet: {avg_adaptnet_frames_to_peak:.2f} frames (±{std_adaptnet_frames_to_peak:.2f})")
        else:
            print("  AdaptNet: No peaks reached within 10% after the change point.")

    # Plot the final graph with shaded error bars
    plt.figure(figsize=(30, 18))
    reverse_frames_array = np.array(reverse_frames_list)

    motionnet_avg_array = np.array(motionnet_avg_frames_to_peak_list)
    adaptnet_avg_array = np.array(adaptnet_avg_frames_to_peak_list)
    motionnet_std_array = np.array(motionnet_std_frames_to_peak_list)
    adaptnet_std_array = np.array(adaptnet_std_frames_to_peak_list)

    # Get current axis
    ax = plt.gca()

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make tick lines longer
    ax.tick_params(axis='both', which='major', length=10, width=2)

    # Plot Motionnet-R with shaded error
    plt.plot(reverse_frames_array, motionnet_avg_array, marker='o', label='Motionnet-R', color='blue')
    plt.fill_between(reverse_frames_array,
                     motionnet_avg_array - motionnet_std_array,
                     motionnet_avg_array + motionnet_std_array,
                     color='blue', alpha=0.2)

    # Plot AdaptNet with shaded error
    plt.plot(reverse_frames_array, adaptnet_avg_array, marker='o', label='AdaptNet', color='orange')
    plt.fill_between(reverse_frames_array,
                     adaptnet_avg_array - adaptnet_std_array,
                     adaptnet_avg_array + adaptnet_std_array,
                     color='orange', alpha=0.2)

    plt.xlabel('Reverse Frame')
    plt.ylabel('Average Frames to Reach Within 10% of Peak')
    plt.title('Average Frames to Reach Within 10% of Highest Peak After Change Point')
    plt.xticks(reverse_frames_list)
    plt.legend(fontsize=40, loc='upper right')
    plt.tight_layout()

    # Save the plot
    plt.savefig('../Saved_Images/Average_Frames_vs_Reverse_Frame_with_Error.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    main()
