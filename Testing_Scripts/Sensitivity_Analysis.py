import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import tqdm
import random
import math
from collections import defaultdict
from scipy import stats
from scipy.optimize import curve_fit

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Parameters
im_size = 32
n_kernels = 16
kernel_size = 6
stride_size = 1
rnn_units = 16
act_size = 27

# Configure custom font
font_path = '../misc/fonts/Roboto-Regular.ttf'  # Adjust the path to your font file
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Roboto'

# Update font sizes and axes properties
plt.rcParams.update({
    'font.size': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'axes.linewidth': 1
})

# Define the non-adapted SimpleRNN_n class
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

# Define the non-adapted Model_n class
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

# Define the adapted SimpleRNN class
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

# Define the AdaptiveLayer class
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

# Define the adapted Model class
class Model(nn.Module):
    def __init__(self, adaptation_rate_rnn=0.2):
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

# Function to generate moving dot sequence with speed change at the last frame
def generate_moving_dot_sequence_speed_change(x_speed_initial, x_speed_final, y_speed_initial, y_speed_final, dot_radius, num_frames,
                                              im_height, im_width):
    sequence = torch.zeros(num_frames + 1, im_height, im_width)

    x = random.randint(0, im_width - 1)
    y = random.randint(0, im_height - 1)

    for t in range(num_frames + 1):
        if t < num_frames:
            x_speed = x_speed_initial
            y_speed = y_speed_initial
        else:
            x_speed = x_speed_final
            y_speed = y_speed_final

        x = (x + x_speed) % im_width
        y = (y + y_speed) % im_height

        yy, xx = torch.meshgrid(torch.arange(im_height), torch.arange(im_width), indexing='ij')

        dx = torch.min(torch.abs(xx - x), im_width - torch.abs(xx - x))
        dy = torch.min(torch.abs(yy - y), im_height - torch.abs(yy - y))

        d = torch.sqrt(dx ** 2 + dy ** 2)

        dot = (d <= dot_radius).float() * 255

        sequence[t] = dot

    network_input = torch.zeros(num_frames, 2, im_height, im_width)
    for i in range(num_frames):
        network_input[i, 0] = sequence[i]
        network_input[i, 1] = sequence[i + 1]

    network_input = network_input.unsqueeze(0)

    return network_input

# Load the non-adapted models
models_non_adapted = []
for i in range(1, 11):
    model = Model_n().to(device)
    model.load_state_dict(
        torch.load(f'../Trained_Models/motionnet_full_30epochs_{i}.pt', map_location=device).state_dict())
    models_non_adapted.append(model)

# Load the adapted models
models_adapted = []
for i in range(1, 11):
    model = Model(adaptation_rate_rnn=0.4).to(device)
    model.load_state_dict(
        torch.load(f'../Trained_Models/adaptnet_full_30epochs_{i}_04.pt', map_location=device).state_dict())
    models_adapted.append(model)

# Analysis of hit probabilities
speeds = [-3, -2, -1, 0, 1, 2, 3]
num_sequences = 100  # Number of sequences per condition per direction
num_frames = 10  # Total number of frames in each sequence

speeds_array = np.array(speeds)
speed_to_index = {speed: idx for idx, speed in enumerate(speeds)}

# Initialize results storage
results_na = defaultdict(lambda: [])
results_a = defaultdict(lambda: [])

# Initialize per-model results storage for x-direction
results_na_per_model_x = [defaultdict(float) for _ in range(10)]  # List of dicts, one per model
results_a_per_model_x = [defaultdict(float) for _ in range(10)]  # Similarly for adapted models

# Main loop
for initial_speed in tqdm.tqdm(speeds, desc="Initial Speeds"):
    for changed_speed in speeds:
        key = (initial_speed, changed_speed)
        hits_na_list = []
        hits_a_list = []
        for model_idx in range(10):
            model_na = models_non_adapted[model_idx]
            model_a = models_adapted[model_idx]

            hits_na = 0
            hits_a = 0
            total_sequences = num_sequences * 2  # 100 x sequences + 100 y sequences

            hits_na_x = 0  # For x-direction sequences
            hits_a_x = 0

            # Generate and process x sequences
            for seq_idx in range(num_sequences):
                # Generate x sequence with speed change at the last frame
                input_sequence = generate_moving_dot_sequence_speed_change(
                    x_speed_initial=initial_speed,
                    x_speed_final=changed_speed,
                    y_speed_initial=0,
                    y_speed_final=0,
                    dot_radius=3,
                    num_frames=num_frames,
                    im_height=im_size,
                    im_width=im_size
                )
                input_sequence = input_sequence.to(device)
                # Non-adapted model
                model_na.eval()
                with torch.no_grad():
                    outputs_na, _, _, _ = model_na(input_sequence)
                    output_last_na_x = outputs_na[0, -1, 0].item()  # x-component at last frame
                    if output_last_na_x > 0:
                        hits_na += 1
                        hits_na_x += 1
                # Adapted model
                model_a.eval()
                with torch.no_grad():
                    outputs_a, _, _, _, _ = model_a(input_sequence)
                    output_last_a_x = outputs_a[0, -1, 0].item()  # x-component at last frame
                    if output_last_a_x > 0:
                        hits_a += 1
                        hits_a_x += 1

            # Generate and process y sequences
            for seq_idx in range(num_sequences):
                # Generate y sequence with speed change at the last frame
                input_sequence = generate_moving_dot_sequence_speed_change(
                    x_speed_initial=0,
                    x_speed_final=0,
                    y_speed_initial=initial_speed,
                    y_speed_final=changed_speed,
                    dot_radius=3,
                    num_frames=num_frames,
                    im_height=im_size,
                    im_width=im_size
                )
                input_sequence = input_sequence.to(device)
                # Non-adapted model
                model_na.eval()
                with torch.no_grad():
                    outputs_na, _, _, _ = model_na(input_sequence)
                    output_last_na_y = outputs_na[0, -1, 1].item()  # y-component at last frame
                    if output_last_na_y > 0:
                        hits_na += 1
                # Adapted model
                model_a.eval()
                with torch.no_grad():
                    outputs_a, _, _, _, _ = model_a(input_sequence)
                    output_last_a_y = outputs_a[0, -1, 1].item()  # y-component at last frame
                    if output_last_a_y > 0:
                        hits_a += 1

            # Compute hit rate for this model
            hit_rate_na = hits_na / (2 * num_sequences)
            hit_rate_a = hits_a / (2 * num_sequences)
            # Store the hit rates
            results_na[key].append(hit_rate_na)
            results_a[key].append(hit_rate_a)

            # Compute x-direction hit rates per model
            hit_rate_na_x = hits_na_x / num_sequences
            hit_rate_a_x = hits_a_x / num_sequences
            results_na_per_model_x[model_idx][key] = hit_rate_na_x
            results_a_per_model_x[model_idx][key] = hit_rate_a_x

# Compute average hit rates and organize into matrices
hit_rates_na_matrix = np.zeros((7, 7))
hit_rates_a_matrix = np.zeros((7, 7))
hit_rates_gt_matrix = np.zeros((7, 7))

for i, initial_speed in enumerate(speeds):
    for j, changed_speed in enumerate(speeds):
        key = (initial_speed, changed_speed)
        # Non-Adapted
        hit_rates_na = results_na[key]
        average_hit_rate_na = np.mean(hit_rates_na)
        hit_rates_na_matrix[i, j] = average_hit_rate_na
        # Adapted
        hit_rates_a = results_a[key]
        average_hit_rate_a = np.mean(hit_rates_a)
        hit_rates_a_matrix[i, j] = average_hit_rate_a
        # Ground Truth
        if changed_speed > 0:
            hit_rates_gt_matrix[i, j] = 1  # Expected hit
        elif changed_speed == 0:
            hit_rates_gt_matrix[i, j] = 0.5  # Neutral
        else:
            hit_rates_gt_matrix[i, j] = 0  # Expected miss

# Function to plot a single hit rate matrix
def plot_hit_rate_matrix(hit_rate_matrix, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(hit_rate_matrix), cmap='viridis', interpolation='nearest', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Hit Probability', rotation=270, labelpad=15)
    plt.xticks(ticks=range(len(speeds)), labels=speeds)
    plt.yticks(ticks=range(len(speeds)), labels=list(reversed(speeds)))
    plt.xlabel('Final speed')
    plt.ylabel('Initial speed')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.show()

# Plot Ground Truth Hit Probabilities
plot_hit_rate_matrix(hit_rates_gt_matrix, 'Ground Truth Hit Probabilities', '../Saved_Images/truth_matrix.svg')

# Plot Hit Probabilities (MotionNet-R)
plot_hit_rate_matrix(hit_rates_na_matrix, 'Hit Probabilities (MotionNet-R)', '../Saved_Images/Mnet_sens_matrix.svg')

# Plot Hit Probabilities (AdaptNet)
plot_hit_rate_matrix(hit_rates_a_matrix, 'Hit Probabilities (AdaptNet)', '../Saved_Images/Anet_sens_matrix.svg')

# Now compute the average hit probabilities across initial speeds for each final speed
mean_hit_rates_na = np.mean(hit_rates_na_matrix, axis=0)  # Shape: (7,)
mean_hit_rates_a = np.mean(hit_rates_a_matrix, axis=0)  # Shape: (7,)

mean_hit_rates_gt = np.mean(hit_rates_gt_matrix, axis=0)

#######################################
# Sigmoid Fits and Statistical Tests
#######################################

# Build hit rate matrices per model for x-direction sequences
num_speeds = len(speeds)
hit_rates_na_matrices_x = []  # List of matrices, one per model
hit_rates_a_matrices_x = []

for model_idx in range(10):
    # Initialize matrices for this model
    hit_rate_matrix_na_x = np.zeros((num_speeds, num_speeds))
    hit_rate_matrix_a_x = np.zeros((num_speeds, num_speeds))

    # For each initial_speed and changed_speed
    for i, initial_speed in enumerate(speeds):
        for j, changed_speed in enumerate(speeds):
            key = (initial_speed, changed_speed)
            # Non-Adapted
            hit_rate_na_x = results_na_per_model_x[model_idx][key]
            hit_rate_matrix_na_x[i, j] = hit_rate_na_x
            # Adapted
            hit_rate_a_x = results_a_per_model_x[model_idx][key]
            hit_rate_matrix_a_x[i, j] = hit_rate_a_x
    # Append the matrices
    hit_rates_na_matrices_x.append(hit_rate_matrix_na_x)
    hit_rates_a_matrices_x.append(hit_rate_matrix_a_x)

# Ground Truth Matrix (same for all models)
hit_rates_gt_matrix_x = np.zeros((num_speeds, num_speeds))
for i, initial_speed in enumerate(speeds):
    for j, changed_speed in enumerate(speeds):
        if changed_speed > 0:
            hit_rates_gt_matrix_x[i, j] = 1  # Expected hit
        elif changed_speed == 0:
            hit_rates_gt_matrix_x[i, j] = 0.5  # Neutral
        else:
            hit_rates_gt_matrix_x[i, j] = 0  # Expected miss

# Speeds array
speeds_array = np.array(speeds)

# Define the logistic function for curve fitting
def logistic_function(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

# Store slopes
slopes_mean_na = []
slopes_mean_a = []

slopes_diag_na = []
slopes_diag_a = []

# Store hit rates for plotting
mean_hit_rates_na_list = []
mean_hit_rates_a_list = []
diag_hit_rates_na_list = []
diag_hit_rates_a_list = []

# For averaging sigmoids
sigmoid_values_mean_na = []
sigmoid_values_mean_a = []
sigmoid_values_diag_na = []
sigmoid_values_diag_a = []

# Speeds for smooth curve plotting
x_smooth = np.linspace(speeds_array.min(), speeds_array.max(), 100)

for model_idx in range(10):
    # Non-adapted model
    # Mean over initial speeds
    mean_hit_rates_na = np.mean(hit_rates_na_matrices_x[model_idx], axis=0)  # Shape (7,)
    mean_hit_rates_na_list.append(mean_hit_rates_na)
    # Diagonal elements
    diag_hit_rates_na = np.diag(hit_rates_na_matrices_x[model_idx])  # Shape (7,)
    diag_hit_rates_na_list.append(diag_hit_rates_na)

    # Adapted model
    mean_hit_rates_a = np.mean(hit_rates_a_matrices_x[model_idx], axis=0)
    mean_hit_rates_a_list.append(mean_hit_rates_a)
    diag_hit_rates_a = np.diag(hit_rates_a_matrices_x[model_idx])
    diag_hit_rates_a_list.append(diag_hit_rates_a)

    # Fit logistic function to mean_hit_rates_na
    try:
        popt_na_mean, _ = curve_fit(logistic_function, speeds_array, mean_hit_rates_na, bounds=(0, [1., 5., 5.]))
        L_na_mean, x0_na_mean, k_na_mean = popt_na_mean
        slope_na_mean = (L_na_mean * k_na_mean) / 4  # Slope at midpoint
        slopes_mean_na.append(slope_na_mean)
        sigmoid_values_mean_na.append(logistic_function(x_smooth, *popt_na_mean))
    except RuntimeError:
        slopes_mean_na.append(np.nan)
        sigmoid_values_mean_na.append(np.full_like(x_smooth, np.nan))

    # Fit logistic function to diag_hit_rates_na
    try:
        popt_na_diag, _ = curve_fit(logistic_function, speeds_array, diag_hit_rates_na, bounds=(0, [1., 5., 5.]))
        L_na_diag, x0_na_diag, k_na_diag = popt_na_diag
        slope_na_diag = (L_na_diag * k_na_diag) / 4  # Slope at midpoint
        slopes_diag_na.append(slope_na_diag)
        sigmoid_values_diag_na.append(logistic_function(x_smooth, *popt_na_diag))
    except RuntimeError:
        slopes_diag_na.append(np.nan)
        sigmoid_values_diag_na.append(np.full_like(x_smooth, np.nan))

    # Adapted model
    # Fit logistic function to mean_hit_rates_a
    try:
        popt_a_mean, _ = curve_fit(logistic_function, speeds_array, mean_hit_rates_a, bounds=(0, [1., 5., 5.]))
        L_a_mean, x0_a_mean, k_a_mean = popt_a_mean
        slope_a_mean = (L_a_mean * k_a_mean) / 4  # Slope at midpoint
        slopes_mean_a.append(slope_a_mean)
        sigmoid_values_mean_a.append(logistic_function(x_smooth, *popt_a_mean))
    except RuntimeError:
        slopes_mean_a.append(np.nan)
        sigmoid_values_mean_a.append(np.full_like(x_smooth, np.nan))

    # Fit logistic function to diag_hit_rates_a
    try:
        popt_a_diag, _ = curve_fit(logistic_function, speeds_array, diag_hit_rates_a, bounds=(0, [1., 5., 5.]))
        L_a_diag, x0_a_diag, k_a_diag = popt_a_diag
        slope_a_diag = (L_a_diag * k_a_diag) / 4  # Slope at midpoint
        slopes_diag_a.append(slope_a_diag)
        sigmoid_values_diag_a.append(logistic_function(x_smooth, *popt_a_diag))
    except RuntimeError:
        slopes_diag_a.append(np.nan)
        sigmoid_values_diag_a.append(np.full_like(x_smooth, np.nan))

# Remove NaN values from slopes and sigmoid values (in case curve fitting failed for some models)
def remove_nan_entries(slopes, sigmoid_values):
    valid_indices = [i for i, s in enumerate(slopes) if not np.isnan(s)]
    slopes_clean = [slopes[i] for i in valid_indices]
    sigmoid_values_clean = [sigmoid_values[i] for i in valid_indices]
    return slopes_clean, sigmoid_values_clean

slopes_mean_na, sigmoid_values_mean_na = remove_nan_entries(slopes_mean_na, sigmoid_values_mean_na)
slopes_mean_a, sigmoid_values_mean_a = remove_nan_entries(slopes_mean_a, sigmoid_values_mean_a)
slopes_diag_na, sigmoid_values_diag_na = remove_nan_entries(slopes_diag_na, sigmoid_values_diag_na)
slopes_diag_a, sigmoid_values_diag_a = remove_nan_entries(slopes_diag_a, sigmoid_values_diag_a)

# Convert hit rates lists to arrays
mean_hit_rates_na_array = np.array(mean_hit_rates_na_list)  # Shape (num_models, num_speeds)
mean_hit_rates_a_array = np.array(mean_hit_rates_a_list)
diag_hit_rates_na_array = np.array(diag_hit_rates_na_list)
diag_hit_rates_a_array = np.array(diag_hit_rates_a_list)

# Compute mean and standard deviation across models
mean_hit_rates_na_mean = np.mean(mean_hit_rates_na_array, axis=0)
mean_hit_rates_na_std = np.std(mean_hit_rates_na_array, axis=0)

mean_hit_rates_a_mean = np.mean(mean_hit_rates_a_array, axis=0)
mean_hit_rates_a_std = np.std(mean_hit_rates_a_array, axis=0)

diag_hit_rates_na_mean = np.mean(diag_hit_rates_na_array, axis=0)
diag_hit_rates_na_std = np.std(diag_hit_rates_na_array, axis=0)

diag_hit_rates_a_mean = np.mean(diag_hit_rates_a_array, axis=0)
diag_hit_rates_a_std = np.std(diag_hit_rates_a_array, axis=0)

# Perform independent t-tests between the slopes
# Incongruent case (average over initial speeds)
t_stat_incongruent, p_value_incongruent = stats.ttest_ind(slopes_mean_na, slopes_mean_a, equal_var=False)

# Congruent case (diagonal elements)
t_stat_congruent, p_value_congruent = stats.ttest_ind(slopes_diag_na, slopes_diag_a, equal_var=False)

# Display the results with higher precision
print("Independent t-test between MotionNet-R and AdaptNet slopes (Incongruent Case):")
print(f"t-statistic = {t_stat_incongruent:.6f}, p-value = {p_value_incongruent:.10e}")

print("\nIndependent t-test between MotionNet-R and AdaptNet slopes (Congruent Case):")
print(f"t-statistic = {t_stat_congruent:.6f}, p-value = {p_value_congruent:.10e}")

# Font Configuration for Sigmoid plots
plt.rcParams.update({
    'font.size': 50,
    'axes.titlesize': 50,
    'axes.labelsize': 50,
    'xtick.labelsize': 50,
    'ytick.labelsize': 50,
    'axes.linewidth': 1,
})

# Plot averaged sigmoid curves for incongruent case
plt.figure(figsize=(20, 12))

# Plot average data points with error bars
plt.errorbar(speeds, mean_hit_rates_na_mean, yerr=mean_hit_rates_na_std, fmt='o', color='blue',
             label='MotionNet-R Data')
plt.errorbar(speeds, mean_hit_rates_a_mean, yerr=mean_hit_rates_a_std, fmt='o', color='orange', label='AdaptNet Data')

# Plot averaged sigmoid curves
plt.plot(x_smooth, np.nanmean(sigmoid_values_mean_na, axis=0), color='blue', linestyle='-',
         label='MotionNet-R Average Sigmoid')
plt.plot(x_smooth, np.nanmean(sigmoid_values_mean_a, axis=0), color='orange', linestyle='-',
         label='AdaptNet Average Sigmoid')

plt.xlabel('Final x-speed')
plt.ylabel('Average Hit Probability')
plt.title('Incongruent')
plt.legend()

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick length
plt.tick_params(axis='both', which='major', length=20, width=2)
plt.tick_params(axis='both', which='minor', length=20, width=2)

plt.tight_layout()
plt.savefig('../Saved_Images/Incongruent_Sigmoids.svg', format='svg')
plt.show()

# Plot averaged sigmoid curves for congruent case
plt.figure(figsize=(20, 12))

# Plot average data points with error bars
plt.errorbar(speeds, diag_hit_rates_na_mean, yerr=diag_hit_rates_na_std, fmt='o', color='blue',
             label='MotionNet-R Data')
plt.errorbar(speeds, diag_hit_rates_a_mean, yerr=diag_hit_rates_a_std, fmt='o', color='orange', label='AdaptNet Data')

# Plot averaged sigmoid curves
plt.plot(x_smooth, np.nanmean(sigmoid_values_diag_na, axis=0), color='blue', linestyle='-',
         label='MotionNet-R Average Sigmoid')
plt.plot(x_smooth, np.nanmean(sigmoid_values_diag_a, axis=0), color='orange', linestyle='-',
         label='AdaptNet Average Sigmoid')

plt.xlabel('Final x-speed')
plt.ylabel('Hit Probability')
plt.title('Congruent')
plt.legend()

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick length
plt.tick_params(axis='both', which='major', length=20, width=2)
plt.tick_params(axis='both', which='minor', length=20, width=2)

plt.tight_layout()
plt.savefig('../Saved_Images/Congruent_Sigmoids.svg', format='svg')
plt.show()
