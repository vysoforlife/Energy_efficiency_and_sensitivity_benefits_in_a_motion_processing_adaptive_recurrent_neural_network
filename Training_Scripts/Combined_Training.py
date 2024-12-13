import gzip
import pickle
import statistics
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from scipy.stats import pearsonr

# Device configuration
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
noise_level = 0.5  # Only used in the second network


# Load data
def load_data(filepath):
    with gzip.GzipFile(filepath, 'rb') as infile:
        obj = infile.read()
    temp_storage = pickle.loads(obj)
    images = temp_storage['images']
    labels = temp_storage['labels']
    return images, labels


images, labels = load_data('../data/train_im32_till10_full_3.1')


# Dataset class provides image and labels
class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode, toTrain, labl, fts):
        """Initializes the dataset with features and labels."""
        self.num_samples = num_samples  # number of samples
        self.labl = torch.FloatTensor(labl)
        self.toTrain = toTrain
        self.ftT = torch.FloatTensor(fts)
        self.indVal = 0
        if toTrain == 0:
            self.indVal = self.labl.size(0) - num_samples

        print('Preparing Cluster Inputs')
        thirFeat = []
        for sno in range(num_samples):
            secFeat = []
            for iter in range(timesteps):
                feature_list = [
                    self.ftT[sno + self.indVal, iter, :, :],
                    self.ftT[sno + self.indVal, iter + 1, :, :]
                ]
                tempT = torch.stack(feature_list, dim=0)
                secFeat.append(tempT)
            thirFeat.append(torch.stack(secFeat, dim=0))
        self.fts = torch.stack(thirFeat, dim=0)

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idxa):
        """Fetches the sample at index `idxa`."""
        if self.toTrain == 1:
            return self.fts[idxa].to(device), self.labl[idxa].to(device)
        else:
            return self.fts[idxa].to(device), self.labl[self.indVal + idxa].to(device)


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


# Create datasets and dataloaders
def create_dataloaders(num_frames, batch_size, images, labels):
    train_data = RegressionDataset(
        timesteps=num_frames,
        num_samples=60000,
        mode="linear",
        toTrain=1,
        labl=labels,
        fts=images
    )
    val_data = RegressionDataset(
        timesteps=num_frames,
        num_samples=4000,
        mode="linear",
        toTrain=0,
        labl=labels,
        fts=images
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)
    return train_loader, val_loader


train_loader, val_loader = create_dataloaders(num_of_frames, batch_size, images, labels)


##############################
# First Network Definitions #
##############################

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

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=n_kernels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride_size, stride_size)
        )
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())
        self.rnn = SimpleRNN_n(
            input_size=act_size * act_size * n_kernels,
            hidden_size=rnn_units,
            nonlinearity='relu',
            batch_first=True
        ).to(device)
        self.fc = nn.Linear(rnn_units, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        out_conv = torch.stack([self.conv(x[:, i]) for i in range(num_frames)], dim=1)
        out_flat = torch.stack([self.flatten(out_conv[:, i]) for i in range(num_frames)], dim=1)
        out_rnn, _ = self.rnn(out_flat)
        out_fc = torch.stack([self.fc(out_rnn[:, i]) for i in range(num_frames)], dim=1)
        return out_fc, out_conv, out_flat, out_rnn


###############################
# Second Network Definitions  #
###############################

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
    def __init__(self, num_features, adaptation_rate=0.1, recovery_rate=0.1):
        super(AdaptiveLayer, self).__init__()
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate
        self.register_buffer('adaptation', torch.zeros(1, num_features))

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

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=n_kernels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride_size, stride_size)
        )
        self.flatten = nn.Sequential(nn.Flatten(), nn.ReLU())

        # Adaptive Layer
        self.adaptive_layer = AdaptiveLayer(
            num_features=act_size * act_size * n_kernels,
            adaptation_rate=0.1,
            recovery_rate=0.1
        )

        self.rnn = SimpleRNN(
            input_size=act_size * act_size * n_kernels,
            hidden_size=rnn_units,
            nonlinearity='relu',
            batch_first=True,
            adaptation_rate=0.2,
            recovery_rate=0.1
        ).to(device)
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


#####################
# Training Function #
#####################

def train_models(model_class, num_models, num_epochs, train_loader, criterion, model_save_path, is_adaptive=False):
    """
    Trains multiple instances of a given model class.

    Args:
        model_class (class): The neural network class to instantiate.
        num_models (int): Number of models to train.
        num_epochs (int): Number of epochs to train each model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        model_save_path (str): Path prefix to save models.
        is_adaptive (bool): Whether the model includes adaptive layers.

    Returns:
        List of loss histories for each model.
    """
    models = [model_class().to(device) for _ in range(num_models)]
    optimizers = [Adam(model.parameters(), lr=0.00005) for model in models]
    loss_histories = [[] for _ in range(num_models)]

    print(f"Starting training of {num_models} models for {model_class.__name__}...")

    total_epochs = num_models * num_epochs  # Total number of epochs across all models
    with tqdm.tqdm(total=total_epochs, desc="Training Progress") as pbar:
        for model_idx, (model, optimizer) in enumerate(zip(models, optimizers), 1):
            for epoch in range(num_epochs):
                model.train()
                epoch_model_losses = []

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    targets = targets_to_reg(labels, num_of_frames)

                    outputs = model(inputs)[0]  # Get the first output (predictions)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_model_losses.append(loss.item())

                avg_loss = statistics.mean(epoch_model_losses)
                loss_histories[model_idx - 1].append(avg_loss)

                pbar.set_postfix(model=f"{model_idx}/{num_models}", epoch=f"{epoch + 1}/{num_epochs}",
                                 loss=f"{avg_loss:.3e}")
                pbar.update(1)

            # Save the model
            torch.save(model, f'{model_save_path}_{model_idx}.pt')

    print(f"\nFinished training all {num_models} models for {model_class.__name__}.")
    return loss_histories, models


#####################
# Evaluation Function#
#####################

def evaluate_models(models, val_loader, num_frames, num_models):
    """
    Evaluates the models on the validation set and computes Pearson correlations.

    Args:
        models (List[nn.Module]): List of trained models.
        val_loader (DataLoader): DataLoader for validation data.
        num_frames (int): Number of frames.
        num_models (int): Number of models.

    Returns:
        Tuple containing average correlations and standard deviations.
    """
    # Initialize lists to store predictions and targets
    sc_x = [[[] for _ in range(num_frames)] for _ in range(num_models)]
    sc_y = [[[] for _ in range(num_frames)] for _ in range(num_models)]
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
                targets = targets_to_reg(labels, num_frames)

                outputs = model(inputs)[0]  # Get the first output (predictions)
                for fr in range(num_frames):
                    sc_x[model_idx][fr].append(outputs[:, fr, 0].cpu().numpy())
                    sc_y[model_idx][fr].append(outputs[:, fr, 1].cpu().numpy())

                if model_idx == 0:  # Only need to do this once
                    sc_xT.append(targets[:, 0, 0].cpu().numpy())
                    sc_yT.append(targets[:, 0, 1].cpu().numpy())

            # Concatenate all batches
            actual_x = np.concatenate(sc_xT)
            actual_y = np.concatenate(sc_yT)

            for fr in range(num_frames):
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

    return (avg_corr_x, avg_corr_y, avg_corr_combined,
            std_corr_x, std_corr_y, std_corr_combined)


######################
# Plotting Functions #
######################

def plot_correlations(avg_corr_x, avg_corr_y, avg_corr_combined,
                      std_corr_x, std_corr_y, std_corr_combined, num_frames, title_suffix=""):
    """
    Plots the average Pearson correlations with error bars.

    Args:
        avg_corr_x (np.ndarray): Average Pearson correlation for x.
        avg_corr_y (np.ndarray): Average Pearson correlation for y.
        avg_corr_combined (np.ndarray): Average combined Pearson correlation.
        std_corr_x (np.ndarray): Standard deviation for x correlations.
        std_corr_y (np.ndarray): Standard deviation for y correlations.
        std_corr_combined (np.ndarray): Standard deviation for combined correlations.
        num_frames (int): Number of frames.
        title_suffix (str): Suffix to add to the plot title.
    """
    plt.figure(figsize=(12, 6))
    frames = range(1, num_frames + 1)

    plt.errorbar(frames, avg_corr_x, yerr=std_corr_x, label='Average Correlation x', capsize=5, marker='o')
    plt.errorbar(frames, avg_corr_y, yerr=std_corr_y, label='Average Correlation y', capsize=5, marker='s')
    plt.errorbar(frames, avg_corr_combined, yerr=std_corr_combined, label='Average Combined Correlation', capsize=5,
                 marker='^')

    plt.xlabel('Frame')
    plt.ylabel('Average Pearson Correlation')
    plt.title(f'Average Pearson Correlation between Predicted and Actual Values over Time {title_suffix}')
    plt.xticks(frames)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_training_loss(loss_histories_n, loss_histories_adapt, num_epochs_n, num_epochs_adapt):
    """
    Plots the combined training loss for both networks.

    Args:
        loss_histories_n (List[List[float]]): Loss histories for the first network.
        loss_histories_adapt (List[List[float]]): Loss histories for the second network.
        num_epochs_n (int): Number of epochs for the first network.
        num_epochs_adapt (int): Number of epochs for the second network.
    """
    # Convert loss_histories to numpy arrays
    loss_histories_np_n = np.array(loss_histories_n)  # shape (num_models, num_epochs_n)
    loss_histories_np_adapt = np.array(loss_histories_adapt)  # shape (num_models, num_epochs_adapt)

    # Compute mean and std
    mean_losses_n = np.mean(loss_histories_np_n, axis=0)
    std_losses_n = np.std(loss_histories_np_n, axis=0)

    mean_losses_adapt = np.mean(loss_histories_np_adapt, axis=0)
    std_losses_adapt = np.std(loss_histories_np_adapt, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    epochs_n = range(1, num_epochs_n + 1)
    epochs_adapt = range(1, num_epochs_adapt + 1)

    plt.errorbar(epochs_n, mean_losses_n, yerr=std_losses_n, label='Model_n Training Loss', capsize=5, marker='o')
    plt.errorbar(epochs_adapt, mean_losses_adapt, yerr=std_losses_adapt, label='Adaptive Model Training Loss',
                 capsize=5, marker='s')

    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title('Average Training Loss over Epochs for Both Networks')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


####################
# Training Both Networks #
####################

# Define loss function
criterion = nn.MSELoss().to(device)

# Train first set of models (Model_n)
loss_histories_n, trained_models_n = train_models(
    model_class=Model_n,
    num_models=10,
    num_epochs=100,
    train_loader=train_loader,
    criterion=criterion,
    model_save_path='../models/motionnet_fu3_100epochs'
)

# Train second set of models (Model)
loss_histories_adapt, trained_models_adapt = train_models(
    model_class=Model,
    num_models=10,
    num_epochs=100,
    train_loader=train_loader,
    criterion=criterion,
    model_save_path='../models/adaptnet_fu3_100epochs',
    is_adaptive=True
)

#####################
# Evaluate Both Sets #
#####################

print("\nStarting evaluation of Model_n...")
avg_corr_x_n, avg_corr_y_n, avg_corr_combined_n, std_corr_x_n, std_corr_y_n, std_corr_combined_n = evaluate_models(
    models=trained_models_n,
    val_loader=val_loader,
    num_frames=num_of_frames,
    num_models=10
)

print("Starting evaluation of Adaptive Model...")
avg_corr_x_adapt, avg_corr_y_adapt, avg_corr_combined_adapt, std_corr_x_adapt, std_corr_y_adapt, std_corr_combined_adapt = evaluate_models(
    models=trained_models_adapt,
    val_loader=val_loader,
    num_frames=num_of_frames,
    num_models=10
)

#####################
# Plot Correlations #
#####################

# Plot correlations for Model_n
plot_correlations(
    avg_corr_x=avg_corr_x_n,
    avg_corr_y=avg_corr_y_n,
    avg_corr_combined=avg_corr_combined_n,
    std_corr_x=std_corr_x_n,
    std_corr_y=std_corr_y_n,
    std_corr_combined=std_corr_combined_n,
    num_frames=num_of_frames,
    title_suffix="MotionNet-R"
)

# Plot correlations for Adaptive Model
plot_correlations(
    avg_corr_x=avg_corr_x_adapt,
    avg_corr_y=avg_corr_y_adapt,
    avg_corr_combined=avg_corr_combined_adapt,
    std_corr_x=std_corr_x_adapt,
    std_corr_y=std_corr_y_adapt,
    std_corr_combined=std_corr_combined_adapt,
    num_frames=num_of_frames,
    title_suffix="AdaptNet"
)

##########################
# Plot Combined Training Loss #
##########################

plot_training_loss(
    loss_histories_n=loss_histories_n,
    loss_histories_adapt=loss_histories_adapt,
    num_epochs_n=100,
    num_epochs_adapt=100
)
