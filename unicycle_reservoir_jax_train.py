import os, time
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax, jax.numpy as jnp
from flax.training import train_state
import optax
from unicycle_reservoir_jax import UnicycleReservoir, Readout
from mnist_dataloader import MNISTDataLoader   # returns np.float32 already
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from utils import get_mnist_data
# ---------- Hyper-params ----------
n_inp, n_units, n_classes = 1, 100, 10
seq_len, batch_size, dt = 784, 200, 0.01
lr = 1e-3
epochs = 3
# ----------------------------------

# fixed reservoir -------------------------------------------------------
reservoir = UnicycleReservoir(n_inp=n_inp, n_units=n_units, dt=dt)
init_state = tuple(jnp.zeros((batch_size, n_units)) for _ in range(5))

# --- Torch Readout ---
class ReadoutTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# --- Torch Model Setup ---
device = torch.device("cpu")
readout = ReadoutTorch(input_dim=5 * n_units, output_dim=n_classes).to(device)
optimizer = optim.Adam(readout.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Dummy test functions ---
def test(dataloader):
    readout.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            angular_input = torch.zeros_like(images)

            features = []
            for i in range(images.shape[0]):
                u = jnp.array(images[i].numpy())
                a = jnp.array(angular_input[i].numpy())
                init_state = tuple(jnp.zeros((n_units,)) for _ in range(5))
                final_state = reservoir.integrate_dynamics(u, a, init_state)
                features.append(np.array(final_state[-1]))

            x = torch.tensor(np.stack(features), dtype=torch.float32).to(device)
            logits = readout(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

    return correct / total

# --- Training Loop ---
def train_loop(train_loader, valid_loader=None, test_loader=None, n_epochs=10):
    for epoch in range(n_epochs):
        readout.train()
        progress_bar = tqdm(train_loader)
        for images, labels in progress_bar:
            images = images.reshape(images.shape[0], 1, 784).permute(2, 0, 1)
            angular_input = torch.zeros_like(images)

            optimizer.zero_grad()
            u = jnp.array(images.numpy())
            a = jnp.array(angular_input.numpy())
            init_state = tuple(jnp.zeros((batch_size, n_units)) for _ in range(5))
            start = time.time()
            states_over_time = reservoir.integrate_dynamics(u, a, init_state)
            end = time.time()
            print(f"Integration time: {end - start:.3f}s")

            # Prepare readout input
            readout_input = reservoir.get_readout_input(states_over_time)
            start = time.time()
            x = torch.from_numpy(np.array(readout_input))
            y = labels
            end = time.time()
            print(f"Readout input preparation time: {end - start:.3f}s")
            start = time.time()
            logits = readout(x)
            end = time.time()
            print(f"Readout forward pass time: {end - start:.3f}s")

            start = time.time()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            end = time.time()
            print(f"Backpropagation and optimization time: {end - start:.3f}s")

            progress_bar.set_postfix(loss=loss.item(), time=f"{end - start:.3f}s")

        if valid_loader:
            val_acc = test(valid_loader)
            print(f"Validation Accuracy: {val_acc:.4f}")
        if test_loader:
            test_acc = test(test_loader)
            print(f"Test Accuracy: {test_acc:.4f}")


train_loader, valid_loader, test_loader = get_mnist_data(bs_train=batch_size, bs_test=batch_size, classes=[0,1,2,3,4,5,6,7,8,9], 
                                                         new_fraction=0.5, test_fraction=0.5, path="data/")
train_loop(train_loader, valid_loader, test_loader, n_epochs=10)