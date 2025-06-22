import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# VAE definition (same as in training)
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        onehot = F.one_hot(labels, 10).float()
        z_cat = torch.cat([z, onehot], dim=1)
        return self.decoder(z_cat)

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# Load model
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pt", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate 5 Images"):
    with torch.no_grad():
        z = torch.randn(5, 20)
        labels = torch.tensor([digit] * 5)
        samples = model.decode(z, labels).view(-1, 1, 28, 28)

        grid = make_grid(samples, nrow=5, padding=2)
        npimg = grid.numpy()

        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
