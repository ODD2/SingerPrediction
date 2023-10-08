# %%
import torch
from singer_identity import load_model
model = load_model('byol')

audio_batch = torch.randn(10,44100)   # Get audio from somewhere (here in 44.1 kHz), shape: (batch_size, n_samples)
embeddings = model(audio_batch)  # shape: (batch_size, 1000)
# %%
embeddings.shape

# %%
