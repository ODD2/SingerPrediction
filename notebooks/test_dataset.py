# %%
from dts import ArtistDataset, AudioMeta
import logging
logging.basicConfig(level="DEBUG")
x = ArtistDataset(
    root_dir="dataset",
    split="train",
    audio_name="vocals.mp3",
    n_fft=512,
    duration=20,
    target_sr=16000,
    close_th=3
)
len(x)
# %%
import random
from tqdm import tqdm
from IPython.display import Audio
# %%
x.class_samples
# %%
import numpy as np
# %%
i = random.randrange(0, len(x))
data = x[i]
Audio(data["waveform"], rate=data["sr"], autoplay=True)
# %%
# for i in tqdm(range(len(x))):
#     data = x[i]
#     assert data["waveform"].shape[0] > 0
# %%
