# %%

import numpy as np
from torchaudio.utils import download_asset
from matplotlib.patches import Rectangle
from IPython.display import Audio
import matplotlib.pyplot as plt
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


print(torch.__version__)
print(torchaudio.__version__)

N_FFT = 512

# %%

torch.random.manual_seed(0)


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower",
              aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


# %%
# Load audio
SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(
    # "dataset/train/aerosmith/Aerosmith/02-Somebody/vocals.mp3"
    "dataset/valid/fleetwood_mac/Tango_In_The_Night/03-Everywhere/vocals.mp3"
)
SPEECH_WAVEFORM = SPEECH_WAVEFORM[:, 970200:3131100]
SPEECH_WAVEFORM = torchaudio.functional.resample(
    SPEECH_WAVEFORM, SAMPLE_RATE, new_freq=16000
)
SAMPLE_RATE = 16000


# Define transform
spectrogram = T.Spectrogram(n_fft=N_FFT)

# Perform transform
spec = spectrogram(SPEECH_WAVEFORM)
# # Plot spectrogram and waveform
fig, axs = plt.subplots(2, 1)
plot_waveform(
    SPEECH_WAVEFORM,
    SAMPLE_RATE,
    title="Original waveform", ax=axs[0]
)
plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
Audio(SPEECH_WAVEFORM, rate=SAMPLE_RATE)
# %%
spec.shape
# %%

torch.sum(spec, dim=1)[0, 0]

# %%


def cut_mute(spect, sample_rate, hop_size):
    def index_to_second(idx, re=False):
        if (re):
            return idx / hop_size * sample_rate
        else:
            return idx * hop_size / sample_rate

    spect = torch.mean(spect, dim=1)
    plt.plot(range(spect.shape[1])[:600], spect[0][:600])
    plt.show()
    split = []
    onset = 0
    close = 0
    status = 1
    onset_th = 0.5 * (sample_rate / hop_size)
    close_th = 2 * (sample_rate / hop_size)

    channels = spect.shape[0]

    spect = torch.cat(
        [
            spect,
            torch.zeros((channels, int(close_th) + 1)),
            torch.ones((channels, 1))
        ],
        dim=1
    ).transpose(0, 1)

    is_sound = torch.any(
        ~(spect < 1e-1),
        dim=1,
        keepdim=True
    ).flatten().tolist()

    for i, v in enumerate(is_sound):
        if (status == 1):
            if (not v):
                close = i
                status = 0
        else:
            if (v):
                if ((close - onset) > onset_th):
                    if ((i - close) > close_th):
                        interval = (
                            index_to_second(onset),
                            index_to_second(close)
                        )
                        split.append(interval)
                        onset = i
                    else:
                        pass
                else:
                    onset = i
                status = 1

    return np.array(split)


print(cut_mute(spec, SAMPLE_RATE, N_FFT // 2))

# %%
