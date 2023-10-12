# %%
import torch
import librosa
import torchaudio
import audiomentations as alb
from IPython.display import Audio


def save_before_after(*waveforms):
    for i, waveform in enumerate(waveforms):
        torchaudio.save(
            uri=f"./audio{i}.mp3",
            src=torch.from_numpy(waveform).view(1, -1),
            sample_rate=16000,
            backend="ffmpeg"
        )


# %%
waveform, sr = librosa.load(
    "dataset/test/0987/vocals.mp3",
    sr=16000,
    mono=True,
    offset=10,
    duration=10
)
waveform2, sr = librosa.load(
    "dataset/test/0001/vocals.mp3",
    sr=16000,
    mono=True,
    offset=10,
    duration=10
)
print(waveform.max(), waveform2.max())
save_before_after(
    waveform,
    waveform2,
    (waveform2 + waveform / waveform.max() * waveform2.max() * 0.9)
)

# %%
transform = alb.GainTransition(
    min_gain_db=-6,
    max_gain_db=6,
    min_duration=0.3,
    max_duration=0.8,
    duration_unit="fraction",
    p=1
)
save_before_after(waveform, transform(waveform, sr))

# %%
