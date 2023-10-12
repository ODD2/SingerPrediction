import os
import csv
import torch
import pickle
import random
import logging
import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from glob import glob
from tqdm import tqdm
from enum import IntEnum, auto
import audiomentations as alb


def cut_mute(spec, sample_rate, hop_size, onset_th=0.1, close_th=1):
    """
        onset_th: determine the valid interval for a signing segment.
        close_th: determine the maximum interval to concatenate between two signing segments.
    """
    def index_to_second(idx, re=False):
        if (re):
            return idx / hop_size * sample_rate
        else:
            return idx * hop_size / sample_rate

    spec = torch.mean(spec, dim=1)
    split = []
    onset = 0
    close = 0
    status = 1
    onset_th = onset_th * (sample_rate / hop_size)
    close_th = close_th * (sample_rate / hop_size)

    channels = spec.shape[0]
    spec = torch.cat(
        [
            spec,
            torch.zeros((channels, int(close_th) + 1)),
            torch.ones((channels, 1))
        ],
        dim=1
    ).transpose(0, 1)

    is_sound = torch.any(
        ~(spec < 1e-1),
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


class AudioMeta:
    def __init__(self, name, intervals):
        self.name = name
        self.clips = len(intervals)
        self.intervals = intervals

    def set_path(self, path):
        self.path = path


class SingerLabel(IntEnum):
    aerosmith = 0
    beatles = auto()
    creedence_clearwater_revival = auto()
    cure = auto()
    dave_matthews_band = auto()
    depeche_mode = auto()
    fleetwood_mac = auto()
    garth_brooks = auto()
    green_day = auto()
    led_zeppelin = auto()
    madonna = auto()
    metallica = auto()
    prince = auto()
    queen = auto()
    radiohead = auto()
    roxette = auto()
    steely_dan = auto()
    suzanne_vega = auto()
    tori_amos = auto()
    u2 = auto()


class ArtistDataset():
    def __init__(
            self,
            root_dir,
            audio_name,
            split,
            duration,
            mono=True,
            n_fft=512,
            onset_th=0.1,
            close_th=1.0,
            target_sr=16000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.n_fft = n_fft
        self.split = split
        self.duration = duration
        self.onset_th = onset_th
        self.close_th = close_th
        self.target_sr = target_sr
        self.audio_name = audio_name
        self.mono = mono

        self.load_audio_meta()

        self.augmentations = alb.Compose([
            alb.TanhDistortion(),
            alb.AddGaussianNoise(),
            alb.TimeStretch(),
            alb.Reverse(),
            alb.Gain(),
            alb.Mp3Compression(
                min_bitrate=8,
                max_bitrate=32
            ),
            alb.GainTransition(
                min_gain_db=-6,
                max_gain_db=6,
                min_duration=0.3,
                max_duration=0.8,
                duration_unit="fraction",
                p=1
            )
        ])

    def load_audio_meta(self):
        assert not self.split == "test", "test dataset not implemented"

        # cache file
        cache_file = f".cache/artist-{self.split}.pickle"

        base_dir = os.path.join(self.root_dir, self.split)
        # load cache if exists
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.audio_metas = pickle.load(f)
        else:
            # build item list and cache
            self.audio_metas = []

            for file in tqdm(glob(os.path.join(base_dir, "*/*/*/", self.audio_name))):
                # load audio and align
                waveform, sample_rate = librosa.load(file, sr=None, mono=False)
                waveform = torch.from_numpy(waveform)
                waveform = F.resample(
                    waveform,
                    orig_freq=sample_rate,
                    new_freq=self.target_sr
                )
                sample_rate = self.target_sr
                # convert to specrogram
                spec = F.spectrogram(
                    waveform=waveform,
                    n_fft=self.n_fft,
                    win_length=self.n_fft,
                    hop_length=self.n_fft // 2,
                    power=2,
                    normalized=False,
                    window=torch.hann_window(self.n_fft),
                    pad=0,
                )
                # find singing segment with parameters
                voice_segments_secs = cut_mute(
                    spec,
                    sample_rate=sample_rate,
                    hop_size=self.n_fft // 2,
                    onset_th=self.onset_th,
                    close_th=self.close_th
                )

                # ignore audio without voice segments
                if len(voice_segments_secs) == 0:
                    continue

                # save audio metadata for further usage
                name = file[(len(base_dir) + 1):-(len(self.audio_name) + 1)]
                self.audio_metas.append(
                    AudioMeta(name, voice_segments_secs)
                )

            # cache metas
            os.makedirs(".cache/", exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.audio_metas, f)

        # build audio table
        self.audio_table = {}
        for meta in self.audio_metas:
            meta.set_path(os.path.join(base_dir, meta.name, self.audio_name))
            self.audio_table[meta.name] = {
                "meta": meta
            }

        # read csv and build label
        with open(os.path.join(self.root_dir, f"{self.split}.txt"), "r") as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                name, singer = row[0], row[1]
                name = name[:-4]
                try:
                    self.audio_table[name]["label"] = SingerLabel[singer]
                except Exception as e:
                    logging.warning(
                        f"Failed to Match Singer & Label.({name}, {e})"
                    )

        # remove audio if not specified by the split file
        for name in list(self.audio_table.keys()):
            if (not "label" in self.audio_table[name]):
                self.audio_table.pop(name)

        # build segment list
        self.segment_list = []
        for audio_data in self.audio_table.values():
            for interval in audio_data["meta"].intervals:
                clips = int((interval[1] - interval[0]) // self.duration)
                if clips > 0:
                    self.segment_list.append(
                        {
                            "data": audio_data,
                            "clips": clips,
                            "start_at": interval[0]
                        }
                    )

        # build auxiliary structures.
        self.stack_video_clips = [0]
        self.class_samples = [0 for _ in range(self.num_classes)]
        for segment in self.segment_list:
            clips = segment["clips"]
            label = segment["data"]["label"]
            self.stack_video_clips.append(
                self.stack_video_clips[-1] + clips
            )
            self.class_samples[label] += clips
        self.stack_video_clips.pop(0)

    @property
    def num_classes(self):
        return len(SingerLabel)

    def __len__(self):
        return self.stack_video_clips[-1]

    def __getitem__(self, idx):
        if (self.split == "train"):
            if (random.random() < 0.5):
                ######### MIXED ##########
                data1 = self.getitem(idx)
                data2 = self.getitem(random.choice(range(len(self))))
                data1["waveform"] += (
                    data2["waveform"] / data2["waveform"].max() *
                    data1["waveform"].max() * 0.5
                )
                data1["label"] = (
                    data1["label"] * 0.7 +
                    data2["label"] * 0.3
                )
                result = data1
            else:
                ######### GENERIC ##########
                result = self.getitem(idx)

            # do augmentation
            # waveform = self.augmentations(
            #     samples=result["waveform"],
            #     sample_rate=result["sr"]
            # )
            # result["waveform"] = waveform.copy()

        elif (self.split == "valid"):
            result = self.getitem(idx)
        else:
            raise NotImplementedError()

        return result

    def getitem(self, idx):
        # load segment
        segment_idx, segment_offset, segment_data = self.segment_info(idx)
        # segment & audio attributes
        audio_file = segment_data["data"]["meta"].path
        audio_label = segment_data["data"]["label"]
        audio_offset_sec = (
            segment_data["start_at"] +
            self.duration * segment_offset
        )
        # load audio
        audio_waveform, audio_sr = librosa.load(
            audio_file,
            sr=self.target_sr,
            mono=self.mono,
            offset=audio_offset_sec,
            duration=self.duration
        )

        logging.debug(f"Audio File:{audio_file}")
        logging.debug(f"Audio Label:{audio_label}")
        logging.debug(f"Audio Offset Sec:{audio_offset_sec}")
        logging.debug(f"Audio SR:{audio_sr}")

        label = np.array([
            (1. if i == audio_label else 0.)
            for i in range(self.num_classes)
        ])

        return dict(
            label=label,
            waveform=audio_waveform,
            sr=audio_sr
        )

    def segment_info(self, idx):
        segment_idx = next(
            i for i, x in enumerate(self.stack_video_clips)
            if idx < x
        )
        if segment_idx == 0:
            segment_offset = idx
        else:
            segment_offset = idx - self.stack_video_clips[segment_idx - 1]
        return segment_idx, segment_offset, self.segment_list[segment_idx]


if __name__ == "__main__":
    pass
    # ArtistDataset(
    #     root_dir="dataset",
    #     split="train",
    #     audio_name="vocals.mp3",
    #     n_fft=512,
    #     target_sr=16000
    # )
    x = ArtistDataset(
        root_dir="dataset",
        split="train",
        audio_name="vocals.mp3",
        n_fft=512,
        duration=10,
        target_sr=16000
    )
    x[4493]
