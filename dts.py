import os
import csv
import torch
import pickle
import logging
import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from glob import glob
from tqdm import tqdm


def cut_mute(spec, sample_rate, hop_size, onset_th=0.5, close_th=2):
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


class ArtistDataset():
    def __init__(
            self,
            root_dir,
            audio_name,
            split,
            duration,
            mono=True,
            n_fft=512,
            onset_th=1,
            close_th=0.5,
            uni_sample_rate=16000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.n_fft = n_fft
        self.split = split
        self.duration = duration
        self.onset_th = onset_th
        self.close_th = close_th
        self.uni_sample_rate = uni_sample_rate
        self.audio_name = audio_name
        self.mono = mono

        self.load_audio_meta()

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
                waveform, sample_rate = librosa.load(file, sr=None)
                waveform = torch.from_numpy(waveform)
                waveform = F.resample(
                    waveform,
                    orig_freq=sample_rate,
                    new_freq=self.uni_sample_rate
                )
                sample_rate = self.uni_sample_rate
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
                    hop_size=self.n_fft // 2
                )

                # ignore audio without voice segments
                if len(voice_segments_secs) == 0:
                    continue

                # filter the segments that're shoter than 'duration'
                voice_segments_secs = voice_segments_secs[
                    (
                        voice_segments_secs[:, 1] -
                        voice_segments_secs[:, 0]
                    ) > self.duration
                ]

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
        self.singer_label = {}
        with open(os.path.join(self.root_dir, f"{self.split}.txt"), "r") as f:
            csvreader = csv.reader(f)
            for row in csvreader:

                name, singer = row[0], row[1]
                name = name[:-4]
                if (not singer in self.singer_label):
                    self.singer_label[singer] = len(self.singer_label)
                try:
                    self.audio_table[name]["label"] = self.singer_label[singer]
                except Exception as e:
                    logging.warning(
                        f"Failed to Match Singer & Label.({name}, {e})"
                    )
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
        for segment in self.segment_list:
            self.stack_video_clips.append(
                self.stack_video_clips[-1] + segment["clips"]
            )
        self.stack_video_clips.pop(0)

    @property
    def num_classes(self):
        return len(self.singer_label)

    def __len__(self):
        return self.stack_video_clips[-1]

    def __getitem__(self, idx):
        # load segment
        segment_idx, segment_offset, segment_data = self.segment_info(idx)
        # segment & audio attributes
        audio_file = segment_data["data"]["meta"].path
        audio_label = segment_data["data"]["label"]
        audio_offset_sec = (
            segment_data["start_at"] +
            self.duration * segment_offset
        )
        ####### TORCH AUDIO #######
        # audio_sr = torchaudio.info(audio_file,backend="ffmpeg").sample_rate
        # # derive the audio frame offset
        # frame_offset = int(audio_offset_sec * audio_sr)
        # # and the number of frames to retrieve
        # num_frames = int(self.duration * audio_sr)
        # # load the audio segment waveform
        # audio_waveform = torchaudio.load(
        #     audio_file,
        #     frame_offset=frame_offset,
        #     num_frames=num_frames,
        #     backend="ffmpeg"
        # )[0]
        ####### LIBROSA #######
        audio_waveform, audio_sr = librosa.load(
            audio_file,
            sr=None,
            mono=self.mono,
            offset=audio_offset_sec,
            duration=self.duration
        )
        audio_waveform = torch.from_numpy(audio_waveform)

        logging.debug(f"Audio File:{audio_file}")
        logging.debug(f"Audio Label:{audio_label}")
        logging.debug(f"Audio Offset Sec:{audio_offset_sec}")
        logging.debug(f"Audio SR:{audio_sr}")

        return dict(
            label=audio_label,
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
    #     uni_sample_rate=16000
    # )
    x = ArtistDataset(
        root_dir="dataset",
        split="train",
        audio_name="vocals.mp3",
        n_fft=512,
        duration=5,
        uni_sample_rate=16000
    )
    x[4095]
