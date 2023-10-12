import os
import torch
import random
import librosa
import argparse
import numpy as np
import torchaudio.transforms as T
import torchaudio.functional as F

from glob import glob
from tqdm import tqdm
from dts import cut_mute, SingerLabel
from train import Base, load_singer_anchors, TARGET_SR
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix
N_FFT = 512
ONSET_TH = 0.1
CLOSE_TH = 10
DEVICE = "cuda"


@torch.inference_mode()
def main(args):
    # import model & load weights
    model = Base(len(SingerLabel), load_singer_anchors(args.duration))
    model.load_state_dict(torch.load(args.weight_path, "cpu"))
    model.eval()
    model.to(DEVICE)

    dataset_labels = []
    dataset_probs = []
    dataset_preds = []
    # testing audio file list
    audio_files = sorted(glob(os.path.join(args.root_dir, args.glob_exp)))

    for file in tqdm(audio_files):
        # determine the item idx
        label = file.split('/')[-4]

        # find singing segment
        waveform, sample_rate = librosa.load(file, sr=TARGET_SR, mono=False)
        waveform = torch.from_numpy(waveform)

        # convert to specrogram
        spec = F.spectrogram(
            waveform=waveform,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=N_FFT // 2,
            power=2,
            normalized=False,
            window=torch.hann_window(N_FFT),
            pad=0,
        )

        # find singing segment with parameters
        voice_segments_secs = cut_mute(
            spec,
            sample_rate=sample_rate,
            hop_size=N_FFT // 2,
            onset_th=ONSET_TH,
            close_th=CLOSE_TH
        )

        # convert to mono after mute cutting
        waveform = waveform.mean(dim=0)

        if (len(voice_segments_secs) == 0):
            continue

        # concat the singing segments for latter.
        concat_waveform = torch.cat(
            [
                torch.cat(
                    [
                        waveform[
                            int(beg_sec * sample_rate):
                            int(end_sec * sample_rate)
                        ],
                        torch.zeros(int(sample_rate // 2))
                    ],
                    dim=0
                )
                for beg_sec, end_sec in voice_segments_secs
            ],
            dim=0
        )

        # crop the segments in to specified duration
        cropped_segments = []
        for duration in range(args.duration, 3, -1):
            beg_sample = 0
            while (
                (
                    beg_sample + sample_rate * duration
                ) < concat_waveform.shape[0]
            ):
                end_sample = int(beg_sample + sample_rate * duration)
                cropped_segments.append(
                    concat_waveform[beg_sample:end_sample]
                )
                beg_sample = end_sample

            # if the concat_waveform is able to meet the specified duration, break the loop.
            if len(cropped_segments) > 0:
                break

        if (len(cropped_segments) == 0):
            # if the concat_waveform is unalbe to meeat the minimum duration, ignore it.
            print(
                f"undable to process audio {file}, due to missing singing segments."
            )
            # maybe I can win the lottery.
            sorted_classes = [
                i for i in range(len(SingerLabel))
            ]
            random.shuffle(sorted_classes)
        else:
            # average the segment scores as the song's score
            cropped_segments = torch.stack(cropped_segments)
            logits = []
            for beg_batch in range(0, cropped_segments.shape[0], args.batch_size):
                end_batch = beg_batch + args.batch_size
                logits.append(
                    model(cropped_segments[beg_batch:end_batch].to(DEVICE))
                )
            logits = torch.cat(logits, dim=0).cpu()
            probs = logits.softmax(dim=-1).mean(dim=0)
            dataset_probs.append(probs.flatten().tolist())
            dataset_labels.append(SingerLabel[label].value)
            dataset_preds.append(probs.flatten().argmax().item())

    top1 = top_k_accuracy_score(
        dataset_labels,
        dataset_probs,
        k=1
    )

    top3 = top_k_accuracy_score(
        dataset_labels,
        dataset_probs,
        k=3
    )

    conf = confusion_matrix(dataset_labels, dataset_preds)
    print(top1, top3, conf)


# done
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--root_dir", type=str, default="./dataset/valid/")
    parser.add_argument("--glob_exp", type=str, default="*/*/*/vocals.mp3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--duration", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
