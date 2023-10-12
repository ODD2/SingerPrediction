import os
import torch
import wandb
import pickle
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from dts import ArtistDataset, AudioMeta, SingerLabel
from singer_identity import load_model
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
# logging.basicConfig(level="DEBUG")
DEVICE = "cuda"
TARGET_SR = 44100
BEST_SCORE = float("-inf")


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# create model
class Base(nn.Module):
    def __init__(self, num_classes, label_anchor, embed_dim=1280, input_sr=44100):
        super().__init__()
        self.encoder = load_model('byol', input_sr=input_sr)
        self.encoder.requires_grad_(False)
        self.tune_layers = [-1, -2, -3, -4, -5]
        for l in self.tune_layers:
            self.encoder.encoder.net[2].features[l].requires_grad_(True)
        self.encoder.encoder.net[2].classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, embed_dim)
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.label_anchor_waveform = [
            label_anchor[label]
            for label in SingerLabel
        ]
        self.label_anchor_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.label_anchor_waveform[label].shape[0]
                    ),
                    requires_grad=True
                )
                for label in SingerLabel
            ]
        )

    def get_features(self, x):
        return self.head(self.encoder(x))

    def forward(self, x):
        features = self.get_features(x)
        logits = [None for _ in range(self.num_classes)]

        for i, waveform in enumerate(self.label_anchor_waveform):
            logits[i] = (
                (
                    (self.embed_dim**(-0.5)) *
                    (features @ self.get_features(waveform.to(x.device)).T)
                ) @
                self.label_anchor_weights[i].softmax(dim=-1)
            )

        logits = torch.stack(logits, dim=1)
        return logits

    def evaluate(self, x, y, weight=None, train=False):
        logits = self(x)

        if (train):
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=-1),
                y
            )
        else:
            loss = torch.nn.functional.cross_entropy(
                logits,
                y
            )

        return dict(
            loss=loss,
            logits=logits
        )

    def train(self, mode=True):
        if (mode):
            self.encoder.eval()
            self.head.train()
        else:
            self.encoder.eval()
            self.head.eval()


def class_sample_to_weight(class_samples):
    class_samples = torch.tensor(
        class_samples,
        dtype=float,
        device=DEVICE
    )
    weight = 1 / (class_samples / class_samples.mean())
    weight[weight < 0.5] = 0.5
    weight[weight > 2] = 2
    return weight


def load_singer_anchors(duration=20, file_path="singer_samples.pickle"):
    with open(file_path, "rb") as f:
        singer_samples = pickle.load(f)
        label_anchor = {}
        segment_samples = duration * TARGET_SR
        for singer in singer_samples:
            label = SingerLabel[singer]
            anchors = []
            for _data in singer_samples[singer]:
                full_audio = torchaudio.functional.resample(
                    waveform=torch.from_numpy(_data["waveform"]),
                    orig_freq=_data["sr"],
                    new_freq=TARGET_SR
                )
                beg_sample = 0
                while ((beg_sample + segment_samples) <= full_audio.shape[0]):
                    anchors.append(
                        full_audio[beg_sample:(beg_sample + segment_samples)]
                    )
                    beg_sample += segment_samples
            label_anchor[label] = torch.stack(
                anchors,
                dim=0
            )
    return label_anchor


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# validation function
@torch.no_grad()
def valid(model, dataloaders):
    global BEST_SCORE
    model.eval()
    dataset_probs = []
    dataset_labels = []
    dataset_loss = []

    for batch in tqdm(dataloaders["valid"]):
        x = batch["waveform"]
        y = batch["label"]
        result = model.evaluate(
            x=x.to(DEVICE),
            y=y.to(DEVICE)
        )
        loss = result["loss"]
        logits = result["logits"]
        dataset_probs += logits.softmax(dim=-1).tolist()
        dataset_labels += y.argmax(dim=1).tolist()
        dataset_loss.append(loss.mean().detach().cpu().item())

    top1 = top_k_accuracy_score(dataset_labels, dataset_probs, k=1)
    top3 = top_k_accuracy_score(dataset_labels, dataset_probs, k=3)
    loss = sum(dataset_loss) / len(dataset_loss)

    if (top1 > BEST_SCORE):
        BEST_SCORE = top1
        torch.save(model.state_dict(), f"models/{wandb.run.id}.pt")

    return dict(
        top1=top1,
        top3=top3,
        loss=loss
    )


# train function
def train(model, dataloaders, optimizer, epochs, lr_scheduler):
    global_step = 0
    train_dataloader = dataloaders["train"]

    class_weight = class_sample_to_weight(
        train_dataloader.dataset.class_samples
    )

    for _epoch in range(epochs):
        wandb.log(
            {"epoch": _epoch},
            step=global_step
        )

        model.zero_grad()
        model.train()
        wandb.log(
            {
                "lr": get_lr(optimizer)
            }
        )
        _batch = 0
        for batch in tqdm(train_dataloader):
            _batch += 1
            global_step += 1

            x = batch["waveform"]
            y = batch["label"]

            result = model.evaluate(
                x=x.to(DEVICE),
                y=y.to(DEVICE),
                weight=class_weight,
                train=True
            )

            loss = result["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()

            wandb.log(
                {
                    "train/loss": loss
                },
                step=global_step
            )

        # validation
        metrics = valid(model, dataloaders)
        model.zero_grad()
        model.train()
        wandb.log(
            {
                f"valid/{name}": value
                for name, value in metrics.items()
            },
            step=global_step
        )

        # step lr scheduler
        if (not type(lr_scheduler) == type(None)):
            lr_scheduler.step()


# main
def main(args):
    set_seed(1019)
    # create dataloaders
    dataloaders = {
        split: DataLoader(
            ArtistDataset(
                root_dir="./dataset/",
                audio_name="vocals.mp3",
                split=split,
                duration=args.duration,
                target_sr=TARGET_SR
            ),
            shuffle=(True if split == "train" else False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=(True if split == "train" else False)
        )
        for split in ["train", "valid"]
    }

    # create singer anchors
    label_anchor = load_singer_anchors(duration=args.duration)

    # create model
    num_classes = dataloaders["train"].dataset.num_classes
    model = Base(
        num_classes=num_classes,
        label_anchor=label_anchor,
        input_sr=TARGET_SR
    )
    model.to(DEVICE)

    # create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0
    )

    # create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        div_factor=10,
        final_div_factor=10,
        total_steps=args.epoch,
        pct_start=0.1,
        anneal_strategy="linear"
    )

    # init wandb
    wandb.init(
        project="NTUHW1",
        mode=("offline"if args.test else "online")
    )
    wandb.watch(models=model, log="gradients", log_freq=100)

    # train model
    train(
        model,
        dataloaders,
        optimizer,
        epochs=args.epoch,
        lr_scheduler=lr_scheduler
    )

    # terminate
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
