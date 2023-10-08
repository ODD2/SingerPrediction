import torch
import wandb
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
from dts import ArtistDataset, AudioMeta
from singer_identity import load_model
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
# logging.basicConfig(level="DEBUG")
DEVICE = "cuda"
VALID_RATIO = 0.5


# create model
class Base(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = load_model('byol')
        self.encoder.requires_grad_(False)
        self.head = torch.nn.Sequential(
            nn.Dropout(),
            nn.LayerNorm(1000),
            nn.Linear(1000, classes)
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def evaluate(self, x, y):
        logits = self(x)
        return dict(
            loss=torch.nn.functional.cross_entropy(logits, y),
            logits=logits
        )

    def train(self, mode=True):
        if (mode):
            self.encoder.eval()
            self.head.train()
        else:
            self.encoder.eval()
            self.head.eval()


# validation function
@torch.no_grad()
def valid(model, dataloaders):
    model.eval()
    dataset_logits = []
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
        dataset_logits += logits.softmax(dim=-1).tolist()
        dataset_labels += y.tolist()
        dataset_loss.append(loss.detach().cpu().item())

    top1 = top_k_accuracy_score(dataset_labels, dataset_logits, k=1)
    top3 = top_k_accuracy_score(dataset_labels, dataset_logits, k=3)
    loss = sum(dataset_loss) / len(dataset_loss)

    return dict(
        top1=top1,
        top3=top3,
        loss=loss
    )


# train function
def train(model, dataloaders, optimizer, epochs):
    valid_interval = int(len(dataloaders["train"]) * VALID_RATIO)
    global_step = 0
    for _epoch in range(epochs):
        model.zero_grad()
        model.train()
        _batch = 0
        for batch in tqdm(dataloaders["train"]):
            _batch += 1
            global_step += 1

            x = batch["waveform"]
            y = batch["label"]

            result = model.evaluate(
                x=x.to(DEVICE),
                y=y.to(DEVICE)
            )

            loss = result["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log(
                {
                    "train/loss": loss
                },
                step=global_step
            )

            if ((_batch % valid_interval) == (valid_interval - 1)):
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


# main
def main(args):
    # create dataloaders
    dataloaders = {
        split: DataLoader(
            ArtistDataset(
                root_dir="./dataset/",
                audio_name="vocals.mp3",
                split=split,
                duration=args.duration
            ),
            shuffle=(True if split == "train" else False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
        )
        for split in ["train", "valid"]
    }
    # create model
    model = Base(dataloaders["train"].dataset.num_classes)
    model.to(DEVICE)
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters())
    # init wandb
    wandb.init(project="NTUHW1")
    # train model
    train(
        model,
        dataloaders,
        optimizer,
        epochs=args.epoch
    )
    # terminate
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
