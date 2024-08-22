import os

import lightning as L
import torch
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from timm.optim import SGDP, AdamW

from data import CityscapesDataModule
from models import SegmentationNetwork, reparameterize_model
from utils.criterion import Criterion
from utils.metrics import Metrics


class LigthningNetwork(L.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()

        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.example_input_array = torch.rand(1, 3, 1024, 2048)

        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.iou_metrics = Metrics(
            class_names=self.class_names,
            table_fmt="fancy_grid",
            missing_val="-",
            ignore_index=255,
            eps=1e-6,
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        rep_model = reparameterize_model(self.model)
        y_hat = rep_model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.iou_metrics.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.iou_metrics.collect()

        self.log("mIoU", self.iou_metrics.metrics["iou"].mean(), on_epoch=True)

        self.print(f"\n\n{self.iou_metrics}\n\n")

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-2,
            # momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.trainer.max_steps, power=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train(
    exp_name,
    save_dir="exp",
    ohem_ratio=0.9,
    n_min_divisor=16,
    ignore_index=255,
    root="datasets/cityscapes",
    train_size=(1024, 1024),
    test_size=(1024, 2048),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    train_batch_size=8,
    test_batch_size=1,
    num_workers=8,
    precision="16-mixed",
    max_steps=120_000,
):

    dm = CityscapesDataModule(
        root=root,
        train_size=train_size,
        test_size=test_size,
        mean=mean,
        std=std,
        ignore_index=ignore_index,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    dm.setup("fit")

    seg_model = SegmentationNetwork(
        layers=[2, 2, 4, 2],
        embed_dims=[48, 96, 192, 384],
        mlp_ratios=[3, 3, 3, 3],
        downsamples=[True, True, True, True],
        head_dim=64,
    )
    criterion = Criterion(ohem_ratio=ohem_ratio, n_min_divisor=n_min_divisor)

    model = LigthningNetwork(seg_model, criterion)

    work_dir = os.path.join(save_dir, exp_name)

    tb_logger = pl_loggers.TensorBoardLogger(work_dir, name="tb_logs", log_graph=True)
    csv_logger = pl_loggers.CSVLogger(work_dir, name="csv_logs")

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=work_dir,
        filename="checkpoint-{epoch:02d}-{mIoU:.2f}",
        monitor="mIoU",
        mode="max",
        save_last=True,
        save_top_k=5,
        every_n_epochs=5,
    )
    # throughput_logger = pl_callbacks.ThroughputMonitor(
    #     batch_size_fn=lambda x: x[0].shape[0]
    # )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        callbacks=[checkpoint_callback],
        logger=[tb_logger, csv_logger],
        max_steps=max_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=5,
        benchmark=True,
        default_root_dir=work_dir,
        # detect_anomaly=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(exp_name="cityscapes_v0")
