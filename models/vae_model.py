import os
from pathlib import Path

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNetVAE
from monai.transforms import Activations, AsDiscrete, Compose
from tqdm import tqdm

from metrics import ValidationEvaluator


class VAEModel:
    def __init__(self, device, train_loader, val_loader, epochs):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.learning_rate = 0.01
        self.weight_decay = 1e-8
        self.amp = False
        self.global_step = 0
        self.log_table = []
        self.weights_dir = os.path.join(os.path.abspath(__file__), "..", "weights")
        self.build_model()

    def build_model(self):
        self.model = SegResNetVAE(
            input_image_size=(128, 128, 64),
            vae_estimate_std=True,
            vae_default_std=0.3,
            vae_nz=256,
            spatial_dims=3,
            init_filters=8,
            in_channels=4,
            out_channels=3,
            dropout_prob=None,
            act="RELU",
            norm=("GROUP", {"num_groups": 8}),
            use_conv_final=True,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            upsample_mode="nontrainable",
        ).to(self.device)

        self.post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(
            include_background=True, reduction="mean_batch"
        )

        self.loss_function = DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            foreach=True,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train(self):
        best_metric = -1
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch}/{self.epochs}",
                unit="img",
            ) as pbar:
                for batch in self.train_loader:
                    images, target = batch["images"].to(self.device), batch["mask"].to(
                        self.device
                    )
                    VAE_loss = 0
                    with torch.autocast(
                        self.device.type if self.device.type != "mps" else "cpu",
                        enabled=self.amp,
                    ):
                        masks_pred, VAE_loss = self.model(images)
                        loss = self.loss_function(masks_pred, target) + VAE_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    pbar.update(images.shape[0])
                    self.global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                self.log_table.append(
                    {"epoch": epoch + 1, "loss": epoch_loss / len(self.train_loader)}
                )

            ValidationEvaluator.validate(
                self.model,
                self.val_loader,
                self.dice_metric,
                self.dice_metric_batch,
                self.post_trans,
                self.device,
                self.amp,
                VAE_param=True,
            )
            current_metric_value = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
            self.dice_metric_batch.reset()
            if current_metric_value > best_metric:
                Path(self.weights_dir).mkdir(parents=True, exist_ok=True)
                state_dict = self.model.state_dict()
                saving_path = os.path.join(
                    self.weights_dir, "best_checkpoint_dice_val_score.pth"
                )
                torch.save(state_dict, saving_path)
                best_metric = current_metric_value
