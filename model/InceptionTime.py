"""
InceptionTime.py
----------------
PyTorch Lightning implementation of InceptionTime for time-series classification.

Architecture:
- Inception 1D modules with multi-scale kernels
- Optional 1x1 bottleneck before large convolutions
- Residual (projection) connection after every 3 Inception modules
- Global Average Pooling + Linear classification head
- Logs sklearn metrics at the end of each epoch (val/test):
    balanced_accuracy, accuracy, precision/recall/f1, cohen_kappa

Input shape: (batch, channels, length)
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    accuracy_score,
)
from sklearn.metrics import confusion_matrix

from loss_function.FocalCosAgeLoss import FocalCosAgeLoss


__all__ = [
    "InceptionModule1D",
    "ResidualAdd",
    "InceptionTimeLightning",
]


def _same_padding(k: int) -> int:
    """Return padding size that approximates 'same' for stride=1 Conv1d."""
    return k // 2


class Conv1dSame(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size[0]
        d = self.dilation[0]
        pad_total = d * (k - 1)
        left = pad_total // 2
        right = pad_total - left
        x = F.pad(x, (left, right))
        # padding=0 vì ta đã pad thủ công
        return F.conv1d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class InceptionModule1D(nn.Module):
    """
    InceptionTime 1D module:
      - (optional) 1x1 bottleneck to reduce channels
      - 3 parallel Conv1d branches with different kernel sizes
      - 1 branch: MaxPool1d -> 1x1 Conv
      - Concatenate along channel dim -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_size: int = 41,
        use_bottleneck: bool = True,
        bottleneck_channels: int = 32,
    ) -> None:
        super().__init__()

        # Match the Keras reference: use (kernel_size - 1), then halve
        base_k = max(1, kernel_size - 1)
        k_sizes: List[int] = [max(1, base_k // (2 ** i)) for i in range(3)]

        self.use_bottleneck = bool(use_bottleneck and in_channels > 1)
        bneck_out = bottleneck_channels if self.use_bottleneck else in_channels

        self.bottleneck = (
            nn.Conv1d(in_channels, bneck_out, kernel_size=1, bias=False)
            if self.use_bottleneck
            else nn.Identity()
        )

        # 3 parallel conv branches
        self.conv_branches = nn.ModuleList(
            [
                Conv1dSame(
                    bneck_out,
                    nb_filters,
                    kernel_size=k,
                    padding="same",
                    bias=False,
                )
                for k in k_sizes
            ]
        )

        # pooling branch: pool -> 1x1 conv
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(num_features=nb_filters * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        z = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(z) for conv in self.conv_branches]
        pool_out = self.pool_conv(self.pool(x))
        # pool_out = F.pad(pool_out, (0, 1))
        y = torch.cat(conv_outs + [pool_out], dim=1)  # (B, 4*nb_filters, L)
        y = self.bn(y)
        y = self.act(y)
        return y


class ResidualAdd(nn.Module):
    """
    Projection shortcut for residual connection.
    Uses 1x1 Conv + BN if channels don't match; identity otherwise.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.proj = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        y = self.proj(x_in)
        y = y + x_out
        return self.act(y)


class InceptionTimeLightning(pl.LightningModule):
    """
    InceptionTime for time-series classification with sklearn metrics logging.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for univariate, >1 for multivariate).
    nb_classes : int
        Number of target classes.
    nb_filters : int, default=32
        Filters per branch (total output channels per module = 4 * nb_filters).
    use_residual : bool, default=True
        Enable residual connection after every 3 Inception modules.
    use_bottleneck : bool, default=True
        Enable 1x1 bottleneck before wide convolutions when in_channels > 1.
    depth : int, default=6
        Number of Inception modules stacked.
    kernel_size : int, default=41
        Base kernel size (effective sizes: (k-1), (k-1)/2, (k-1)/4).
    bottleneck_size : int, default=32
        Channels of the 1x1 bottleneck if enabled.
    lr : float, default=1e-3
        Learning rate for Adam.
    class_weights : Optional[torch.Tensor], default=None
        Weights for CrossEntropyLoss, shape [nb_classes].
    sklearn_average : str, default="macro"
        Averaging for precision/recall/f1 in sklearn ("macro", "weighted", "micro", or None).
    """

    def __init__(
        self,
        in_channels: int,
        nb_classes: int,
        nb_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
        bottleneck_size: int = 32,
        lr: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        sklearn_average: str = "macro",
        use_bmi: bool = False,
        use_sex: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        C_in = in_channels
        C_out = nb_filters * 4  # concat 4 branches

        # build modules
        modules = []
        in_ch = C_in
        for _ in range(depth):
            modules.append(
                InceptionModule1D(
                    in_channels=in_ch,
                    nb_filters=nb_filters,
                    kernel_size=kernel_size,
                    use_bottleneck=use_bottleneck,
                    bottleneck_channels=bottleneck_size,
                )
            )
            in_ch = C_out
        self.inception_modules = nn.ModuleList(modules)

        # residual connector after every 3 modules
        self.use_residual = use_residual
        self.block_size = 3
        n_blocks = depth // self.block_size
        self.residuals = nn.ModuleList()
        for b in range(n_blocks):
            in_res_ch = C_in if b == 0 else C_out
            self.residuals.append(ResidualAdd(in_channels=in_res_ch, out_channels=C_out))

        # head
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Calculate input feature size for FC
        in_feature = C_out + 20 # 20 for basic HRV
        if use_bmi and use_sex:
            in_feature += 2
        elif use_bmi or use_sex:
            in_feature += 1
            
        self.use_bmi = use_bmi
        self.use_sex = use_sex
        
        self.fc = nn.Sequential(
            nn.Linear(in_feature, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # Adding dropout as seen in ResNet example
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, nb_classes),
        )

        # loss
        self.criterion = FocalCosAgeLoss(class_weights=class_weights)

        # hparams
        self.lr = lr
        self.nb_classes = nb_classes
        self.sklearn_average = sklearn_average

        # buffers for sklearn metrics collection
        self._val_pred, self._val_true = [], []
        self._test_pred, self._test_true = [], []
        self.cm_test = None

    # -------------------------- forward --------------------------
    def forward(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        x_full: (B, C, L) where L includes tail metadata
        """
        if x_full.dim() == 2:
            x_full = x_full.unsqueeze(1)

        B, C, L = x_full.shape
        tail = 20
        if self.use_sex and self.use_bmi:
            tail = 22
        elif self.use_sex or self.use_bmi:
            tail = 21
            
        if L < tail:
             raise ValueError(f"Sequence length L={L} < required tail={tail}.")

        # Split RRI and HRV
        x_rri = x_full[:, :, :-tail]
        x_hrv = x_full[:, :, -tail:]
        x_hrv = x_hrv.reshape(B, -1) # (B, tail*C) -> typically C=1 here likely, depends on input

        z = x_rri
        input_res = z
        block_idx = 0

        for i, mod in enumerate(self.inception_modules):
            z = mod(z)
            if self.use_residual and (i % self.block_size == self.block_size - 1):
                z = self.residuals[block_idx](input_res, z)
                input_res = z
                block_idx += 1

        z = self.gap(z).squeeze(-1)  # (B, C_out)
        
        # Concatenate metadata
        z = torch.cat((z, x_hrv), dim=1)
        
        logits = self.fc(z)
        return logits

    # -------------------------- steps ----------------------------
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_pred, self._val_true = [], []

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self._val_pred.append(preds.detach().cpu())
        self._val_true.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if len(self._val_true) == 0:
            return
        y_true = torch.cat(self._val_true).numpy()
        y_pred = torch.cat(self._val_pred).numpy()

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.sklearn_average, zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')


        self.log("val_balanced_acc", bal_acc, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log(f"val_precision", float(prec))
        self.log(f"val_recall", float(rec))
        self.log(f"val_f1", float(f1), prog_bar=True)
        self.log("val_kappa", kappa, prog_bar=False)

    def on_test_epoch_start(self) -> None:
        self._test_pred, self._test_true = [], []

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self._test_pred.append(preds.detach().cpu())
        self._test_true.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if len(self._test_true) == 0:
            return
        y_true = torch.cat(self._test_true).numpy()
        y_pred = torch.cat(self._test_pred).numpy()

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.sklearn_average, zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        self.cm_test = confusion_matrix(y_true, y_pred, labels=np.arange(self.nb_classes))

        self.log("test_balanced_acc", bal_acc)
        self.log("test_accuracy", acc)
        self.log(f"test_precision", float(prec))
        self.log(f"test_recall", float(rec))
        self.log(f"test_f1", float(f1))
        self.log("test_kappa", kappa)

    # -------------------------- optim ----------------------------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=50, min_lr=1e-4
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


if __name__ == "__main__":
    # Minimal sanity check with random input including metadata tail
    # tail = 20 (base) + 2 (sex+bmi) = 22
    B, C, L_ts, K = 2, 1, 128, 4
    metadata_len = 22
    L_total = L_ts + metadata_len
    
    x = torch.randn(B, C, L_total)
    
    # Enable sex and bmi to test full path
    model = InceptionTimeLightning(in_channels=C, nb_classes=K, use_sex=True, use_bmi=True)
    with torch.no_grad():
        logits = model(x)
    print("Input shape:", x.shape)
    print("logits:", logits.shape)  # (2, 4)
