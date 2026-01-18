import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import pytorch_lightning as pl
from loss_function.HybridLoss import HybridLoss
from loss_function.CoralLoss import CoralLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    accuracy_score,
)
from .dlutils import *


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return activation()


class SublayerConnection(nn.Module):

    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            # print(self.a)
            # print(torch.mean(torch.abs(x) / torch.abs(out_x)))
            return x + self.dropout(self.a * out_x)


class _ConvEncoderLayer(nn.Module):
    def __init__(self, kernel_size, d_model, d_ff=256, dropout=0.1, activation="relu",
                 enable_res_param=True, norm='batch', small_ks=3, re_param=True, device='cuda:0'):
        super(_ConvEncoderLayer, self).__init__()

        self.norm_tp = norm
        self.re_param = re_param

        # DeepWise Conv. Add & Norm
        if self.re_param:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding='same', groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, self.small_ks, stride=1, padding='same', groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding='same', groups=d_model)
        else:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)

        self.dw_act = get_activation_fn(activation)

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, 1),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d_model, 1, 1))

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merge_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad

        # Lấy weight/bias gốc
        w_large = self.DW_conv_large.weight  # (C, 1, large_ks)
        w_small = self.DW_conv_small.weight  # (C, 1, small_ks)
        b_large = self.DW_conv_large.bias  # (C,)
        b_small = self.DW_conv_small.bias  # (C,)

        # Pad kernel nhỏ cho cùng kích thước rồi cộng (KHÔNG in-place)
        w_small_padded = F.pad(w_small, (left_pad, right_pad), value=0)

        w_merged = w_large + w_small_padded  # tensor mới
        b_merged = b_large + b_small  # tensor mới

        return w_merged, b_merged

    def forward(self, src):  # [B, C, L]

        ## Deep-wise Conv Layer
        if not self.re_param:
            src = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                src = self.sublayerconnect1(src, self.dw_act(large_out + small_out))
            else:  # testing/predict phase
                # 🔹 Không đụng self.DW_infer nữa, chỉ dùng weight/bias merged
                w_merged, b_merged = self._get_merge_param()

                # padding = 'same' cho Conv1d: padding = (kernel_size - 1) // 2
                pad = (self.large_ks - 1) // 2
                merge_out = F.conv1d(
                    src,
                    w_merged,
                    b_merged,
                    stride=1,
                    padding=pad,
                    groups=src.size(1),
                )
                src = self.sublayerconnect1(src, self.dw_act(merge_out))

        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm

        src2 = self.sublayerconnect2(src, src2)  # Add: residual connection with residual dropout

        # Norm: batchnorm or layernorm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2

        return src2


class _ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=[19, 19, 29, 29, 37, 37], dropout=0.1, activation='gelu',
                 n_layers=3, enable_res_param=False, norm='batch', re_param=False, device='cuda:0'):
        super(_ConvEncoder, self).__init__()
        self.layers = nn.ModuleList([_ConvEncoderLayer(kernel_size[i], d_model, d_ff=d_ff, dropout=dropout,
                                                       activation=activation, enable_res_param=enable_res_param,
                                                       norm=norm,
                                                       re_param=re_param, device=device) \
                                     for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class ConvTimeNet_backbone(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, n_layers: int = 3, d_model: int = 128,
                 d_ff: int = 256, dropout=0.1, act: str = "relu", pooling_tp='max', fc_dropout: float = 0.,
                 enable_res_param=False, dw_ks=[7, 13, 19], norm='batch', use_embed=True, re_param=False,
                 device: str = 'cuda:0'):
        r"""ConvTST (Conv-based Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        super(ConvTimeNet_backbone, self).__init__()
        assert n_layers == len(dw_ks), "dw_ks should match the n_layers!"

        self.c_out, self.seq_len = c_out, seq_len

        # Input Embedding
        self.use_embed = use_embed
        self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        # W_pos = torch.empty((seq_len, d_model), device=device)
        # nn.init.uniform_(W_pos, -0.02, 0.02)
        # self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _ConvEncoder(d_model, d_ff, kernel_size=dw_ks, dropout=dropout, activation=act, \
                                    n_layers=n_layers, enable_res_param=enable_res_param, norm=norm, re_param=re_param,
                                    device=device)

        self.flatten = nn.Flatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == 'cat' else d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout)

    def create_head(self, nf, c_out, act="gelu", pooling_tp='max', fc_dropout=0., **kwargs):
        layers = []
        if pooling_tp == 'cat':
            layers = [get_activation_fn(act), self.flatten]
            if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        elif pooling_tp == 'mean':
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == 'max':
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]

        layers += [nn.Linear(nf, c_out)]

        # could just be used in classifying task
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - use_embed=True : [B, c_in, L]
          - use_embed=False: [B, d_model, L]  (đã embed sẵn, như out_patch.permute)
        """
        if self.use_embed:
            # x: (B, c_in, L) -> (B, L, c_in) -> (B, L, d_model) -> (B, d_model, L)
            u = self.W_P(x.transpose(2, 1))
            u = u.transpose(2, 1).contiguous()
        else:
            # đã đúng dạng (B, d_model, L)
            u = x

        # Encoder: nhận (B, d_model, L)
        z = self.encoder(u)  # (B, d_model, L)

        # Head: pool + Linear
        return self.head(z)  # (B, c_out=256)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.name = 'Model'
        patch_size, patch_stride = configs.patch_size, configs.patch_stride

        # DePatch Embedding
        in_channel, out_channel, seq_len = configs.enc_in, configs.d_model, configs.seq_len
        self.depatchEmbedding = DeformablePatch(
            in_channel, out_channel, seq_len,
            patch_size, patch_stride
        )

        # ConvTimeNet Backbone
        new_len = self.depatchEmbedding.new_len
        c_in, c_out = out_channel, configs.num_class
        dropout = configs.dropout
        d_ff, d_model, dw_ks = configs.d_ff, configs.d_model, configs.dw_ks

        block_num, enable_res_param, re_param = len(dw_ks), True, True

        self.main_net = ConvTimeNet_backbone(
            c_in, c_out, new_len, block_num,
            d_model, d_ff, dropout,
            act='gelu', dw_ks=dw_ks,
            enable_res_param=enable_res_param,
            re_param=re_param, norm='batch',
            use_embed=False, device='cuda:0'  # nên chỉnh lại cho phù hợp Lightning nếu cần
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_mark_enc, x_dec, x_mark_dec, mask hiện chị chưa dùng
        out_patch = self.depatchEmbedding(x_enc)  # (B, d_model, N_patch)
        output = self.main_net(out_patch)  # [bs, num_class]
        return output


class ConvTimeNetLightning(pl.LightningModule):
    def __init__(
            self,
            # configs,
            patch_size = 4,
            patch_stride = 2,
            in_channels: int = 1,
            nb_classes: int = 4,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            dropout: float = 0.1,
            d_ff=256,
            d_model=64,
            dw_ks = [19, 19, 29, 29, 37, 37],
            seq_len=1,
            class_weights: Optional[torch.Tensor] = None,
            sklearn_average: str = "macro",
            bmi_sex: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.sklearn_average = sklearn_average

        self.nb_classes = nb_classes
        self.bmi_sex = bmi_sex
        # Loss & metrics
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # -----------------------------------------------------------------
        # trừ đi độ dài hrv
        if self.bmi_sex:
            seq_len -= 22
        else:
            seq_len -= 20
        # DePatch Embedding
        self.depatchEmbedding = DeformablePatch(in_channels, d_model, seq_len, patch_size, patch_stride)

        # ConvTimeNet Backbone
        self.new_len = self.depatchEmbedding.new_len
        c_in, c_out, dropout = d_model, nb_classes, dropout,

        block_num, enable_res_param, re_param = len(dw_ks), True, True

        self.main_net = ConvTimeNet_backbone(
            c_in=c_in,
            c_out=256,  # NOT nb_classes nữa
            seq_len=self.new_len,
            n_layers=block_num,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act='gelu',
            dw_ks=dw_ks,
            enable_res_param=enable_res_param,
            re_param=re_param,
            norm='batch',
            use_embed=False,
            device='cuda:0',
            pooling_tp='max',  # vẫn dùng pooling ở đây
            fc_dropout=0.0,  # có thể để 0, head ngoài lo dropout
        )

        # in_feature = 256 + (20 if concat_hrv else 0)
        in_feature = 256 + 20
        if self.bmi_sex:
            in_feature += 2

        self.head = nn.Sequential(
            nn.Linear(in_feature, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, nb_classes),
        )

        self.criterion = HybridLoss(class_weights=class_weights)
        # self.criterion = CoralLoss(pos_weight=pos_weight)

        self.acc = MulticlassAccuracy(num_classes=nb_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=nb_classes, average="macro")

    # def forward(self, x):
    #     # x: (B, C, L) or (B, L) handled by DataModule
    #     out_patch = self.depatchEmbedding(x)  # [bs, ffn_output(before softmax)]
    #     logits = self.main_net(out_patch.permute(0, 2, 1))
    #     return logits

    def forward(self, x_full):
        # Đảm bảo (B, C, L)
        if x_full.dim() == 2:
            x_full = x_full.unsqueeze(1)

        B, C, L = x_full.shape
        tail = 22 if getattr(self, "bmi_sex", False) else 20
        if L < tail:
            raise ValueError(f"Sequence length L={L} < required tail={tail}.")

        # 1) Tách RRI và HRV
        x_rri = x_full[:, :, :-tail]  # (B, C, L_rri)
        x_hrv = x_full[:, :, -tail:]  # (B, C, tail)
        x_hrv = x_hrv.reshape(B, -1)  # (B, tail*C)

        # 2) De-patch + ConvTimeNet backbone
        out_patch = self.depatchEmbedding(x_rri)  # ví dụ (B, N_patch, d_model)
        x_main = self.main_net(out_patch)

        # 3) Nối feature ConvTimeNet + HRV
        x = torch.cat([x_main, x_hrv], dim=1)  # (B, feature_dim + tail*C)

        # 4) MLP head cuối
        logits = self.head(x)  # (B, nb_classes)
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

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_true = batch

        logits = self(x)
        # Hỗ trợ cả multiclass và binary 1-logit
        if logits.ndim == 2 and logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1)  # (B, C)
        else:
            p1 = torch.sigmoid(logits).view(-1)  # (B,)
            probs = torch.stack([1 - p1, p1], dim=1)  # (B, 2)

        preds = probs.argmax(dim=1)

        return {
                "y_pred": preds.detach().cpu().numpy(),
                # "probs": probs.detach().cpu().numpy(),
                "y_true": y_true.detach().cpu().numpy(),
        }


    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch", "frequency": 1},
        }