import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Weighted CE + Focal term + beta * MSE(age)
    - class_medians: median tuổi cho từng lớp
    - class_weights: mảng numpy/tensor shape [C] hoặc None
    - gamma: độ mạnh focal
    - beta: hệ số MSE-age
    """
    def __init__(self, class_medians=(24, 35, 44, 55),
                 class_weights=None, gamma=2.0, beta=0.1):
        super().__init__()
        self.gamma = float(gamma)
        self.beta  = float(beta)

        # buffers (không tạo thuộc tính cùng tên trước khi register)
        self.register_buffer("medians", torch.tensor(class_medians, dtype=torch.float32))
        if class_weights is not None:
            cw = class_weights.detach().clone().float()
        else:
            cw = None
        self.register_buffer("class_weights", cw)

        # CE sẽ dùng buffer weight (có thể None)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        # self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        med = self.medians.to(logits.device)  # [C]

        # --- CE (weighted) ---
        ce_loss = self.ce(logits, labels)

        # --- Focal trên CE ---
        log_probs     = F.log_softmax(logits, dim=1)
        ce_per_sample = F.nll_loss(log_probs, labels, reduction='none')
        pt            = torch.exp(-ce_per_sample).clamp_min(1e-8)
        focal_loss    = ((1 - pt) ** self.gamma * ce_per_sample).mean()

        # --- MSE tuổi (kỳ vọng từ probs @ medians) ---
        probs    = F.softmax(logits, dim=1)
        pred_age = (probs * med).sum(dim=1)
        true_age = med[labels]
        mse_loss = F.mse_loss(pred_age, true_age)

        return ce_loss + focal_loss + self.beta * mse_loss
