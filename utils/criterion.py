import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(
        self,
        ohem_ratio=0.7,
        n_min_divisor=16,
        ignore_index=255,
    ):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.n_min_divisor = n_min_divisor
        self.ignore_index = ignore_index

    def _ohem_loss(self, pred, target):
        loss = F.cross_entropy(
            pred, target, reduction="none", ignore_index=self.ignore_index
        )

        loss = loss.view(-1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        loss = loss[valid_mask]

        sorted_loss, _ = torch.sort(loss, descending=True)

        num_hard_examples = int(self.ohem_ratio * sorted_loss.size(0))
        hard_loss = sorted_loss[:num_hard_examples]

        return hard_loss.mean()

    def forward(self, pred, target):
        return self._ohem_loss(pred, target)
