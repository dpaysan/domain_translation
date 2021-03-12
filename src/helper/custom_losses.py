from abc import ABC
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

# Todo make the implementation more general and not actively broadcast for the regression losses


class RobustWeightedMSELoss(MSELoss, ABC):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super(RobustWeightedMSELoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #target = target.float().view(-1,1)
        loss = torch.mean(torch.pow((target - input), 4))
        return loss


class RobustMSELoss(MSELoss, ABC):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor):
        target = target.float().view(-1, 1)
        return super().forward(input=input, target=target)


class RobustL1Loss(L1Loss, ABC):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.float().view(-1, 1)
        return super().forward(input=input, target=target)


class RobustBCELoss(BCELoss, ABC):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(
            weight=weight, size_average=size_average, reduce=reduce, reduction=reduction
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        return super().forward(input=input, target=target)


class RobustBCEWithLogitsLoss(BCEWithLogitsLoss, ABC):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        return super().forward(input=input, target=target)


class RobustCrossEntropyLoss(CrossEntropyLoss, ABC):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.long()
        return super().forward(input=input, target=target)
