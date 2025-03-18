import torch
from typing import Tuple


def train_classifier(
    probs: torch.Tensor,
    lbls: torch.Tensor,
) -> Tuple[int, int, int, int]:
    # print(*[f"{k}: {v.shape}" for k, v in inputs.items()])

    lbls = lbls.argmax(dim=1)
    probs = probs.argmax(dim=1)

    boncorrect = 0
    morcorrect = 0
    bonincorrect = 0
    morincorrect = 0
    for lbl, prob in zip(lbls, probs):
        if lbl == 0.0:  # TODO: check labels bon = 0, mor = 1
            if prob == lbl:
                boncorrect += 1
            else:
                bonincorrect += 1
        else:
            if prob == lbl:
                morcorrect += 1
            else:
                morincorrect += 1

    return boncorrect, bonincorrect, morcorrect, morincorrect
