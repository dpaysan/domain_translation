from torch import Tensor


def accuracy(outputs:Tensor, labels:Tensor)->float:
    n_samples = outputs.size(0).item()
    preds = outputs.argmax(dim=1).view(-1)
    correct = preds.eq(labels.view(-1)).float().sum().item()
    return correct/n_samples