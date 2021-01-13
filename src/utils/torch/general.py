import torch
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToPILImage,
    ToTensor,
)
from torch.nn import Module, L1Loss, MSELoss, Sigmoid, Softmax, ReLU


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
    # return torch.device("cpu")


def get_transformation_dict_for_train_val_test():
    train_transforms = Compose(
        [
            ToPILImage(),
            # Resize(64),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            ToPILImage(),
            # Resfrom src.utils.torch.general import get_deviceize(64),
            ToTensor(),
        ]
    )
    test_transforms = val_transforms
    transformation_dict = {
        "train": train_transforms,
        "val": val_transforms,
        "test": test_transforms,
    }
    return transformation_dict


def get_latent_distance_loss(loss_type: str = "mae") -> Module:
    if loss_type == "mae":
        latent_distance_loss = L1Loss()
    elif loss_type == "mse":
        latent_distance_loss = MSELoss()
    else:
        raise RuntimeError(
            "Unknown loss type given: {}, expected mse or mae.".format(loss_type)
        )
    return latent_distance_loss


def get_activation_module(output_activation: str = "sigmoid"):
    if output_activation == "sigmoid":
        return Sigmoid()
    elif output_activation == "softmax":
        return Softmax()
    elif output_activation == "relu":
        return ReLU()
    else:
        raise RuntimeError("Unknown activation function {}".format(output_activation))
