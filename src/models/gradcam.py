""" inspired by https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py"""
from typing import Tuple, List

import cv2
import torch
from torch import nn
from torch.autograd import Function
from torch import Tensor


class FeatureExtractor:
    """ Class to extract activations and register gradients for intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feature_activation_maps = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                feature_activation_maps += [x]
        final_feature_map = x
        return final_feature_map, feature_activation_maps


class ModelExtractor:
    """ Class to extract network output, activation and gradients from targetted layers."""

    def __init__(
        self, model: nn.Module, feature_module: nn.Module, target_layers: List[str]
    ):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feature_activation_maps = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                x, feature_activation_maps = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = x.view(x.size(0), -1)
                x = module(x)
        final_model_output = x
        return final_model_output, feature_activation_maps


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, device):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        self.extractor = ModelExtractor(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, x, target_node):
        input_img = x.to(self.device)

        # Get target activations, and output from the full model
        final_model_output, feature_activation_maps = self.extractor(x)

        # Create one-hot-encoded vector to query GradCAM for the target node (given by its index)
        one_hot = torch.zeros((1, final_model_output.size()[-1]))
        one_hot[:, target_node] = 1
        one_hot = one_hot.to(self.device)

        # Nullify the activation of all nodes in the target layer except the queried node
        one_hot = torch.sum(one_hot * final_model_output)

        # Reset gradients
        self.feature_module.zero_grad()
        self.model.zero_grad()

        # Backpropagate gradients
        one_hot.backward(retain_graph=True)

        # Get the gradients for the weights of the final layer of the of the feature extractor model
        gradient = self.extractor.get_gradients()[-1].detach()

        # Get activation map of the final layer of the feature extractor e.g. the final convolutional layer
        target_feature_map = feature_activation_maps[-1][0]

        # Compute weights by global average pooling of the gradients
        weights = torch.mean(gradient, dim=(2, 3))[0, :]
        # Create placeholder for the class-activation-maps of the same size as the final feature map
        cam = torch.zeros(target_feature_map.shape[1:]).to(self.device)

        # Sum up the contributions of the individual channels of the final feature map of the extractor
        # weighted by their gradients
        for i, w in enumerate(weights):
            cam += w * target_feature_map[i, :, :]

        # Compute the maximum
        cam = torch.max(cam, torch.zeros_like(cam))
        cam = torch.from_numpy(cv2.resize(cam.cpu().data.numpy(), input_img.shape[2:]))
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        # output = torch.mul(x, positive_mask)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        # Mask gradient where both input and output are non-zero
        # positive_mask_input = (input_img > 0).type_as(grad_output)
        # positive_mask_gradient = (grad_output > 0).type_as(grad_output)
        # grad_input = torch.mul(torch.mul(grad_output, positive_mask_input), positive_mask_gradient)
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device
        self.model = model.to(device)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # recursively replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_node):
        input_img = input_img.to(self.device)
        input_img = input_img.requires_grad_(True)

        final_model_output = self.forward(input_img)

        # Create one-hot-encoded vector to query GradCAM for the target node (given by its index)
        one_hot = torch.zeros((1, final_model_output.size()[-1]))
        one_hot[:, target_node] = 1
        one_hot.requires_grad_(True)
        one_hot = one_hot.to(self.device)

        # Nullify the activation of all nodes in the target layer except the queried node
        one_hot = torch.sum(one_hot * final_model_output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
