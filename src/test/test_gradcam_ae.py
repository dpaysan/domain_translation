import pickle
import numpy as np
import cv2
import torch
from torch import nn
from skimage import io

from src.models.ae import VanillaConvAE
from src.models.gradcam import GradCam, GuidedBackpropReLUModel
from torchvision.transforms import Compose, ToTensor, ToPILImage

from src.utils.torch.general import get_device

base_path = "/home/daniel/"

ae_model_weights = torch.load(base_path+"Downloads/best_model.pth")
ae_model = VanillaConvAE()
ae_model.load_state_dict(ae_model_weights)

class FullEncoder(nn.Module):
    def __init__(self, feature_extractor : nn.Module, latent_mapper : nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.latent_mapper = latent_mapper

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(1,-1)
        return self.latent_mapper(x)

feature_extractor = ae_model.encoder
full_encoder = FullEncoder(feature_extractor, ae_model.latent_mapper)

image = io.imread("/home/daniel/PycharmProjects/domain_translation/data/cd4/nuclear_crops_all_experiments/128px/labeled_scaled_max_intensity_resized_images/11J1_CD4T_488Coro1A_555RPL10A_D001_05_nucleus_472_97_2_13.tif")
image = np.array(image, dtype=np.float32)
input_image = torch.from_numpy(image.copy()).unsqueeze(0)
#transform_pipeline = Compose([ToPILImage(), ToTensor()])
#input_image = transform_pipeline(input_image)


device = get_device()

feature_extractor.to(device)
full_encoder.to(device)
input_image = input_image.to(device).unsqueeze(0)

query_node = 115

grad_cam = GradCam(model=full_encoder, feature_module=full_encoder.feature_extractor, target_layer_names=["4"],
                   device=device)

grayscale_cam = grad_cam(input_image, query_node)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


grayscale_cam = grayscale_cam.cpu().data.numpy()
grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam = show_cam_on_image(image, grayscale_cam)

gb_model = GuidedBackpropReLUModel(model=full_encoder, device=device)
gb = gb_model(input_image, query_node)
gb = gb.transpose((1, 2, 0))

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

cv2.imwrite("cam115.jpg", cam)
cv2.imwrite('gb115.jpg', gb)
cv2.imwrite('cam_gb115.jpg', cam_gb)


