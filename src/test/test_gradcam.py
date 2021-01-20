import cv2
from src.models.gradcam import GuidedBackpropReLUModel, GradCam
from torchvision.models import resnet50
from torchvision import transforms
import numpy as np
import torch

from src.utils.torch.general import get_device

image_path = "/home/daniel/Downloads/cat_dog.jpg"


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite("h.jpg", heatmap)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


device = get_device()
model = resnet50(pretrained=True).to(device)
grad_cam = GradCam(model=model, feature_module=model.layer4, \
                   target_layer_names=["2"], device=device)

img = cv2.imread("/home/daniel/Downloads/epoch_426_inputs_0.jpg", 1)
img = cv2.resize(img, (640,480))
img = np.float32(img) / 255
#img = cv2.resize(img, (224, 224))
# Opencv loads as BGR:
#img = img[:, :, ::-1]
#img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
input_img = preprocess_image(img)
#input_img = torch.from_numpy(img).view(1,3,224,224)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = 1
grayscale_cam = grad_cam(input_img.to(device), target_category)

grayscale_cam = grayscale_cam.cpu().data.numpy()
grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
cam = show_cam_on_image(img, grayscale_cam)

gb_model = GuidedBackpropReLUModel(model=model, device=device)
gb = gb_model(input_img, target_category)
gb = gb.transpose((1, 2, 0))

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

cv2.imwrite("cam.jpg", cam)
cv2.imwrite('gb.jpg', gb)
cv2.imwrite('cam_gb.jpg', cam_gb)
cv2.imwrite("gs_cam.jpg", grayscale_cam)
