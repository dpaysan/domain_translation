import os
from typing import List, Tuple
from numpy import ndarray
import cv2
from skimage import io
import numpy as np

from src.utils.basic.io import get_data_list


def min_max_scale_images(images: List[ndarray]) -> List[ndarray]:
    scaled_images = []
    for image in images:
        min_x = image.min()
        max_x = image.max()
        image = (image - min_x) / (max_x - min_x)
        scaled_images.append(image)
    return scaled_images


def get_max_intensity_images(images: List[ndarray]) -> List[ndarray]:
    max_intensity_images = []
    for image in images:
        max_intensity_image = image.max(axis=0)
        max_intensity_images.append(max_intensity_image)
    return max_intensity_images


def resize_images(images: List[ndarray], size: Tuple[int, int] = (64, 64)):
    scaled_images = []
    for image in images:
        image = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_CUBIC)
        scaled_images.append(image)
    return scaled_images


def read_images_from_disk(image_dir: str) -> Tuple[List[str], List[ndarray]]:
    image_locs = sorted(get_data_list(image_dir, absolute_path=True))
    image_names = sorted(get_data_list(image_dir, absolute_path=False))
    images = []
    for image_loc in image_locs:
        images.append(np.float32(io.imread(image_loc)))
    return image_names, images


def save_images_to_disk(images: List[ndarray], image_names: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(image_names)):
        io.imsave(fname=os.path.join(save_dir, image_names[i]), arr=images[i])


def get_mean_and_std_of_images(images: List[ndarray]) -> dict:
    images = np.array(images)
    stats = {"mean": images.mean(axis=0), "std": images.std(axis=0)}
    print(stats)
    return stats


def run_and_visualize_preprocessing_pipelin(
    image_dir: str = "../../data/nuclear_crops_all_experiments/images/",
):
    image_names, images = read_images_from_disk(image_dir=image_dir)
    scaled_images = min_max_scale_images(images)
    save_images_to_disk(
        images=scaled_images,
        image_names=image_names,
        save_dir="../../data/nuclear_crops_all_experiments/scaled_images/",
    )
    max_intensity_images = get_max_intensity_images(images=scaled_images)
    save_images_to_disk(
        images=max_intensity_images,
        image_names=image_names,
        save_dir=(
            "../../data/nuclear_crops_all_experiments/scaled_max_intensity_images/"
        ),
    )
    resized_images = resize_images(max_intensity_images)
    save_images_to_disk(
        images=resized_images,
        image_names=image_names,
        save_dir="../../data/nuclear_crops_all_experiments/scaled_max_intensity_resized_images/",
    )
    # stats = get_mean_and_std_of_images(resized_images)
    # io.imsave(fname='mean_image.tif', arr=stats['mean'])


if __name__ == "__main__":
    run_and_visualize_preprocessing_pipelin()
