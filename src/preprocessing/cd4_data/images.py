import os
from shutil import copyfile
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray
from skimage import io

from src.utils.basic.io import get_file_list


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
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
        image = np.clip(image, 0, 1)
        scaled_images.append(image)
    return scaled_images


def read_images_from_disk(
    image_dir: str, image_locs: List = None
) -> Tuple[List[str], List[ndarray]]:
    if image_locs is None:
        image_locs = sorted(get_file_list(image_dir, absolute_path=True))
    image_names = sorted(get_file_list(image_dir, absolute_path=False))
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


def get_only_images_with_labels(
    image_dir: str, label_fname: str, id_column: str = "nucleus_id"
):
    image_ids = get_file_list(
        root_dir=image_dir, absolute_path=False, file_ending=False
    )
    image_locs = get_file_list(root_dir=image_dir)
    image_df = pd.DataFrame.from_dict({id_column: image_ids, "image_loc": image_locs})
    label_ids = pd.read_csv(label_fname, index_col=0)
    image_df = image_df.merge(label_ids, on=id_column)
    _, images = read_images_from_disk(
        image_dir=image_dir, image_locs=list(image_df.loc[:, "image_loc"])
    )
    image_names = [
        image_name + ".tif" for image_name in list(image_df.loc[:, id_column])
    ]
    return image_names, images


def run_and_visualize_preprocessing_pipeline(
    image_dir: str = "../../../data/cd4/nuclear_crops_all_experiments/images/",
):
    image_names, images = read_images_from_disk(image_dir=image_dir)
    scaled_images = min_max_scale_images(images)
    save_images_to_disk(
        images=scaled_images,
        image_names=image_names,
        save_dir="../../../data/cd4/nuclear_crops_all_experiments/scaled_images/",
    )
    max_intensity_images = get_max_intensity_images(images=scaled_images)
    save_images_to_disk(
        images=max_intensity_images,
        image_names=image_names,
        save_dir="../../../data/cd4/nuclear_crops_all_experiments/scaled_max_intensity_images/",
    )
    resized_images = resize_images(max_intensity_images, size=(128,128))
    save_images_to_disk(
        images=resized_images,
        image_names=image_names,
        save_dir="../../../data/cd4/nuclear_crops_all_experiments/scaled_max_intensity_resized_images/",
    )

    image_names, labeled_images = get_only_images_with_labels(
        image_dir="../../../data/cd4/nuclear_crops_all_experiments/scaled_max_intensity_resized_images/",
        label_fname=(
            "../../../data/cd4/nuclear_crops_all_experiments/simple_image_labels.csv"
        ),
        id_column="nucleus_id",
    )
    save_images_to_disk(
        images=labeled_images,
        image_names=image_names,
        save_dir="../../../data/cd4/nuclear_crops_all_experiments/labeled_scaled_max_intensity_resized_images/",
    )


def copy_labeled_images(
    image_dir: str = "../../../data/cd4/nuclear_crops_all_experiments/images/",
    label_fname: str = "../../../data/cd4/nuclear_crops_all_experiments/simple_image_labels.csv",
    id_column: str = "nucleus_id",
    output_dir: str = "../../../data/cd4/nuclear_crops_all_experiments/labeled_images/",
):
    image_ids = get_file_list(
        root_dir=image_dir, absolute_path=False, file_ending=False
    )
    image_locs = get_file_list(root_dir=image_dir)
    image_df = pd.DataFrame.from_dict({id_column: image_ids, "image_loc": image_locs})
    label_ids = pd.read_csv(label_fname, index_col=0)
    image_df = image_df.merge(label_ids, on=id_column)
    image_names = [
        image_name + ".tif" for image_name in list(image_df.loc[:, id_column])
    ]
    os.makedirs(output_dir, exist_ok=True)
    for image_name in image_names:
        copyfile(image_dir + image_name, output_dir + image_name)

    # stats = get_mean_and_std_of_images(resized_images)
    # io.imsave(fname='mean_image.tif', arr=stats['mean'])


if __name__ == "__main__":
    # copy_labeled_images()
    run_and_visualize_preprocessing_pipeline()
