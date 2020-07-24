from src.data.datasets import TorchNucleiImageDataset, TorchSeqDataset
from torchvision.transforms import Compose


def test_create_img_dataset(
    image_dir: str, image_label_fname: str, transform_pipeline: Compose = None
) -> bool:
    try:
        img_dataset = TorchNucleiImageDataset(
            image_dir=image_dir,
            label_fname=image_label_fname,
            transform_pipeline=transform_pipeline,
        )
        print("Image data set loaded of length", len(img_dataset))
        print("Last sample is ", img_dataset[len(img_dataset) - 1])
    except Exception as e:
        print(e)
        return False
    return True


def test_create_seq_dataset(
    seq_data_and_labels_fname: str, transform_pipeline: Compose = None
):
    try:
        seq_dataset = TorchSeqDataset(
            seq_data_and_labels_fname=seq_data_and_labels_fname,
            transform_pipeline=transform_pipeline,
        )
        print("Image data set loaded of length", len(seq_dataset))
        print("Last sample is ", seq_dataset[len(seq_dataset) - 1])
    except Exception as e:
        print(e)
        return False
    return True


def run_all_tests(
    image_dir: str,
    image_label_fname: str,
    seq_data_and_labels_fname: str,
    image_transform_pipeline: Compose = None,
    seq_data_transform_pipeline: Compose = None,
):
    successful_tests = 0
    total_tests = 2
    successful_tests += test_create_img_dataset(
        image_dir=image_dir,
        image_label_fname=image_label_fname,
        transform_pipeline=image_transform_pipeline,
    )
    successful_tests += test_create_seq_dataset(
        seq_data_and_labels_fname=seq_data_and_labels_fname,
        transform_pipeline=seq_data_transform_pipeline,
    )
    print("Test successful completed: {} of {}".format(successful_tests, total_tests))


if __name__ == "__main__":
    image_dir = (
        "../../data/nuclear_crops_all_experiments/labeled_scaled_max_intensity_resized_images/"
    )
    image_label_fname = (
        "../../data/nuclear_crops_all_experiments/simple_image_labels.csv"
    )

    rna_seq_data_and_label_fname = "../../data/cda_rna_seq/rna_seq_data_and_labels.csv"

    run_all_tests(
        image_dir=image_dir,
        image_label_fname=image_label_fname,
        seq_data_and_labels_fname=rna_seq_data_and_label_fname,
    )
