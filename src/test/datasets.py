from src.data.datasets import TorchCellImageDataSet


def test_create_img_dataset(root_dir):
    img_dataset = TorchCellImageDataSet(data_dir=root_dir)
    print("Data set loaded of length", len(img_dataset))
    print("Last sample is ", img_dataset[len(img_dataset) - 1])
    return True


def run_all_tests(root_dir):
    successful_tests = 0
    total_tests = 1
    successful_tests += test_create_img_dataset(root_dir)
    print("Test successful completed: {} of {}".format(successful_tests, total_tests))


if __name__ == "__main__":
    root_dir = "../../data/final_cancer_data/"
    run_all_tests(root_dir)
