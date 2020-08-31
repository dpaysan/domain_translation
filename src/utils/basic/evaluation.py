import os
from typing import Tuple

from pandas import DataFrame

from src.utils.basic.io import get_file_list


class PairedDomainLogAnalyzer(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.knn_acc_dicts = []
        self.reconstruction_error_dict = {}
        self.latent_distances = []
        self.experiment_type = None

        self.analyze_log_file()

    def analyze_log_file(self):
        with open(self.log_file) as f:
            lines = f.readlines()
            test_metrics = False
            for line in lines:
                line = line.lower()
                # Filter for test statistics
                if "test loss statistics" in line:
                    test_metrics = True
                    self.knn_acc_dicts.append({})
                elif "train" in line or "val" in line:
                    test_metrics = False

                if test_metrics:
                    if "reconstruction loss" in line:
                        words = line.split()
                        domain = words[words.index("domain:") - 1]
                        score = float(words[-1])
                        if domain not in self.reconstruction_error_dict:
                            self.reconstruction_error_dict[domain] = []
                        self.reconstruction_error_dict[domain].append(score)
                    if "latent l1 distance" in line:
                        score = float(line.split()[-1])
                        self.latent_distances.append(score)
                    if "-nn accuracy" in line.lower():
                        idx = line.index("-nn accuracy")
                        k = int(line[idx - 2 : idx])
                        score = float(line.split()[-1])
                        self.knn_acc_dicts[-1][k] = score
        if len(self.latent_distances) > 1:
            self.experiment_type = "cv"
        else:
            self.experiment_type = "train_val_test"


def evaluate_partly_integrated_latent_space_paired_data_experiments(
    experiments_root_dir: str, log_file_type: str = ".log"
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    log_files = get_file_list(
        root_dir=experiments_root_dir,
        absolute_path=True,
        file_ending=True,
        file_type_filter=log_file_type,
    )
    knn_results = DataFrame(
        columns=["experiment_id", "shared_ratio", "supervision", "k", "knn_accuracy"]
    )

    reconstruction_results = DataFrame(
        columns=[
            "experiment_id",
            "shared_ratio",
            "supervision",
            "domain",
            "reconstruction_loss",
        ]
    )

    latent_distance_results = DataFrame(
        columns=["experiment_id", "shared_ratio", "supervision", "latent_distance"]
    )

    for log_file in log_files:
        configuration = os.path.split(os.path.split(log_file)[0])[1]
        idx = configuration.index("_")
        shared_ratio = int(configuration[:idx])
        supervision = int(configuration[idx + 1 :])

        analyzer = PairedDomainLogAnalyzer(log_file=log_file)

        for latent_distance in analyzer.latent_distances:
            latent_distance_results = latent_distance_results.append(
                {
                    "experiment_id": configuration,
                    "shared_ratio": shared_ratio,
                    "supervision": supervision,
                    "latent_distance": latent_distance,
                },
                ignore_index=True,
            )

        for domain, scores in analyzer.reconstruction_error_dict.items():
            for score in scores:
                reconstruction_results = reconstruction_results.append(
                    {
                        "experiment_id": configuration,
                        "shared_ratio": shared_ratio,
                        "supervision": supervision,
                        "domain": domain,
                        "reconstruction_loss": score,
                    },
                    ignore_index=True,
                )

        for knn_dict in analyzer.knn_acc_dicts:
            for k, score in knn_dict.items():
                knn_results = knn_results.append(
                    {
                        "experiment_id": configuration,
                        "shared_ratio": shared_ratio,
                        "supervision": supervision,
                        "k": k,
                        "knn_accuracy": score,
                    },
                    ignore_index=True,
                )
    return reconstruction_results, latent_distance_results, knn_results
