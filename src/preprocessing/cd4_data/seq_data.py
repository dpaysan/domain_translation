from typing import Tuple

import pandas as pd
from pandas import DataFrame

from src.utils.basic.general import get_class_weights


def merge_seq_data_and_labels(seq_data: DataFrame, seq_labels: DataFrame) -> DataFrame:
    seq_data = seq_data.transpose()

    merged_data = seq_data.join(seq_labels, how="right")
    merged_data = merged_data.rename(columns={"label": "binary_label"})
    return merged_data


def read_seq_data_and_labels(
        data_fname: str,
        label_fname: str,
        label_codes=None,
        label_column_name: str = "label",
) -> Tuple[DataFrame, DataFrame]:
    if label_codes is None:
        label_codes = {1: 0, 0: 1}
    seq_data = pd.read_csv(data_fname, index_col=0)
    seq_labels = pd.read_csv(label_fname, index_col=0)
    seq_labels.loc[:, label_column_name] = seq_labels[label_column_name].map(
        label_codes
    )
    return seq_data, seq_labels


def save_seq_data_and_labels(seq_data: DataFrame, seq_data_save_fname: str) -> None:
    seq_data.to_csv(seq_data_save_fname)


def get_seq_label_class_weights(
        simple_labels_fname: str = "../../../data/cd4/cda_rna_seq/rna_seq_data_and_labels.csv",
) -> dict:
    return get_class_weights(simple_labels_fname=simple_labels_fname)


def run_seq_8k_pbmc_data_pipeline(
        data_fname: str = "../../../data/cd4/cda_rna_seq/filtered_lognuminorm_pc_rp_7633genes_1396cellsnCD4.csv",
        label_fname: str = "../../../data/cd4/cda_rna_seq/labels_nCD4_corrected.csv",
        seq_data_save_fname: str = "../../../data/cd4/cda_rna_seq/rna_seq_data_and_labels.csv",
) -> None:
    seq_data, seq_labels = read_seq_data_and_labels(
        data_fname=data_fname, label_fname=label_fname
    )
    merged_seq_data = merge_seq_data_and_labels(
        seq_data=seq_data, seq_labels=seq_labels
    )
    save_seq_data_and_labels(
        seq_data=merged_seq_data, seq_data_save_fname=seq_data_save_fname
    )


def run_seq_10k_pbmc_data_pipeline_genesets(
        data_fname: str = "../../../data/cd4/cd4_rna_seq_pbmc_10k/gene_kegg_pathway_filtered_nCD4.csv",
        label_fname: str = "../../../data/cd4/cd4_rna_seq_pbmc_10k/nCD4_labels_0quiescent_1poised.csv",
        seq_data_save_fname="../../../data/cd4/cd4_rna_seq_pbmc_10k/gene_kegg_pathway_data_and_labels.csv"):
    seq_data = pd.read_csv(data_fname, index_col=0)
    seq_labels = pd.read_csv(label_fname, index_col=0)
    merged_seq_data_and_labels = merge_seq_data_and_labels(seq_data, seq_labels)
    merged_seq_data_and_labels.to_csv(seq_data_save_fname)


if __name__ == "__main__":
    # run_seq_8k_pbmc_data_pipeline()
    # get_seq_label_class_weights()
    run_seq_10k_pbmc_data_pipeline_genesets()
