import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler


class SeqDataPreprocessor(object):
    def __init__(
        self,
        count_data_floc: str = None,
        cell_data_floc: str = None,
        structure_data_floc: str = None,
        sample_column: str = None,
        structure_column: str = None,
        label_name: str = None,
    ):
        self.count_data_floc = count_data_floc
        self.cell_data_floc = cell_data_floc
        self.structure_data_floc = structure_data_floc
        self.sample_column = sample_column
        self.structure_column = structure_column
        self.label_name = label_name

        self.data = None
        self.labels = None

    def init_data(self, cell_filter: Tuple = ("cell_name", "A549")):
        sparse_gene_counts = mmread(self.count_data_floc)
        structure_counts = sparse_gene_counts.todense().transpose()

        cell_data = pd.read_csv(self.cell_data_floc)
        structure_data = pd.read_csv(self.structure_data_floc)

        if cell_filter is not None:
            cell_data = cell_data.loc[cell_data[cell_filter[0]] == cell_filter[1]]
            cell_type_matches = cell_data.index
            structure_counts = structure_counts[list(cell_type_matches)]

        self.data = DataFrame(
            data=structure_counts,
            index=list(cell_data[self.sample_column]),
            columns=list(structure_data[self.structure_column]),
        )

        if self.label_name in cell_data:
            cell_data = cell_data.set_index(self.sample_column)
            # Replace 3 hour label with 2
            self.labels = cell_data[self.label_name].replace(3, 2)
            # self.labels = self.labels.set_index(cell_data.index)

    def set_data(self, data):
        self.data = data
        if self.labels is not None:
            self.labels = self.labels.loc[self.data.index]

    def filter_data(self, columns_to_keep: List = None, rows_to_keep: List = None):
        if columns_to_keep is not None:
            self.data = self.data[columns_to_keep]
        if rows_to_keep is not None:
            self.data = self.data.loc[rows_to_keep]
            if self.labels is not None:
                self.labels = self.labels[rows_to_keep]

    def transform_count_data(self, mode: str = "logx1"):
        if mode == "logx1":
            self.data = self.data.transform(lambda x: np.log(x + 1))
        elif mode == "standardize":
            index = self.data.index
            cols = self.data.columns
            sc = StandardScaler()
            self.data = sc.fit_transform(self.data)
            self.data = DataFrame(data=self.data, index=index, columns=cols)
        else:
            raise NotImplemented("Unknown mode given: {}.".format(mode))

    def save_data_and_labels_to_disk(self, save_floc, label_column: str = "label"):
        data = self.data
        if self.labels is not None:
            data[label_column] = self.labels
        os.makedirs(os.path.split(save_floc)[0], exist_ok=True)
        data.to_csv(save_floc)


class RnaDataPreprocessor(SeqDataPreprocessor):
    def __init__(
        self,
        count_data_floc: str = None,
        cell_data_floc: str = None,
        structure_data_floc: str = None,
        sample_column: str = None,
        structure_column: str = None,
        label_name: str = "treatment_time",
        de_analysis_floc: str = None,
        qval_column: str = "qval",
        threshold: float = 0.05,
    ):
        super().__init__(
            count_data_floc=count_data_floc,
            cell_data_floc=cell_data_floc,
            structure_data_floc=structure_data_floc,
            sample_column=sample_column,
            structure_column=structure_column,
            label_name=label_name,
        )
        self.de_analysis_floc = de_analysis_floc
        self.qval_column = qval_column
        self.threshold = threshold
        self.de_data = None

    def get_differently_expressed_structures(self,):
        self.de_data = pd.read_csv(self.de_analysis_floc)
        self.de_data = self.de_data.loc[
            self.de_data[self.qval_column] < self.threshold, self.structure_column
        ]
        self.data = self.data[list(self.de_data)]


class PairedSeqPreprocessor(object):
    def __init__(
        self,
        seq_data_processor_1: SeqDataPreprocessor,
        seq_data_processor_2: SeqDataPreprocessor,
    ):
        self.processor_1 = seq_data_processor_1
        self.processor_2 = seq_data_processor_2

    def select_paired_data_only(self):
        data_1 = self.processor_1.data
        data_2 = self.processor_2.data
        index1 = data_1.index
        index2 = data_2.index

        intersection_index = pd.Index.intersection(index1, index2)
        data_1 = data_1.loc[intersection_index]
        data_2 = data_2.loc[intersection_index]

        self.processor_1.set_data(data_1)
        self.processor_2.set_data(data_2)

    def add_labels_to_paired_data(self):
        if self.processor_1.labels is not None:
            self.processor_2.labels = self.processor_1.labels
        elif self.processor_2.labels is not None:
            self.processor_1.labels = self.processor_2.labels
        else:
            raise RuntimeError("Only unlabeled data is given.")


def run_rna_atac_tf_pipeline(
    rna_count_data_floc: str = "../../../data/a549_dex/raw/rna_gene_count.txt",
    rna_cell_data_floc="../../../data/a549_dex/raw/rna_cell.txt",
    rna_structure_data_floc: str = "../../../data/a549_dex/raw/rna_gene.txt",
    rna_sample_column: str = "sample",
    rna_structure_column: str = "gene_id",
    atac_tf_motif_data_floc: str = "../../../data/a549_dex/processed/atac_tf_motifs.csv",
    rna_de_analysis_floc: str = "../../../data/a549_dex/analyses/de_genes.csv",
    processed_rna_floc: str = "../../../data/a549_dex/processed/rna_data.csv",
    processed_atac_floc: str = "../../../data/a549_dex/processed/atac_data.csv",
):

    rna_preprocessor = RnaDataPreprocessor(
        count_data_floc=rna_count_data_floc,
        cell_data_floc=rna_cell_data_floc,
        structure_data_floc=rna_structure_data_floc,
        sample_column=rna_sample_column,
        structure_column=rna_structure_column,
        de_analysis_floc=rna_de_analysis_floc,
    )

    rna_preprocessor.init_data()
    rna_preprocessor.get_differently_expressed_structures()

    atac_tf_motif_data = pd.read_csv(atac_tf_motif_data_floc, index_col=0).transpose()

    atac_preprocessor = SeqDataPreprocessor()
    atac_preprocessor.set_data(atac_tf_motif_data)

    paired_processor = PairedSeqPreprocessor(
        seq_data_processor_1=rna_preprocessor, seq_data_processor_2=atac_preprocessor
    )

    paired_processor.select_paired_data_only()

    paired_processor.processor_1.transform_count_data(mode="logx1")
    paired_processor.processor_1.transform_count_data(mode="standardize")

    paired_processor.processor_2.transform_count_data(mode="logx1")
    paired_processor.processor_2.transform_count_data(mode="standardize")

    paired_processor.add_labels_to_paired_data()

    paired_processor.processor_1.save_data_and_labels_to_disk(processed_rna_floc)
    paired_processor.processor_2.save_data_and_labels_to_disk(processed_atac_floc)


if __name__ == "__main__":
    run_rna_atac_tf_pipeline()
