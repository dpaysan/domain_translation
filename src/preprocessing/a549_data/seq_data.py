import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler



def get_combined_data(
    count_data_floc: str = "../../../data/a549_dex/raw/rna_only_gene_count.txt",
    cell_data_floc: str = "../../../data/a549_dex/raw/rna_only_a549_cell.txt",
    structure_data_floc: str = "../../../data/a549_dex/raw/rna_only_a549_gene.txt",
    sample_column: str = "sample",
    structure_column: str = "gene_id",
    filter_cell_type: bool = True,
    add_treatment_time_label:bool=True
) -> DataFrame:
    sparse_gene_counts = mmread(count_data_floc)
    structure_counts = sparse_gene_counts.todense().transpose()

    cell_data = pd.read_csv(cell_data_floc)
    structure_data = pd.read_csv(structure_data_floc)

    if filter_cell_type:
        cell_data = cell_data.loc[cell_data["cell_name"] == "A549"]
        cell_type_matches = cell_data.index
        structure_counts = structure_counts[list(cell_type_matches)]

    structure_df = DataFrame(
        data=structure_counts,
        index=list(cell_data[sample_column]),
        columns=list(structure_data[structure_column]),
    )

    if add_treatment_time_label:
        try:
            structure_df['label'] = cell_data['treatment_time']
        except KeyError:
            pass

    return structure_df


def filter_data(
    structure_df: DataFrame, columns_to_keep: List = None, rows_to_keep: List = None
) -> DataFrame:
    filtered_df = structure_df.copy()
    if columns_to_keep is not None:
        filtered_df = filtered_df[columns_to_keep]
    if rows_to_keep is not None:
        filtered_df = filtered_df.loc[rows_to_keep]
    return filtered_df


def transform_count_data(count_df: DataFrame, mode: str = "logx1") -> DataFrame:
    transformed_df = count_df.copy()
    if mode == "logx1":
        transformed_df.transform(lambda x: np.log2(x + 1))
    elif mode == "standardize":
        sc = StandardScaler()
        transformed_df = sc.fit_transform(transformed_df)
    else:
        raise NotImplemented("Unknown mode given: {}.".format(mode))
    return transformed_df


def get_differently_expressed_structures(
    rna_data: DataFrame,
    de_analysis_floc: str = "../../../data/a549_dex/analyses/de_genes.csv",
    structure_column: str = "gene_id",
) -> DataFrame:
    de_data = pd.read_csv(de_analysis_floc)
    de_data = de_data.loc[de_data["qval"] < 0.05, structure_column]
    differently_expressed_structures = list(de_data)
    rna_data = rna_data[differently_expressed_structures]
    return rna_data.copy()


def get_paired_data_only(
    structure_df1: DataFrame, structure_df2: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    index1 = structure_df1.index
    index2 = structure_df2.index

    intersection_index = pd.Index.intersection(index1, index2)
    intersected_df1 = structure_df1.copy()
    intersected_df2 = structure_df2.copy()
    intersected_df1 = intersected_df1.loc[intersection_index]
    intersected_df2 = intersected_df2.loc[intersection_index]

    return intersected_df1, intersected_df2


def save_data_to_disk(dataframe: DataFrame, save_floc: str):
    os.makedirs(os.path.split(save_floc)[0], exist_ok=True)
    dataframe.to_csv(save_floc)


def get_transcription_factor_motif_data(
    tf_motif_data_floc: str = "../../../data/a549_dex/processed/atac_tf_motifs.csv",
) -> DataFrame:
    tf_motif_data = pd.read_csv(tf_motif_data_floc, index_col=0)
    return tf_motif_data

def add_treatment_time_label_to_atac_data(atac_data:DataFrame, rna_data:DataFrame)->Tuple[DataFrame, DataFrame]:
    labeled_atac_data = atac_data.copy().sort_index()
    labeled_rna_data = rna_data.copy().sort_index()
    assert len(atac_data) != len(rna_data)
    labeled_atac_data['label'] = labeled_rna_data['label']
    return labeled_atac_data, labeled_rna_data



def run_rna_atac_tf_pipeline(
    rna_count_data_floc: str = "../../../data/a549_dex/raw/rna_gene_count.txt",
    rna_cell_data_floc="../../../data/a549_dex/raw/rna_cell.txt",
    rna_structure_data_floc: str = "../../../data/a549_dex/raw/rna_gene.txt",
    rna_sample_column: str = "sample",
    rna_structure_column: str = "gene_id",
    atac_tf_motif_data_floc: str = '../../../data/a549_dex/processed/atac_tf_motifs.csv',
    atac_count_data_floc: str = "../../../data/a549_dex/raw/atac_peak_count.txt",
    atac_cell_data_floc: str = "../../../data/a549_dex/raw/atac_cell.txt",
    atac_structure_data_floc: str = "../../../data/a549_dex/raw/atac_peak.txt",
    atac_sample_column: str = "sample",
    atac_structure_column: str = "peak",
    rna_de_analysis_floc: str = "../../../data/a549_dex/analyses/de_genes.csv",
    processed_rna_floc: str = "../../../data/a549_dex/processed/rna_data.csv",
    processed_atac_floc: str = "../../../data/a549_dex/processed/atac_data.csv",
):

    rna_data = get_combined_data(
        count_data_floc=rna_count_data_floc,
        cell_data_floc=rna_cell_data_floc,
        structure_data_floc=rna_structure_data_floc,
        sample_column=rna_sample_column,
        structure_column=rna_structure_column,
        filter_cell_type=True,
        add_treatment_time_label=True,
    )

    # atac_data = get_combined_data(
    #     count_data_floc=atac_count_data_floc,
    #     cell_data_floc=atac_cell_data_floc,
    #     structure_data_floc=atac_structure_data_floc,
    #     sample_column=atac_sample_column,
    #     structure_column=atac_structure_column,
    #     filter_cell_type=False,
    # )

    atac_data = get_transcription_factor_motif_data(tf_motif_data_floc=atac_tf_motif_data_floc).transpose()

    rna_data, atac_data = get_paired_data_only(rna_data, atac_data)

    rna_data = get_differently_expressed_structures(
        rna_data=rna_data,
        de_analysis_floc=rna_de_analysis_floc,
        structure_column=rna_structure_column,
    )
    rna_data = transform_count_data(count_df=rna_data, mode="logx1")
    scaled_rna_data = transform_count_data(count_df=rna_data, mode="standardize")
    rna_data = DataFrame(
        data=scaled_rna_data, index=rna_data.index, columns=rna_data.columns
    )

    atac_data = transform_count_data(count_df=atac_data, mode="logx1")
    scaled_atac_data = transform_count_data(count_df=atac_data, mode="standardize")
    atac_data = DataFrame(
        data=scaled_atac_data, index=atac_data.index, columns=atac_data.columns
    )

    save_data_to_disk(dataframe=rna_data, save_floc=processed_rna_floc)
    save_data_to_disk(dataframe=atac_data, save_floc=processed_atac_floc)


if __name__ == "__main__":
    run_rna_atac_tf_pipeline()
