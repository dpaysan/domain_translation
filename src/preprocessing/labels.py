import pandas as pd


def get_compressed_image_labels(
    input_labels_fname: str = "../../data/nuclear_crops_all_experiments/protein_ratios_full.csv",
    output_labels_fname: str = "../../data/nuclear_crops_all_experiments/simple_image_labels.csv",
    data_name_column: str = "Label",
    cluster_column_name="mycl",
    label_codes=None,
):
    if label_codes is None:
        label_codes = {1: 1, 2: 0}
    df_labels = pd.read_csv(input_labels_fname, index_col=0)
    df_labels.loc[:, cluster_column_name] = df_labels[cluster_column_name].map(
        label_codes
    )
    df_labels = df_labels.rename(
        columns={data_name_column: "nucleus_id", cluster_column_name: "binary_label"}
    )
    df_labels = df_labels.drop(
        columns=[
            c for c in df_labels.columns if c not in ["nucleus_id", "binary_label"]
        ]
    )
    df_labels.to_csv(output_labels_fname)
    return


if __name__ == "__main__":
    get_compressed_image_labels()
