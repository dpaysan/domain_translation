import pandas as pd


def dict_to_csv(data: dict, save_path: str):
    data = pd.DataFrame.from_dict(data=data)
    data.to_csv(save_path)
