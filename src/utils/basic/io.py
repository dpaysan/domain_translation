import os
from typing import List


def get_data_list(
    root_dir: str, absolute_path: bool = True, file_ending: bool = True
) -> List:

    assert os.path.exists(root_dir)
    list_of_data_locs = []
    for (root_dir, dirname, filename) in os.walk(root_dir):
        for file in filename:
            if not file_ending:
                file = file[: file.index(".")]
            if absolute_path:
                list_of_data_locs.append(os.path.join(root_dir, file))
            else:
                list_of_data_locs.append(file)
    return sorted(list_of_data_locs)
