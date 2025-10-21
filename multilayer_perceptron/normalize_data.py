import pandas as pd
from typing import List, Dict, Tuple, Hashable
from enums.index import norm_values

def get_normalized_data(path: str):
    df = pd.read_csv(f'./model_data/{path}')
    data = df.to_dict(orient='records')
    maximum_data = get_max_value_data(data)
    

    x_matrix = []
    y_matrix = []
    for info in data:
        keys = list(info.keys())

        x_list = []
        for key in keys:
            if key == keys[-1]:
                y_matrix.append([info[key] / 3])
                continue

            x_list.append(info[key] / maximum_data[key])
        x_matrix.append(x_list)

    return x_matrix, y_matrix

def get_max_value_data(data: List[Dict[Hashable, float]]):
    if len(data) == 0:
        raise ValueError("get_max_value_data needs data to extract its maximum values therefore being able to normalize it!")

    keys = list(data[0].keys())
    maximum_values = {}
    for key in keys:
        max_value = max(d[key] for d in data)
        if max_value > 4:
            max_value *= 1.2
        maximum_values[key] = max_value

    return maximum_values

