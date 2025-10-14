"""Normalization specific to the example"""

from typing import List, Any, Dict
from enums.index import norm_values


def normalize(data: List[Dict[str, Any]]) -> List[float]:
    if len(data) == 0:
        return
    
    max_idade = 120
    max_altura = 250
    max_peso = 300
    max_percentual_gordura = 50

    return [
        { 
            "x": [
                row['idade'] / max_idade,
                norm_values["sexo"][row["sexo"]],
                norm_values["nivel_atividade"][row["nivel_atividade"]],
                norm_values["tipo_corporal"][row["tipo_corporal"]],
                row['altura_cm'] / max_altura,
                row['peso_kg'] / max_peso
            ],
            "y": [
                row["percentual_gordura"] / max_percentual_gordura
            ]
        } for row in data
    ]