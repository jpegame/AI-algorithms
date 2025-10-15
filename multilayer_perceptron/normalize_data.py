"""Normalization specific to the example"""

from typing import List, Any, Dict, Tuple
from enums.index import norm_values

max_idade = 120
max_altura = 250
max_peso = 300
max_percentual_gordura = 50

def normalize(data: List[Dict[str, Any]]) -> Tuple[List[float],List[float]]:
    if len(data) == 0:
        return

    return [
        [
            row['idade'] / max_idade,
            norm_values["sexo"][row["sexo"]],
            norm_values["nivel_atividade"][row["nivel_atividade"]],
            norm_values["tipo_corporal"][row["tipo_corporal"]],
            row['altura_cm'] / max_altura,
            row['peso_kg'] / max_peso
        ] for row in data
    ], [[row["percentual_gordura"] / max_percentual_gordura] for row in data]

def normalize_input(data: Dict[str, Any]) -> List[float]:
    return [
        data['idade'] / max_idade,
        norm_values["sexo"][data["sexo"]],
        norm_values["nivel_atividade"][data["nivel_atividade"]],
        norm_values["tipo_corporal"][data["tipo_corporal"]],
        data['altura_cm'] / max_altura,
        data['peso_kg'] / max_peso
    ]