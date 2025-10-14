from data_without_normalization import DATA_WITHOUT_NORMALIZATION
from normalize_data import normalize
from weight import generate_random_weights

def main():
    normalized_info = normalize(DATA_WITHOUT_NORMALIZATION)
    if len(normalized_info) == 0:
        return
    
    input_size = len(normalized_info[0]['x'])
    hidden_size = len(normalized_info)
    output_size = len(normalized_info[0]['y'])
    print(initial_values(input_size, hidden_size, output_size))

def initial_values(input_size: int, hidden_size: int, output_size: int):
    weight_input_hidden_layer = generate_random_weights(input_size, hidden_size)
    weight_hidden_output_layer = generate_random_weights(hidden_size, output_size)

    bias_hidden = [0 for _ in range(hidden_size)]
    bias_output = [0 for _ in range(output_size)]

    return weight_input_hidden_layer, weight_hidden_output_layer, bias_hidden, bias_output

def train(data: list[dict], num_epochs: int, n: float):
    for _ in range(num_epochs):
        pass

if __name__ == "__main__":
    main()