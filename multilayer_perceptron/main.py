from data_without_normalization import DATA_WITHOUT_NORMALIZATION
from normalize_data import normalize, max_percentual_gordura, normalize_input
from neural_network import NeuralNetwork

def main():
    x, y = normalize(DATA_WITHOUT_NORMALIZATION)
    if len(x) == 0:
        return
    
    input_size = len(x[0])
    hidden_size = len(x)
    output_size = len(y[0])
    learning_rate = 0.1

    network = NeuralNetwork(x, y, input_size, hidden_size, output_size, learning_rate)
    network.train(5_000)


    entrada = [
        {"idade": 20, "sexo": "M", "altura_cm": 172, "peso_kg": 90, "nivel_atividade": "baixo", "tipo_corporal": "endomorfo"},
        {"idade": 30, "sexo": "M", "altura_cm": 172, "peso_kg": 120, "nivel_atividade": "baixo", "tipo_corporal": "endomorfo"},
        {"idade": 10, "sexo": "M", "altura_cm": 152, "peso_kg": 80, "nivel_atividade": "baixo", "tipo_corporal": "endomorfo"},
        {"idade": 10, "sexo": "M", "altura_cm": 152, "peso_kg": 40, "nivel_atividade": "baixo", "tipo_corporal": "endomorfo"},
    ]

    for x_new in entrada:
        y_pred = network.predict(normalize_input(x_new))
        print("Predição:", y_pred[0] * max_percentual_gordura)

if __name__ == "__main__":
    main()