from normalize_data import get_normalized_data
from neural_network import NeuralNetwork

def main():
    print("Executing normalization for data...")
    x,y = get_normalized_data('train.csv')
    if len(x) == 0:
        return
    
    input_size = len(x[0])
    hidden_size = len(x)
    output_size = len(y[0])
    learning_rate = 0.1

    print("Training neural network...")
    network = NeuralNetwork(x, y, input_size, hidden_size, output_size, learning_rate)
    network.train(100)

    print('Weights in hidden input layer:')
    print(network.weights_input_hidden)

    print('=' * 50)

    print('Weights in hidden output layer: ')
    print(network.weights_hidden_output)

if __name__ == "__main__":
    main()