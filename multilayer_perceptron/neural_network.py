from weight import generate_random_weights
from utils.index import dot, add, sigmoid, sigmoid_derivative, transpose, multiply_elementwise, subtract, sum_rows, mean_squared_error

class NeuralNetwork:
    def __init__(self, x, y, input_size, hidden_size, output_size, learning_rate):
        self.x: list[list[float]] = x
        self.y: list[list[float]] = y
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.learning_rate: float = learning_rate

        self.weights_input_hidden: list[list[float]] = generate_random_weights(input_size, hidden_size)
        self.weights_hidden_output: list[list[float]] = generate_random_weights(hidden_size, output_size)

        self.bias_hidden = [0 for _ in range(hidden_size)]
        self.bias_output = [0 for _ in range(output_size)]

    def feed_forward(self):
        self.hidden_activation = add(dot(self.x, self.weights_input_hidden), self.bias_hidden)
        self.hidden_output = sigmoid(self.hidden_activation)

        self.output_activation = add(dot(self.hidden_output, self.weights_hidden_output), self.bias_output)
        self.predicted_output = sigmoid(self.output_activation)
        return self.predicted_output

    def backward(self):
        output_error = subtract(self.y, self.predicted_output)
        output_delta = multiply_elementwise(output_error, sigmoid_derivative(self.predicted_output))

        hidden_error = dot(output_delta, transpose(self.weights_hidden_output))
        hidden_delta = multiply_elementwise(hidden_error, sigmoid_derivative(self.hidden_output))

        hidden_T = transpose(self.hidden_output)
        X_T = transpose(self.x)

        grad_ho = dot(hidden_T, output_delta)
        grad_ih = dot(X_T, hidden_delta)

        grad_ho = [[val * self.learning_rate for val in row] for row in grad_ho]
        grad_ih = [[val * self.learning_rate for val in row] for row in grad_ih]

        bias_output_grad = [[val * self.learning_rate for val in row] for row in sum_rows(output_delta)]
        bias_hidden_grad = [[val * self.learning_rate for val in row] for row in sum_rows(hidden_delta)]

        self.weights_hidden_output = add(self.weights_hidden_output, grad_ho)
        self.bias_output = add(self.bias_output, bias_output_grad)
        self.weights_input_hidden = add(self.weights_input_hidden, grad_ih)
        self.bias_hidden = add(self.bias_hidden, bias_hidden_grad)

    def predict(self, X):
        if isinstance(X[0], (int, float)):
            X = [X]

        hidden_activation = add(dot(X, self.weights_input_hidden), self.bias_hidden)
        hidden_output = sigmoid(hidden_activation)

        output_activation = add(dot(hidden_output, self.weights_hidden_output), self.bias_output)
        predicted_output = sigmoid(output_activation)

        if len(predicted_output) == 1:
            return predicted_output[0]
        return predicted_output

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.feed_forward()
            self.backward()
            if epoch % int(num_epochs / 10) == 0:
                loss = mean_squared_error(self.y, self.predicted_output)
                print(loss)
