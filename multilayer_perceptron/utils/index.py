from math import e

def sigmoid(x):
    if isinstance(x, (int, float)):
        return 1 / (1 + (e ** (-x)))
    elif isinstance(x, list):
        return [sigmoid(i) for i in x]
    else:
        raise TypeError("Unsupported type for sigmoid")

def sigmoid_derivative(x):
    if isinstance(x, (int, float)):
        return x * (1 - x)
    elif isinstance(x, list):
        return [sigmoid_derivative(i) for i in x]
    else:
        raise TypeError("Unsupported type for sigmoid_derivative")

def transpose(matrix):
    return [list(i) for i in zip(*matrix)]

def multiply_elementwise(a, b):
    return [[x * y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]

def is_vector(x):
    return all(not isinstance(i, (list, tuple)) for i in x)

def add(a, b):
    def to_2d(x):
        if isinstance(x[0], list):
            return x
        return [x]
    a_2d = to_2d(a)
    b_2d = to_2d(b)

    if len(a_2d[0]) != len(b_2d[0]):
        raise ValueError(f"Cannot add arrays with different number of columns: {len(a_2d[0])} vs {len(b_2d[0])}")

    result = [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a_2d, b_2d)]
    if not isinstance(a[0], list):
        return result[0]
    return result

def subtract(a, b):
    if isinstance(a[0], list):
        return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
    else:
        return [x - y for x, y in zip(a, b)]

def dot(a, b):
    b_T = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in b_T] for row in a]

def sum_rows(matrix):
    n_cols = len(matrix[0])
    return [[sum(row[i] for row in matrix) for i in range(n_cols)]]

def mean_squared_error(y, output):
    total = 0.0
    n = 0
    for row_y, row_out in zip(y, output):
        for yi, oi in zip(row_y, row_out):
            total += (yi - oi) ** 2
            n += 1
    return total / n
