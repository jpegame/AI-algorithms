from math import isclose

def activation_function(v):
    return 1 if v >= 0 else 0

def generate_final_weight(
    data: list[dict[str, float]],
    w_list: list[float],
    bias: float,
    n: float,
    max_time: int,
    epsilon: float = 1e-9,
    round_weights_digits: int | None = 10
) -> list[float]:
    continue_running = True
    num_time = 0
    while continue_running and num_time < max_time:
        continue_running = False
        for entry in data:
            print("\nNew iteration:")
            y = bias
            print(f'y = {bias}', end='')
            for x, w in zip(entry['x'], w_list):
                y += x * w
                print(f' + ({x} * {w:.2f})', end='')
            y = round(y, 5)
            print(f'\ny = {y:.2f}')

            error = entry['y'] - activation_function(y)
            print(f'f({y:.2f}) = {activation_function(y)}')
            print(f'e = ({entry["y"]:.2f} - {activation_function(y)}) = {error:.2f}')

            if isclose(error, 0.0, abs_tol=epsilon):
                continue

            continue_running = True
            for i, x in enumerate(entry["x"]):
                update = n * error * x
                print(f'w{i + 1} = {w_list[i]:.10f} + ({n} * {error} * {x}) = ', end='')
                w_list[i] += update

                if round_weights_digits is not None:
                    w_list[i] = round(w_list[i], round_weights_digits)

                print(f'{w_list[i]:.10f}')
        
        print('=' * 30)
        num_time += 1

    return w_list