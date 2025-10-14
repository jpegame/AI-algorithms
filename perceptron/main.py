from algorithm import generate_final_weight

dados = [
    {
        "x": [0, 0],
        "y": 0
    },
    {
        "x": [0, 1],
        "y": 0
    },
    {
        "x": [1, 0],
        "y": 0
    },
    {
        "x": [1, 1],
        "y": 1
    }
]

w_list, bias, n = [0.3, 0.2], -0.3, 0.1
max_epocas = 10_000

w_list_result = generate_final_weight(dados, w_list, bias, n, max_epocas)

print(f"WEIGHTS:", end=" ")
for w in w_list_result:
    print(f'{w:.2f}', end=", ")
