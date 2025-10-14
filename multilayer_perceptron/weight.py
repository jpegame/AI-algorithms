import math
import random

def generate_random_weights(*size):
    if not size:
        size = (1,)
    n = math.prod(size)
    result = []
    
    for _ in range((n + 1) // 2):
        u1, u2 = random.random(), random.random()
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        result.extend([z0, z1])
    
    result = result[:n]
    if len(size) == 1:
        return result if size[0] > 1 else result[0]
    else:
        it = iter(result)
        def reshape(shape):
            if len(shape) == 1:
                return [next(it) for _ in range(shape[0])]
            return [reshape(shape[1:]) for _ in range(shape[0])]
        return reshape(size)
