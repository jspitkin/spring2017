def f(vec):
    return vec[0] * vec[0] + (vec[1] * vec[1]) / 4

def gradient(vec):
    grad = [0, 0]
    grad[0] = vec[0] * 2
    grad[1] = vec[1] / 2
    return grad

def step(vec, eta):
    grad = gradient(vec)
    grad[0] = vec[0] - grad[0] * eta
    grad[1] = vec[1] - grad[1] * eta
    return grad
    
eta = 1.0001
step_count = 1
vec = [1, 1]
while(True):
    print('Step', step_count, vec, f(vec), eta)
    step_count += 1
    vec = step(vec, eta)
    if (f(vec) < 0.0001):
        break
