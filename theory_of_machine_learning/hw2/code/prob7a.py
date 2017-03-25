import math

def f(vec, eta):
    return abs(vec[0])

def gradient(vec, eta):
    grad = [0]
    grad[0] = vec[0] / abs(vec[0])
    return grad

def step(vec, eta):
    grad = gradient(vec, eta)
    print('Gradient:', grad)
    grad[0] = vec[0] - grad[0] * eta
    return grad
    
eta = 0.03
step_count = 1
vec = [eta / 2]
while(True):
    print('Step', step_count, vec, f(vec, eta), eta)
    step_count += 1
    vec = step(vec, eta)
    if (abs(f(vec, eta)) == 0):
        break
print('Step', step_count, vec, f(vec, eta), eta)
