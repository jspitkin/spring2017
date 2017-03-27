from random import uniform
from random import randint
import math

def random_points(count):
    points = []
    for _ in range(count):
        points.append([uniform(-1, 1), uniform(-1, 1)])
    return points


def f(vec, points):
    result = 0
    for point in points:
        result = math.pow(vec[0] - point[0], 2) + math.pow(vec[1] - point[1], 2)
    result = result / len(points)
    return result


def f_i(vec, point):
    return math.pow(vec[0] - point[0], 2) + math.pow(vec[1] - point[1], 2)


def gradient(vec, point):
   grad = [0, 0]
   grad[0] = 2 * vec[0] - point[0] - point[0]
   grad[1] = 2 * vec[1] - point[1] - point[1]
   return grad


def step(vec, point, eta):
    grad = gradient(vec, point)
    grad[0] = vec[0] - grad[0] * eta
    grad[1] = vec[1] - grad[1] * eta
    return grad


def optimum(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    x = x / len(points)
    y = y / len(points)
    return [x, y]


def distance(vec_dist, opt_vec, points):
    return abs(f(vec, points) - f(opt_vec, points))


def update_eta(t):
    return 1.0 / t


# Experiment 1 - fixed value for eta
eta = 0.5
average_count = 5
T = 1000
T_sum_10 = 0
T_sum_100 = 0
T_sum_1000 = 0
points = random_points(100)
opt_vec = optimum(points)

for _ in range(average_count):
    vec = [uniform(-100000, 100000), uniform(-100000, 100000)]
    for t in range(1, T+1):
        point = points[randint(0, len(points)-1)] 
        vec = step(vec, point, eta)
        if t == 10:
            T_sum_10 += distance(vec, opt_vec, points)
        elif t == 100:
            T_sum_100 += distance(vec, opt_vec, points)
        elif t == 1000:
            T_sum_1000 += distance(vec, opt_vec, points)

T_sum_10 = T_sum_10 / average_count
T_sum_100 = T_sum_100 / average_count
T_sum_1000 = T_sum_1000 / average_count

print('Results averaged over', average_count, 'experiments with fixed eta.')
print('Distance from optimum when T = 10:', T_sum_10)
print('Distance from optimum when T = 100:', T_sum_100)
print('Distance from optimum when T = 1000:', T_sum_1000)

# Experiment 2 - dynamic value for eta
eta = 0
average_count = 5
T = 1000
T_sum_10 = 0
T_sum_100 = 0
T_sum_1000 = 0
points = random_points(100)
opt_vec = optimum(points)

for _ in range(average_count):
    vec = [uniform(-100000, 1000000), uniform(-100000, 100000)]
    for t in range(1, T+1):
        eta = update_eta(t)
        point = points[randint(0, len(points)-1)] 
        vec = step(vec, point, eta)
        if t == 10:
            T_sum_10 += distance(vec, opt_vec, points)
        elif t == 100:
            T_sum_100 += distance(vec, opt_vec, points)
        elif t == 1000:
            T_sum_1000 += distance(vec, opt_vec, points)

T_sum_10 = T_sum_10 / average_count
T_sum_100 = T_sum_100 / average_count
T_sum_1000 = T_sum_1000 / average_count

print()
print('Results averaged over', average_count, 'experiments with 1/t eta.')
print('Distance from optimum when T = 10:', T_sum_10)
print('Distance from optimum when T = 100:', T_sum_100)
print('Distance from optimum when T = 1000:', T_sum_1000)
