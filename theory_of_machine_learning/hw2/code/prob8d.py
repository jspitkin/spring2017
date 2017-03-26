from random import uniform
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


def gradient(vec, points):



def optimum(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    x = x / len(points)
    y = y / len(points)
    return [x, y]


eta = 0.5
vec = [uniform(-1, 1), uniform(-1, 1)]
points = random_points(100)
