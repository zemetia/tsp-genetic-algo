import numpy as np
from random import randint
import matplotlib.pyplot as plt
import math


class CityNode:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y


def get_dist(a: int, b: int):
    return math.sqrt(
        ((cities[a].x - cities[b].x) ** 2) +
        ((cities[a].y - cities[b].y) ** 2)
    )


def calc_dist(path: np.ndarray):
    total: int = 0
    for i in range(1, city_amount):
        total += get_dist(path[i-1], path[i])
    return total


def mutate(path: np.ndarray, rate: int):
    if randint(0, 100) < rate and path.size > 0:
        a: int = randint(0, path.size-1)
        b: int = randint(0, path.size-1)
        temp: int = path[a]
        path[a] = path[b]
        path[b] = temp


def crossover(size: int, path_a: np.ndarray, path_b: np.ndarray, mutation_rate: int = 40):
    start = randint(0, int(size/3))  # minimal ada 1 array yang ditukar
    end = randint(start, size-1)
    part_a = path_a.copy()[start:end]
    part_b = path_b.copy()[start:end]

    new_a = path_b[~np.in1d(path_b, part_a)]
    new_b = path_a[np.in1d(path_a, part_b)]

    new_a = np.concatenate((new_a[0:start], part_a, new_a[start:]))
    new_b = np.concatenate((new_b[0:start], part_b, new_b[start:]))

    mutate(new_a, mutation_rate)
    mutate(new_b, mutation_rate)

    return [new_a, new_b]


def genetic_algo(generation: int = 100):
    generation_best = []

    population = []
    fitness = []

    for j in range(population_size):
        arr_copy = data.copy()
        np.random.shuffle(arr_copy)
        population.append(arr_copy)
        fitness.append((calc_dist(arr_copy), j))

    fitness.sort()
    generation_best.append(fitness[0])

    for i in range(generation):
        # generate new generation by crossovering the best with random path
        population_new = []
        fitness_new = []

        for j in range(population_size):
            cross = crossover(
                city_amount,
                population[generation_best[i][1]],
                population[randint(0, population_size-1)],
                45
            )[0]

            population_new.append(cross)
            fitness_new.append((calc_dist(cross), j))

        fitness_new.sort()
        generation_best.append(fitness_new[0])
        print(fitness_new[0], population_new[0])

        population = population_new
        fitness = fitness_new

    last_best = generation_best[len(generation_best) - 1]
    return population[last_best[1]], last_best[0]


city_amount = 25
map_size = 100
population_size = 150

data = np.array(range(city_amount))
cities = [CityNode(randint(0, map_size-1), randint(0, map_size-1))
          for i in range(city_amount)]

path, dist = genetic_algo(500)
print("result: ", path, dist)


# x and y initiate
x_coords = [city.x for city in cities]
y_coords = [city.y for city in cities]

# randomly connect the points
lines = []
for i in range(1, city_amount):
    city_a = cities[path[i-1]]
    city_b = cities[path[i]]
    lines.append([(city_a.x, city_b.x), (city_a.y, city_b.y)])

# plot the points and lines
fig, ax = plt.subplots()
for line in lines:
    ax.plot(line[0], line[1], color='blue')
ax.scatter(x_coords, y_coords, color='red')

# add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Distance found {dist}')

# show the plot
plt.show()
