import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import random


class Node:
    def __init__(self, x, y, z, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent


def rrt(start, goal, obstacle_list, rand_area, max_iter=1000):
    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    node_list = [start_node]

    for _ in range(max_iter):
        rand_node = get_random_node(rand_area)
        nearest_node = get_nearest_node(node_list, rand_node)
        new_node = steer(nearest_node, rand_node)

        if collision_check(new_node, obstacle_list):
            node_list.append(new_node)

            if np.linalg.norm([new_node.x - goal_node.x, new_node.y - goal_node.y, new_node.z - goal_node.z]) < 1.0:
                goal_node.parent = new_node
                node_list.append(goal_node)
                return get_path(goal_node)
    return None


def get_random_node(rand_area):
    return Node(np.random.uniform(rand_area[0], rand_area[1]),
                np.random.uniform(rand_area[0], rand_area[1]),
                np.random.uniform(rand_area[0], rand_area[1]))


def get_nearest_node(node_list, rand_node):
    return min(node_list,
               key=lambda node: np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y, node.z - rand_node.z]))


def steer(from_node, to_node, extend_length=1.0):
    direction = np.array([to_node.x - from_node.x, to_node.y - from_node.y, to_node.z - from_node.z])
    length = np.linalg.norm(direction)
    direction = direction / length
    return Node(from_node.x + direction[0] * extend_length,
                from_node.y + direction[1] * extend_length,
                from_node.z + direction[2] * extend_length,
                from_node)


def collision_check(node, obstacle_list):
    for (ox, oy, oz, size) in obstacle_list:
        if np.linalg.norm([node.x - ox, node.y - oy, node.z - oz]) <= size:
            return False
    return True


def get_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append([node.x, node.y, node.z])
        node = node.parent
    return path[::-1]


def genetic_algorithm(path, obstacle_list, max_iter=100):
    population_size = 20
    mutation_rate = 0.1
    population = [path for _ in range(population_size)]

    for _ in range(max_iter):
        population = sorted(population, key=lambda p: path_cost(p))
        new_population = population[:population_size // 2]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child, obstacle_list, rand_area)
            new_population.append(child)

        population = new_population

    return min(population, key=lambda p: path_cost(p))


def path_cost(path):
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    return parent1[:crossover_point] + parent2[crossover_point:]


def mutate(path, obstacle_list, rand_area):
    mutation_point = random.randint(1, len(path) - 2)
    rand_node = get_random_node(rand_area)
    new_path = path[:mutation_point] + [[rand_node.x, rand_node.y, rand_node.z]] + path[mutation_point + 1:]
    if all(collision_check(Node(*node), obstacle_list) for node in new_path):
        return new_path
    return path


def plot_path(path, ax, color='r'):
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color)


def plot_environment(ax, start, goal, obstacles):
    ax.scatter(start[0], start[1], start[2], color='blue', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    for obs in obstacles:
        ax.scatter(obs[0], obs[1], obs[2], color='black', s=100)
    ax.legend()


# 示例
start = [0, 0, 0]
goal = [10, 10, 10]
obstacles = [(5, 5, 5, 1), (6, 6, 6, 1), (7, 8, 7, 1)]
rand_area = [-2, 12]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_environment(ax, start, goal, obstacles)

# 使用RRT生成初始路径
initial_path = rrt(start, goal, obstacles, rand_area)
if initial_path is not None:
    plot_path(initial_path, ax, 'g')
    print("Initial RRT Path:", initial_path)

    # 使用遗传算法优化路径
    optimized_path = genetic_algorithm(initial_path, obstacles)
    plot_path(optimized_path, ax, 'r')
    print("Optimized GA_RRT Path:", optimized_path)
else:
    print("No path found using RRT")

plt.show()
