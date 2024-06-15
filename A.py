import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, x, y, z, cost, heuristic, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

def a_star(start, goal, obstacles):
    open_list = []
    closed_list = set()
    start_node = Node(start[0], start[1], start[2], 0, heuristic(start, goal))
    goal_node = Node(goal[0], goal[1], goal[2], 0, 0)
    heapq.heappush(open_list, start_node)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_environment(ax, start, goal, obstacles)

    while open_list:
        current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y, current_node.z) in closed_list:
            continue

        closed_list.add((current_node.x, current_node.y, current_node.z))

        if current_node.x == goal_node.x and current_node.y == goal_node.y and current_node.z == goal_node.z:
            return reconstruct_path(current_node, ax)

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if (neighbor.x, neighbor.y, neighbor.z) in closed_list or not is_free(neighbor, obstacles):
                continue

            neighbor.cost = current_node.cost + 1
            neighbor.heuristic = heuristic((neighbor.x, neighbor.y, neighbor.z), goal)
            neighbor.parent = current_node
            heapq.heappush(open_list, neighbor)

            # 动态绘图更新
            ax.plot([current_node.x, neighbor.x], [current_node.y, neighbor.y], [current_node.z, neighbor.z], "-g")
            plt.draw()
            plt.pause(0.01)

    return None

def heuristic(point, goal):
    return abs(point[0] - goal[0]) + abs(point[1] - goal[1]) + abs(point[2] - goal[2])

def get_neighbors(node):
    neighbors = []
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        neighbors.append(Node(node.x + dx, node.y + dy, node.z + dz, 0, 0))
    return neighbors

def is_free(node, obstacles):
    for obs in obstacles:
        if node.x == obs[0] and node.y == obs[1] and node.z == obs[2]:
            return False
    return True

def reconstruct_path(node, ax):
    path = []
    while node is not None:
        path.append((node.x, node.y, node.z))
        node = node.parent
    path = path[::-1]
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-r')
    plt.show()
    return path

def plot_environment(ax, start, goal, obstacles):
    ax.scatter(start[0], start[1], start[2], color='blue', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    for obs in obstacles:
        ax.scatter(obs[0], obs[1], obs[2], color='black', s=100)
    ax.legend()

# 示例
start = (0, 0, 0)
goal = (5, 5, 5)
obstacles = [(3, 3, 3), (3, 4, 3), (4, 3, 3)]

path = a_star(start, goal, obstacles)
print("A* Path:", path)
