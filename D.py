import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DStarLite:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.open_list = []
        self.g = {}
        self.rhs = {}
        self.parents = {}
        self.k_m = 0
        self.initialize()

    def initialize(self):
        self.g[self.goal] = float('inf')
        self.rhs[self.goal] = 0
        heapq.heappush(self.open_list, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, node):
        g_rhs = min(self.g.get(node, float('inf')), self.rhs.get(node, float('inf')))
        return (g_rhs + heuristic(node, self.start) + self.k_m, g_rhs)

    def update_vertex(self, node):
        if node != self.goal:
            self.rhs[node] = min(self.g.get(neighbor, float('inf')) + 1 for neighbor in self.get_neighbors(node))
        if node in self.open_list:
            self.open_list.remove((self.calculate_key(node), node))
            heapq.heapify(self.open_list)
        if self.g.get(node, float('inf')) != self.rhs.get(node, float('inf')):
            heapq.heappush(self.open_list, (self.calculate_key(node), node))

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            x, y, z = node[0] + dx, node[1] + dy, node[2] + dz
            if (x, y, z) not in self.obstacles:
                neighbors.append((x, y, z))
        return neighbors

    def compute_shortest_path(self, ax):
        while self.open_list and (self.open_list[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g.get(self.start, float('inf'))):
            k_old, node = heapq.heappop(self.open_list)
            if k_old < self.calculate_key(node):
                heapq.heappush(self.open_list, (self.calculate_key(node), node))
            elif self.g.get(node, float('inf')) > self.rhs.get(node, float('inf')):
                self.g[node] = self.rhs[node]
                for neighbor in self.get_neighbors(node):
                    self.update_vertex(neighbor)
            else:
                self.g[node] = float('inf')
                self.update_vertex(node)
                for neighbor in self.get_neighbors(node):
                    self.update_vertex(neighbor)

            # 动态绘图更新
            ax.scatter(node[0], node[1], node[2], color='green')
            plt.draw()
            plt.pause(0.01)

    def update_obstacles(self, new_obstacles):
        for obstacle in new_obstacles:
            if obstacle not in self.obstacles:
                self.obstacles.add(obstacle)
                for neighbor in self.get_neighbors(obstacle):
                    self.update_vertex(neighbor)
        self.compute_shortest_path(ax)

    def find_path(self, ax):
        self.compute_shortest_path(ax)
        path = []
        node = self.start
        while node != self.goal:
            path.append(node)
            node = min(self.get_neighbors(node), key=lambda n: self.g.get(n, float('inf')))
        path.append(self.goal)
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], '-r')
        plt.show()
        return path

def heuristic(point, goal):
    return abs(point[0] - goal[0]) + abs(point[1] - goal[1]) + abs(point[2] - goal[2])

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_environment(ax, start, goal, obstacles)

d_star_lite = DStarLite(start, goal, obstacles)
path = d_star_lite.find_path(ax)
print("D* Lite Path:", path)

# 更新障碍物
new_obstacles = [(2, 2, 2), (2, 3, 2)]
d_star_lite.update_obstacles(new_obstacles)
new_path = d_star_lite.find_path(ax)
print("Updated D* Lite Path:", new_path)
