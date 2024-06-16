import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.parent = None

class TP_RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, max_time):
        self.start = Node(start[0], start[1], start[2], 0)
        self.goal = Node(goal[0], goal[1], goal[2], max_time)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.max_time = max_time

    def get_random_node(self):
        return Node(np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(0, self.max_time))

    def get_nearest_node_index(self, node_list, rnd):
        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 + (node.z - rnd.z) ** 2 + (node.t - rnd.t) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def collision_check(self, node):
        for (ox, oy, oz, size, start_t, end_t) in self.obstacle_list:
            if start_t <= node.t <= end_t:
                dx = ox - node.x
                dy = oy - node.y
                dz = oz - node.z
                d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if d <= size:
                    return False
        return True

    def planning(self, ax):
        while True:
            rnd = self.get_random_node()
            nind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nind]

            direction = np.array([rnd.x - nearest_node.x, rnd.y - nearest_node.y, rnd.z - nearest_node.z, rnd.t - nearest_node.t])
            direction = direction / np.linalg.norm(direction)
            new_node = Node(nearest_node.x + direction[0], nearest_node.y + direction[1], nearest_node.z + direction[2], nearest_node.t + direction[3])
            new_node.parent = nearest_node

            if not self.collision_check(new_node):
                continue

            self.node_list.append(new_node)

            # 动态绘图更新
            ax.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], [nearest_node.z, new_node.z], "-g")
            plt.draw()
            plt.pause(0.01)

            if np.sqrt((new_node.x - self.goal.x) ** 2 + (new_node.y - self.goal.y) ** 2 + (new_node.z - self.goal.z) ** 2) <= 1.0 and new_node.t <= self.max_time:
                self.goal.parent = new_node
                self.node_list.append(self.goal)
                break

        path = [[self.goal.x, self.goal.y, self.goal.z, self.goal.t]]
        node = self.goal
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y, node.z, node.t])
        return path

def main():
    start = [0, 0, 0]
    goal = [5, 5, 5]
    obstacle_list = [
        (3, 3, 3, 1, 2, 6),  # 动态障碍物，存在于时间2到6之间
    ]
    rand_area = [-2, 7]
    max_time = 10  # 最大时间限制

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(rand_area[0], rand_area[1])
    ax.set_ylim(rand_area[0], rand_area[1])
    ax.set_zlim(rand_area[0], rand_area[1])

    for (ox, oy, oz, size, start_t, end_t) in obstacle_list:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = ox + size * np.outer(np.cos(u), np.sin(v))
        y = oy + size * np.outer(np.sin(u), np.sin(v))
        z = oz + size * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='r', alpha=0.3)

    tp_rrt = TP_RRT(start, goal, obstacle_list, rand_area, max_time)
    path = tp_rrt.planning(ax)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path")
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], '-r')
        plt.show()

if __name__ == '__main__':
    main()
