import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# 定义Node类
class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None


# 计算欧几里得距离
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# 找到树中最近的节点
def nearest(tree, point):
    min_distance = float('inf')
    nearest_node = None
    for node in tree:
        current_distance = distance(node.point, point)
        if current_distance < min_distance:
            min_distance = current_distance
            nearest_node = node
    return nearest_node


# 生成朝向目标点的新节点
def steer(from_node, to_point, max_distance):
    direction = np.array(to_point) - np.array(from_node.point)
    length = np.linalg.norm(direction)
    direction = direction / length
    if length > max_distance:
        length = max_distance
    new_point = from_node.point + direction * length
    new_node = Node(new_point)
    new_node.parent = from_node
    return new_node


# 检查点是否无碰撞
def collision_free(point, obstacles, radius):
    for (ox, oy, oz, r) in obstacles:
        if distance((ox, oy, oz), point) <= r + radius:
            return False
    return True


# 获取从起点到给定节点的路径
def path(node):
    p = []
    while node:
        p.append(node.point)
        node = node.parent
    return p[::-1]


# 更新障碍物位置的函数（模拟移动）
def update_obstacles(obstacles, time_step):
    new_obstacles = []
    for ox, oy, oz, r in obstacles:
        new_obstacles.append((ox + time_step*0.01, oy + time_step*0.01, oz + time_step*0.01, r))
        # print((ox + time_step*0.01, oy + time_step*0.01, oz + time_step*0.01, r))
    return new_obstacles


# RRT-Connect算法
def rrt_connect(start, goal, obstacles, max_distance, radius, max_iter):
    start_node = Node(start)
    goal_node = Node(goal)
    tree_start = [start_node]
    tree_goal = [goal_node]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(start[0],goal[0])
    ax.set_ylim(start[1], goal[1])
    ax.set_zlim(start[2], goal[2])
    plt.ion()
    plt.show()

    for _ in range(max_iter):
        rand_point = np.random.rand(3) * 100
        nearest_start = nearest(tree_start, rand_point)
        new_start = steer(nearest_start, rand_point, max_distance)

        if collision_free(new_start.point, obstacles, radius):
            tree_start.append(new_start)
            ax.plot([nearest_start.point[0], new_start.point[0]],
                    [nearest_start.point[1], new_start.point[1]],
                    [nearest_start.point[2], new_start.point[2]], 'r-')
            plt.draw()
            plt.pause(0.01)

            nearest_goal = nearest(tree_goal, new_start.point)
            new_goal = steer(nearest_goal, new_start.point, max_distance)

            if collision_free(new_goal.point, obstacles, radius):
                tree_goal.append(new_goal)
                ax.plot([nearest_goal.point[0], new_goal.point[0]],
                        [nearest_goal.point[1], new_goal.point[1]],
                        [nearest_goal.point[2], new_goal.point[2]], 'b-')
                plt.draw()
                plt.pause(0.01)

                if distance(new_goal.point, new_start.point) <= max_distance:
                    path_start = path(new_start)
                    path_goal = path(new_goal)
                    plt.ioff()
                    return path_start + path_goal[::-1]
            else:
                tree_start, tree_goal = tree_goal, tree_start
        else:
            tree_start, tree_goal = tree_goal, tree_start

        # 更新障碍物位置并重新绘制
        obstacles = update_obstacles(obstacles, _ * 0.1)


        ax.scatter(*start, color='g', s=100)
        ax.scatter(*goal, color='b', s=100)
        lines = []
        for i,(ox, oy, oz, r) in  enumerate(obstacles):
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = ox + r * np.cos(u) * np.sin(v)
            y = oy + r * np.sin(u) * np.sin(v)
            z = oz + r * np.cos(v)
            line = ax.plot_surface(x, y, z, color='gray', alpha=1)
            lines.append(line)
        ax.set_box_aspect([1, 1, 1])

        plt.draw()
        plt.pause(0.1)
        for line in lines:
            line.remove()


    plt.ioff()
    return None


# 定义参数
start = (0, 0, 0)
goal = (100, 100, 100)
obstacles = [(50, 50, 50, 10), (70, 70, 70, 10)]
max_distance = 5
radius = 1
max_iter = 10000

# 运行RRT-Connect
path = rrt_connect(start, goal, obstacles, max_distance, radius, max_iter)

# 最终路径可视化
if path:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*zip(*path), marker='o', color='r')
    ax.scatter(*start, color='g', s=100)
    ax.scatter(*goal, color='b', s=100)
    for (ox, oy, oz, r) in obstacles:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = ox + r * np.cos(u) * np.sin(v)
        y = oy + r * np.sin(u) * np.sin(v)
        z = oz + r * np.cos(v)
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
else:
    print("No path found")
