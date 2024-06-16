import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

np.random.seed(1000)


# Node class to represent points in the trees
class Node:  # 伪代码中的节点表示
    def __init__(self, point):
        self.point = point  # 节点的坐标
        self.parent = None  # 节点的父节点


# Function to compute the Euclidean distance between two points
def distance(point1, point2):  # 伪代码中的距离计算
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to find the nearest node in the tree to a given point
def nearest(tree, point):  # 伪代码中的nearest(Tstart, qrand)
    # return min(tree, key=lambda node: distance(node.point, point))
    # 初始化最小距离为无穷大，最近的节点为None
    min_distance = float('inf')
    nearest_node = None

    # 遍历树中的每个节点
    for node in tree:
        # 计算当前节点到给定点的距离
        current_distance = distance(node.point, point)

        # 如果当前距离小于已记录的最小距离，更新最小距离和最近的节点
        if current_distance < min_distance:
            min_distance = current_distance
            nearest_node = node

    # 返回最近的节点
    return nearest_node


# Function to generate a new node in the direction of a given point
def steer(from_node, to_point, max_distance):  # 伪代码中的Generate(qrand, qnear, dm)
    direction = np.array(to_point) - np.array(from_node.point)
    length = np.linalg.norm(direction)
    direction = direction / length
    if length > max_distance:
        length = max_distance
    new_point = from_node.point + direction * length
    new_node = Node(new_point)
    new_node.parent = from_node
    return new_node


# Function to check if a point is collision-free
def collision_free(point, obstacles, radius):  # 伪代码中的Qfree检查
    for (ox, oy, oz, r) in obstacles:
        if distance((ox, oy, oz), point) <= r + radius:
            return False
    return True


# Function to retrieve the path from the start to a given node
def path(node):  # 伪代码中的路径生成
    p = []
    while node:
        p.append(node.point)
        node = node.parent
    return p[::-1]


# RRT-Connect algorithm
def rrt_connect(start, goal, obstacles, max_distance, radius, max_iter):
    start_node = Node(start)  # 起始节点
    goal_node = Node(goal)  # 目标节点
    tree_start = [start_node]  # 起始树
    tree_goal = [goal_node]  # 目标树

    for _ in range(max_iter):
        rand_point = np.random.rand(3) * 100  # 伪代码中的Sample()
        nearest_start = nearest(tree_start, rand_point)  # 伪代码中的nearest(Tstart, qrand)
        new_start = steer(nearest_start, rand_point, max_distance)  # 伪代码中的Generate(qrand, qnear, dm)

        if collision_free(new_start.point, obstacles, radius):  # 伪代码中的Qfree检查
            tree_start.append(new_start)
            nearest_goal = nearest(tree_goal, new_start.point)  # 伪代码中的nearest(Tgoal, qnew)
            new_goal = steer(nearest_goal, new_start.point, max_distance)  # 伪代码中的Generate(qnew, q'near, dm)

            if collision_free(new_goal.point, obstacles, radius):  # 伪代码中的Qfree检查
                tree_goal.append(new_goal)

                if distance(new_goal.point, new_start.point) <= max_distance:
                    path_start = path(new_start)
                    path_goal = path(new_goal)
                    return path_start + path_goal[::-1]
            else:
                tree_start, tree_goal = tree_goal, tree_start  # 伪代码中的Swap(Tstart, Tgoal)
        else:
            tree_start, tree_goal = tree_goal, tree_start  # 伪代码中的Swap(Tstart, Tgoal)
    return None


# Define parameters
start = (0, 0, 0)
goal = (100, 100, 100)
obstacles = [(50, 50, 50, 10), (70, 70, 70, 10)]
max_distance = 5
radius = 1
max_iter = 10000

# Run RRT-Connect
path = rrt_connect(start, goal, obstacles, max_distance, radius, max_iter)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if path:
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
