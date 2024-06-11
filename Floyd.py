import numpy as np
import matplotlib.pyplot as plt

def floyd_warshall(n, dist):
    # 初始化路径矩阵
    path = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == np.inf:
                path[i][j] = -1  # 不可达路径标记为-1
            else:
                path[i][j] = j

    # Floyd-Warshall算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = path[i][k]

    return dist, path

def get_path(path, start, end):
    if path[start][end] == -1:
        return []
    route = [start]
    while start != end:
        start = path[start][end]
        route.append(start)
    return route

def plot_graph(dist_matrix, path, nodes, start, end):
    fig, ax = plt.subplots()

    # 绘制节点
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'o', markersize=10, label=f'Node {i}')
        ax.text(x, y, f'  {i}', fontsize=12)

    # 绘制边
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if dist_matrix[i][j] != np.inf and i != j:
                x_values = [nodes[i][0], nodes[j][0]]
                y_values = [nodes[i][1], nodes[j][1]]
                ax.plot(x_values, y_values, 'gray', linestyle='--')

    # 绘制最短路径
    route = get_path(path, start, end)
    if route:
        for k in range(len(route) - 1):
            i, j = route[k], route[k+1]
            x_values = [nodes[i][0], nodes[j][0]]
            y_values = [nodes[i][1], nodes[j][1]]
            ax.plot(x_values, y_values, 'r', linewidth=2.5)

    ax.legend()
    plt.title(f'Shortest Path from Node {start} to Node {end}')
    plt.show()

# 示例：构建图的距离矩阵
n = 4
inf = np.inf
dist = np.array([
    [0, 3, inf, 7],
    [8, 0, 2, inf],
    [5, inf, 0, 1],
    [2, inf, inf, 0]
])

# 节点位置
nodes = [(0, 0), (1, 2), (4, 2), (3, 0)]

# 执行Floyd-Warshall算法
dist, path = floyd_warshall(n, dist)

# 打印最短路径距离矩阵
print("Shortest path distance matrix:")
print(dist)

# 打印路径矩阵
print("Path matrix:")
print(path)

# 绘制从顶点0到顶点3的路径
plot_graph(dist, path, nodes, 0, 3)
