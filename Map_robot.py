import numpy as np

class Map:
    def __init__(self, bounds, obstacles):
        """
        初始化地图。
        :param bounds: (x_min, x_max, y_min, y_max) 定义地图边界。
        :param obstacles: 一个列表，每个元素是 (center, radius) 表示障碍物的圆形。
        """
        self.bounds = bounds
        self.obstacles = obstacles

    def check_collision(self, point):
        """
        检查给定点是否与任何障碍物发生碰撞。
        :param point: (x, y) 需要检查的点。
        :return: 如果点与障碍物碰撞，返回True；否则返回False。
        """
        x, y = point
        x_min, x_max, y_min, y_max = self.bounds
        # 检查点是否在边界内
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return True
        # 检查点是否与障碍物碰撞
        for center, radius in self.obstacles:
            if np.linalg.norm(np.array(point) - np.array(center)) <= radius:
                return True
        return False

    def random_point(self):
        """
        在地图的边界内生成一个随机点。
        :return: (x, y) 随机点。
        """
        x_min, x_max, y_min, y_max = self.bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return (x, y)

# 创建一个具有特定边界和障碍物的地图实例
map_bounds = (0, 100, 0, 100)
map_obstacles = [((50, 50), 10), ((75, 75), 15), ((25, 25), 5)]
map = Map(bounds=map_bounds, obstacles=map_obstacles)
