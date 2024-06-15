import numpy as np
from scipy.spatial.transform import Rotation
import random
from collision_detection import check_collision


def compute_link_info(chain, joint_angles, index_begin=0, y_offset=-1):
    link_info = {}
    joint_angles = np.append(0, np.append(joint_angles, np.array([0])))
    transformation_matrices = chain.forward_kinematics(joint_angles, True)

    for i in range(1, len(joint_angles)):  # 遍历从第一个有效链接到最后一个有效链接
        parent_pos = transformation_matrices[i - 1][:3, 3]
        parent_pos[1] = parent_pos[1] + y_offset
        child_pos = transformation_matrices[i][:3, 3]
        child_pos[1] = child_pos[1] + y_offset

        length = np.linalg.norm(child_pos - parent_pos)
        center_pos = (parent_pos + child_pos) / 2

        link_info[i - 1 + index_begin] = {
            'center_joint': child_pos,
            'joint_R': 0.1,  # 假设关节半径为0.1
            'length': length,
            'center_pos': center_pos,
            'link_R': length / 2
        }

    return link_info


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent


class RRTConnect:
    def __init__(self, chain, start, goal, max_iter=1000, step_size=0.1, goal_sample_rate=0.1):
        self.chain = chain
        self.start = Node(start)
        self.goal = Node(goal)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.tree_start = [self.start]
        self.tree_goal = [self.goal]

    def planning(self, link_info_main):
        for i in range(self.max_iter):
            if random.random() > self.goal_sample_rate:
                rnd_node = self.get_random_node()
            else:
                rnd_node = Node(self.goal.position)

            nearest_node = self.get_nearest_node(self.tree_start, rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if not self.check_collision(new_node, link_info_main):
                self.tree_start.append(new_node)
                if self.connect_trees(self.tree_start, self.tree_goal, link_info_main):
                    return self.get_final_path(new_node)

            nearest_node = self.get_nearest_node(self.tree_goal, rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if not self.check_collision(new_node, link_info_main):
                self.tree_goal.append(new_node)
                if self.connect_trees(self.tree_goal, self.tree_start, link_info_main):
                    return self.get_final_path(new_node)

        return None

    def get_random_node(self):
        rnd_position = np.array([random.uniform(-1, 1) for _ in range(6)])
        return Node(rnd_position)

    def get_nearest_node(self, tree, node):
        distances = [np.linalg.norm(n.position - node.position) for n in tree]
        nearest_index = distances.index(min(distances))
        return tree[nearest_index]

    def steer(self, from_node, to_node):
        direction = to_node.position - from_node.position
        distance = np.linalg.norm(direction)
        direction = direction / distance

        new_position = from_node.position + direction * min(self.step_size, distance)
        return Node(new_position, from_node)

    def connect_trees(self, tree_from, tree_to, link_info_main):
        nearest_node = self.get_nearest_node(tree_to, tree_from[-1])
        new_node = self.steer(nearest_node, tree_from[-1])

        if not self.check_collision(new_node, link_info_main):
            tree_to.append(new_node)
            if np.linalg.norm(new_node.position - tree_from[-1].position) < self.step_size:
                return True

        return False

    def check_collision(self, node, link_info_main):
        link_info_slave = compute_link_info(self.chain, node.position, index_begin=7, y_offset=0)
        link_info = {**link_info_main, **link_info_slave}
        # 获取link_info的大小
        num_links = len(link_info)
        num_features = 3 + 1 + 1 + 3 + 1  # center_joint (3), joint_R (1), length (1), center_pos (3), link_R (1)

        # 初始化一个矩阵来存储link_info中的值
        link_matrix = np.zeros((num_links, num_features))

        # 将link_info中的值填充到矩阵中
        for iii, (link_id, info) in enumerate(link_info.items()):
            link_matrix[iii, :3] = info['center_joint']
            link_matrix[iii, 3] = info['joint_R']
            link_matrix[iii, 4] = info['length']
            link_matrix[iii, 5:8] = info['center_pos']
            link_matrix[iii, 8] = info['link_R']
        collision = check_collision(O1=link_matrix[:7, :3], O2=link_matrix[7:, :3],
                                    R1=link_matrix[:7, 3], R2=link_matrix[7:, 3],
                                    A=link_matrix[:7, 5:8], B=link_matrix[7:, 5:8],
                                    Ra=link_matrix[:7, 8], Rb=link_matrix[7:, 8])
        print(collision)
        return collision

    def get_final_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(node.position)
        return path[::-1]  # Reverse the path
