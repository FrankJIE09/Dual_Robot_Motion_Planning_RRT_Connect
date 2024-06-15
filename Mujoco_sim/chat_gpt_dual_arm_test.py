import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation, Slerp
from ikpy.chain import Chain
from Bézier_curve import bezier_curve
from collision_detection import check_collision
from rrt_connect import RRTConnect  # 假设你有一个RRT-Connect算法实现
import time


# 定义一个函数来计算连杆中心位置、关节位置及其半径
def compute_link_info(chain, joint_angles, index_begin=0, y_offset=-1):
    link_info = {}
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


# 从URDF文件中创建机械臂链
my_chain = Chain.from_urdf_file("./config/ur5e.urdf",
                                active_links_mask=[False, True, True, True, True, True, True, False])
np.set_printoptions(precision=5, suppress=True, linewidth=100)

# 加载Mujoco模型
model = mujoco.MjModel.from_xml_path("./Double_UR5e/scene.xml")
data = mujoco.MjData(model)
model.opt.gravity = (0, 0, -9.8)
mujoco.mj_resetDataKeyframe(model, data, 0)
rz90 = Rotation.from_euler('z', 90, degrees=True)
rz90_matrix = rz90.as_matrix()

# 使用Mujoco的viewer来可视化模型
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 主机械臂的路径生成
    init_pos = data.xpos[-2]
    path_array = np.array([[init_pos[0], init_pos[1], init_pos[2]],
                           [init_pos[0] - 1.5, init_pos[1], init_pos[2] - 0.1],
                           [init_pos[0], init_pos[1], init_pos[2] - 0.2]]
                          ) @ rz90_matrix

    interp_pos = bezier_curve(path_array)
    interp_rots = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

    # 从机械臂路径规划初始化
    init_pos_r = data.xpos[-2]
    target_pos_r = np.array([init_pos_r[0] + 1, init_pos_r[1], init_pos_r[2] - 0.2])

    rrt = RRTConnect(my_chain, start=init_pos_r, goal=target_pos_r, max_iter=1000, step_size=0.1)

    # 计算初始矩阵
    init_matrix = data.xmat[6].reshape(3, 3).copy()
    init_matrix_r = data.xmat[-2].reshape(3, 3).copy()

    i = 0
    iter_count = 0
    target_matrix = np.eye(4)
    target_matrix_r = np.eye(4)

    while i < len(interp_pos):
        # 主机械臂
        target_matrix[:3, :3] = init_matrix @ interp_rots
        target_matrix[:3, 3] = interp_pos[i]

        # 使用逆运动学求解关节角度
        ik_joint = my_chain.inverse_kinematics_frame(target_matrix,
                                                     initial_position=np.append(np.append(np.array(0), data.qpos[:6]),
                                                                                np.array(0)), orientation_mode='all')

        link_info_main = compute_link_info(my_chain, ik_joint)

        if iter_count == 0:
            # 从机械臂路径规划
            rrt_path = rrt.planning(link_info_main)
            if rrt_path is None:
                print("RRT-Connect failed to find a path")
                break

        # 从机械臂
        target_matrix_r[:3, :3] = init_matrix_r @ np.array(rrt_path[iter_count])
        target_matrix_r[:3, 3] = rrt_path[iter_count]

        # 使用逆运动学求解关节角度
        ik_joint_r = my_chain.inverse_kinematics_frame(target_matrix_r,
                                                       initial_position=np.append(np.append(np.array(0), data.qpos[6:]),
                                                                                  np.array(0)), orientation_mode='all')

        link_info_slave = compute_link_info(my_chain, ik_joint_r, index_begin=7, y_offset=0)
        link_info = {**link_info_main, **link_info_slave}

        num_links = len(link_info)
        num_features = 3 + 1 + 1 + 3 + 1  # center_joint (3), joint_R (1), length (1), center_pos (3), link_R (1)

        link_matrix = np.zeros((num_links, num_features))

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

        if collision:
            print("Collision detected")
            break

        # 控制舵机运动
        data.ctrl = np.hstack((ik_joint[1:7], ik_joint_r[1:7]))
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)

        i += 1
        iter_count += 1

        if iter_count >= len(rrt_path):
            iter_count = 0  # Reset the iteration count for the RRT-Connect path

        viewer.sync()
