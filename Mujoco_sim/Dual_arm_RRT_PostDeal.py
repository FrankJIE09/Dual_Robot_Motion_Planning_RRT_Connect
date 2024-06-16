import time
from collision_detection import check_collision
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.ndimage import uniform_filter1d
from ikpy.chain import Chain
from Bézier_curve import bezier_curve
import copy
from rrt_connect_modify import RRTConnect

seg = 500


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
#
np.set_printoptions(precision=5, suppress=True, linewidth=100)

# 加载Mujoco模型
model = mujoco.MjModel.from_xml_path("./Double_UR5e/scene.xml")
data = mujoco.MjData(model)
model.opt.gravity = (0, 0, -9.8)
mujoco.mj_resetDataKeyframe(model, data, 0)
# 创建绕 z 轴旋转 90° 的旋转对象
rz90 = Rotation.from_euler('z', 90, degrees=True)
# 获取旋转矩阵
rz90_matrix = rz90.as_matrix()

# 使用Mujoco的viewer来可视化模型
with mujoco.viewer.launch_passive(model, data) as viewer:
    init_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    init_rpy = init_orientation.as_euler('xyz')

    rpy_array = np.array([[init_rpy[0], init_rpy[1], init_rpy[2]],
                          [init_rpy[0], init_rpy[1] + np.pi / 4, init_rpy[2]],
                          [init_rpy[0], init_rpy[1] + np.pi / 2, init_rpy[2]]]
                         )

    rpy_array_r = np.array([[init_rpy[0], init_rpy[1], init_rpy[2]],
                            [init_rpy[0], init_rpy[1] + np.pi / 4, init_rpy[2]],
                            [init_rpy[0], init_rpy[1] + np.pi / 2, init_rpy[2]]]
                           )
    # 定义关键帧的姿态
    quaternions = [Rotation.from_euler('zyx', rpy_array[0]),
                   Rotation.from_euler('zyx', rpy_array[1], ),
                   Rotation.from_euler('zyx', rpy_array[2])]

    quaternions_r = [Rotation.from_euler('zyx', rpy_array_r[0]),
                     Rotation.from_euler('zyx', rpy_array_r[1], ),
                     Rotation.from_euler('zyx', rpy_array_r[2])]
    key_rots = Rotation.random(len(quaternions))
    key_rots_r = Rotation.random(len(quaternions_r))

    for i in range(quaternions.__len__()):
        key_rots[i] = quaternions[i]

    for i in range(quaternions_r.__len__()):
        key_rots_r[i] = quaternions_r[i]
    # 定义关键帧的时间
    key_times = [0, 2, 4]
    key_times_r = [0, 2, 4]

    # 使用球面插值法生成平滑的插值姿态
    slerp = Slerp(key_times, key_rots)
    slerp_r = Slerp(key_times_r, key_rots_r)

    times = np.linspace(0, 4, seg)
    times_r = np.linspace(0, 4, seg)

    interp_rots = slerp(times)
    interp_rots_r = slerp_r(times_r)

    init_pos = data.xpos[-2]
    init_pos_r = data.xpos[-2]

    path_array = np.array([[init_pos[0], init_pos[1], init_pos[2]],
                           [init_pos[0] - 1.5, init_pos[1], init_pos[2] - 0.1],
                           [init_pos[0] + 0.2, init_pos[1], init_pos[2] - 0.2]]
                          ) @ rz90_matrix

    path_array_r = np.array([[init_pos_r[0], init_pos_r[1], init_pos_r[2]],
                             [init_pos_r[0] + 1.5, init_pos_r[1], init_pos_r[2]],
                             [init_pos_r[0] - 0.2, init_pos_r[1], init_pos_r[2] - 0.2]]
                            ) @ rz90_matrix

    interp_pos = bezier_curve(path_array, seg=seg)
    interp_pos_r = bezier_curve(path_array_r, seg=seg)

    i = 0
    init_matrix = data.xmat[6].reshape(3, 3).copy()
    init_matrix_r = data.xmat[-2].reshape(3, 3).copy()

    target_matrix = np.eye(4)
    target_matrix_r = np.eye(4)
    iter_count = 0
    init_node_matrix_r = np.eye(4)

    init_node_matrix_r[:3, :3] = np.dot(init_matrix_r, interp_rots_r[i].as_matrix())
    init_node_matrix_r[:3, 3] = interp_pos_r[i]
    last_ik_joint_r = my_chain.inverse_kinematics_frame(init_node_matrix_r,
                                                        initial_position=np.append(
                                                            np.append(np.array(0), data.qpos[6:]),
                                                            np.array(0)), orientation_mode='all')
    # 循环控制机械臂的姿态变化
    last_rrt_path = last_ik_joint_r[1:7]
    r_i = 0
    over_delay = 0
    robot_joint = [data.qpos[:6]]
    robot_slave_joint = [data.qpos[6:]]
    while i < seg + 1 and r_i < seg + 1:
        print(i)
        if i >= seg:
            i = seg - 1
        if r_i >= seg:
            r_i = seg - 1
            over_delay = over_delay + 1
            if over_delay > 100:
                break
        # 计算目标变换矩阵
        target_matrix[:3, :3] = np.dot(init_matrix, interp_rots[i].as_matrix())
        target_matrix[:3, 3] = interp_pos[i]

        target_matrix_r[:3, :3] = np.dot(init_matrix_r, interp_rots_r[r_i].as_matrix())
        target_matrix_r[:3, 3] = interp_pos_r[r_i]
        # 使用逆运动学求解关节角度
        ik_joint = my_chain.inverse_kinematics_frame(target_matrix,
                                                     initial_position=np.append(np.append(np.array(0), robot_joint[-1]),
                                                                                np.array(0)), orientation_mode='all')
        ik_joint_r = my_chain.inverse_kinematics_frame(target_matrix_r,
                                                       initial_position=np.append(
                                                           np.append(np.array(0), robot_slave_joint[-1]),
                                                           np.array(0)), orientation_mode='all')

        # 计算连杆信息
        link_info_main = compute_link_info(my_chain, ik_joint)
        rrt = RRTConnect(my_chain, start=last_ik_joint_r[1:7], goal=ik_joint_r[1:7], max_iter=5, step_size=0.1)
        last_ik_joint_r = ik_joint_r.copy()
        if iter_count == 0:
            # 从机械臂路径规划
            rrt_path = rrt.planning(link_info_main)
            if rrt_path is None:
                print("RRT-Connect failed to find a path")
                # break
                data.ctrl = np.hstack((ik_joint[1:7], last_rrt_path[-1]))
                mujoco.mj_step(model, data)
                i = i + 1
                viewer.sync()
                robot_joint.append(ik_joint[1:7])
                robot_slave_joint.append(last_rrt_path[-1])
                continue
        data.ctrl = np.hstack((ik_joint[1:7], rrt_path[-1]))
        last_rrt_path = rrt_path.copy()
        mujoco.mj_step(model, data)
        i = i + 1
        r_i = r_i + 1
        robot_joint.append(ik_joint[1:7])
        robot_slave_joint.append(rrt_path[-1])
        viewer.sync()
robot_joint = np.array(robot_joint)
robot_slave_joint = np.array(robot_slave_joint)

filtered_robot_joint = uniform_filter1d(robot_joint, size=10, mode='reflect', axis=0)
filtered_robot_slave_joint = uniform_filter1d(robot_slave_joint, size=10, mode='reflect', axis=0)
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(len(robot_joint)):
        data.ctrl = np.hstack((filtered_robot_joint[i, :], filtered_robot_slave_joint[i, :]))
        mujoco.mj_step(model, data)
        i = i + 1
        viewer.sync()
