import time
from collision_detection import check_collision
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from ikpy.chain import Chain
from Bézier_curve import bezier_curve
import copy

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

    times = np.linspace(0, 4, 2000)
    times_r = np.linspace(0, 4, 2000)

    interp_rots = slerp(times)
    interp_rots_r = slerp_r(times_r)

    init_pos = data.xpos[-2]
    init_pos_r = data.xpos[-2]
    # import random
    # for _ in range(3):
    #     random_pos = [random.uniform(-1,1) for _ in range(3)]
    path_array = np.array([[init_pos[0], init_pos[1], init_pos[2]],
                           [init_pos[0] - 1.5, init_pos[1], init_pos[2] - 0.1],
                           [init_pos[0], init_pos[1], init_pos[2] - 0.2]]
                          ) @ rz90_matrix

    path_array_r = np.array([[init_pos_r[0], init_pos_r[1], init_pos_r[2]],
                             [init_pos_r[0] + 1, init_pos_r[1], init_pos_r[2]],
                             [init_pos_r[0], init_pos_r[1], init_pos_r[2] - 0.2]]
                            ) @ rz90_matrix

    interp_pos = bezier_curve(path_array)
    interp_pos_r = bezier_curve(path_array_r)

    i = 0
    init_matrix = data.xmat[6].reshape(3, 3).copy()
    init_matrix_r = data.xmat[-2].reshape(3, 3).copy()

    target_matrix = np.eye(4)
    target_matrix_r = np.eye(4)

    # 循环控制机械臂的姿态变化
    while i < 2000:
        # 计算目标变换矩阵
        target_matrix[:3, :3] = np.dot(init_matrix, interp_rots[i].as_matrix())
        # target_matrix[:3, :3] = np.dot(init_matrix, init_orientation.as_matrix())
        target_matrix[:3, 3] = interp_pos[i]

        target_matrix_r[:3, :3] = np.dot(init_matrix_r, interp_rots_r[i].as_matrix())
        target_matrix_r[:3, 3] = interp_pos_r[i]
        # 使用逆运动学求解关节角度
        ik_joint = my_chain.inverse_kinematics_frame(target_matrix,
                                                     initial_position=np.append(np.append(np.array(0), data.qpos[:6]),
                                                                                np.array(0)), orientation_mode='all')
        ik_joint_r = my_chain.inverse_kinematics_frame(target_matrix_r,
                                                       initial_position=np.append(np.append(np.array(0), data.qpos[6:]),
                                                                                  np.array(0)), orientation_mode='all')

        # 存储每个连杆的长度和中心位置
        link_info = {}
        for ii in range(1, model.nbody):  # 跳过世界身体 (world body)
            parent_id = model.body_parentid[ii]
            if parent_id == 0:
                # 跳过根身体
                continue
            parent_pos = data.xpos[parent_id]
            child_pos = data.xpos[ii]

            # 计算连杆长度
            length = np.linalg.norm(child_pos - parent_pos)

            # 计算连杆中心位置
            center_pos = (parent_pos + child_pos) / 2

            # 保存信息
            link_info[ii] = {
                'center_joint': child_pos,
                'joint_R': 0,
                'length': length,
                'center_pos': center_pos,
                'link_R': length / 2
            }
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
        collision = check_collision(O1=link_matrix[:6, :3], O2=link_matrix[6:, :3],
                                    R1=link_matrix[:6, 3], R2=link_matrix[6:, 3],
                                    A=link_matrix[:6, 5:8], B=link_matrix[6:, 5:8],
                                    Ra=link_matrix[:6, 8], Rb=link_matrix[6:, 8])
        print(collision)
        # 计算初始姿态与舵机的关系
        # A = my_chain.forward_kinematics(np.append(np.append(np.array(0), data.qpos[:6]), np.array(0)))
        # init_orientation.as_matrix().T.dot(A[:3, :3])
        # B = my_chain.forward_kinematics(ik_joint)

        # 计算实际位置
        # actual_pos = np.eye(4)
        # actual_pos[:3, :3] = data.xmat[7].reshape(3, 3)
        # actual_pos[:3, 3] = [data.xpos[7][1], -data.xpos[7][0], data.xpos[7][2]]

        # 控制舵机运动

        data.ctrl = np.hstack((ik_joint[1:7], ik_joint_r[1:7]))
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)
        i = i + 1
        viewer.sync()
