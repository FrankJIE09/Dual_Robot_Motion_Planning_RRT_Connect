import numpy as np
# 碰撞检测函数
def check_collision(O1, O2, A, B, R1, R2, Ra, Rb):
    num_links = O1.shape[0]

    # 主机械臂的连接杆与关节之间的碰撞检测
    # for i in range(num_links):
    #     for j in range(num_links):
    #         if i != j:
    #             if np.linalg.norm(O1[i] - O1[j]) < (R1[i] + R1[j]):
    #                 print(f"Collision detected between O1[{i}] and O1[{j}]")
    #                 return 1
    #             if np.linalg.norm(O1[i] - A[j]) < (R1[i] + Ra[j]):
    #                 print(f"Collision detected between O1[{i}] and A[{j}]")
    #                 return 1
    #
    # # 从机械臂的连接杆与关节之间的碰撞检测
    # for i in range(num_links):
    #     for j in range(num_links):
    #         if i != j:
    #             if np.linalg.norm(O2[i] - O2[j]) < (R2[i] + R2[j]):
    #                 print(f"Collision detected between O2[{i}] and O2[{j}]")
    #                 return 1
    #             if np.linalg.norm(O2[i] - B[j]) < (R2[i] + Rb[j]):
    #                 print(f"Collision detected between O2[{i}] and B[{j}]")
    #                 return 1

    # 主机械臂与从机械臂之间的碰撞检测
    for i in range(num_links):
        for j in range(num_links):
            # if np.linalg.norm(O1[i] - O2[j]) < (R1[i] + R2[j]):
            #     print(f"Collision detected between O1[{i}] and O2[{j}]")
            #     return 1
            # if np.linalg.norm(O1[i] - B[j]) < (R1[i] + Rb[j]):
            #     print(f"Collision detected between O1[{i}] and B[{j}]")
            #     return 1
            if np.linalg.norm(A[i] - O2[j]) < (Ra[i] + R2[j]):
                print(f"Collision detected between A[{i}] and O2[{j}]")
                return 1
            if np.linalg.norm(A[i] - B[j]) < (Ra[i] + Rb[j]):
                print(f"Collision detected between A[{i}] and B[{j}]")
                return 1

    # 如果没有检测到碰撞，返回 0
    return 0