import copy

import numpy as np


def rotationMat2eulerAngles(rotation_matrix):
    # 提取旋转矩阵的元素
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    # 计算欧拉角
    # yaw (绕Z轴旋转)
    thetaz = np.arctan2(r21, r11)

    # pitch (绕Y轴旋转)
    thetay = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))

    # roll (绕X轴旋转)
    thetax = np.arctan2(r32, r33)

    return thetax, thetay, thetaz


def eulerAngles2rotationMat(thetax, thetay, thetaz):
    theta = np.array([thetax, thetay, thetaz], dtype=np.float64)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, cos_theta[0], -sin_theta[0]],
            [0, sin_theta[0], cos_theta[0]],
        ],
        dtype=np.float64,
    )

    R_y = np.array(
        [
            [np.cos(thetay), 0, np.sin(thetay)],
            [0, 1, 0],
            [-np.sin(thetay), 0, np.cos(thetay)],
        ],
        dtype=np.float64,
    )

    R_z = np.array(
        [
            [np.cos(thetaz), -np.sin(thetaz), 0],
            [np.sin(thetaz), np.cos(thetaz), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


def points_trans(points: np.ndarray, trans: np.ndarray):
    points_trans = trans[:3, :3].dot(points.T).T
    points_trans += trans[:3, 3]
    return points_trans


def get_data():
    n = 10000
    x = np.random.rand(n, 3).astype(np.float64)
    param = np.random.rand(3, 3).astype(np.float64)
    param[:, :2] *= 0.0001
    # param[0] = (0, 0, 0)
    trans_list = []
    for thetax, thetay, tz in param:
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = eulerAngles2rotationMat(thetax, thetay, 0)
        t[2, 3] = tz
        trans_list.append(t)

    data = [
        {
            0: copy.deepcopy(x[: n // 2]),
            1: copy.deepcopy(x[: n // 2]),
        },
        {
            1: copy.deepcopy(x[n // 2 :]),
            2: copy.deepcopy(x[n // 2 :]),
        },
    ]
    for i in range(len(data)):
        for k in data[i].keys():
            data[i][k] = points_trans(data[i][k], trans_list[k])

    return data, trans_list


def get_A(points, n_c, c_i):
    A = np.zeros((points.shape[0] * 3, 3 * n_c), dtype=np.float64)
    # thetax
    A[1::3, 3 * c_i] = -points1[:, 2]
    A[2::3, 3 * c_i] = points1[:, 1]
    # thetay
    A[0::3, 3 * c_i + 1] = points[:, 2]
    A[2::3, 3 * c_i + 1] = -points[:, 0]
    # tz
    A[2::3, 3 * c_i + 2] = 1
    return A


if __name__ == "__main__":
    data, trans_list = get_data()
    cids = set()
    for d in data:
        cids = cids.union(set(d.keys()))
    n_c = len(cids)

    A_list = []
    B_list = []
    for i in range(len(data)):
        c_is = list(data[i].keys())
        assert len(c_is) == 2
        c_i1 = c_is[0]
        c_i2 = c_is[1]
        points1 = data[i][c_i1]
        points2 = data[i][c_i2]
        A1 = get_A(points1, n_c, c_i1)
        A2 = get_A(points2, n_c, c_i2)
        A = A2 - A1
        B1 = points1.flatten()
        B2 = points2.flatten()
        B = B2 - B1
        A_list.append(A)
        B_list.append(B)
    A = np.concatenate(A_list, axis=0)
    B = np.concatenate(B_list, axis=0)
    # print(np)

    param = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(B))
    new_trans_list = []
    for thetax, thetay, tz in param.reshape(-1, 3):
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = eulerAngles2rotationMat(thetax, thetay, 0)
        t[2, 3] = tz
        new_trans_list.append(t)

    g_t = trans_list[0].dot(np.linalg.pinv(new_trans_list[0]))
    # g_t = np.linalg.pinv(new_trans_list[0])
    for i in range(len(new_trans_list)):
        new_trans_list[i] = np.dot(g_t, new_trans_list[i])

    for i in range(len(trans_list)):
        print("-" * 20)
        print(i)
        print(trans_list[i])
        print(new_trans_list[i])
