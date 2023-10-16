import numpy as np
from scipy.optimize import leastsq
import math

def eulerAngles2rotationMat(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ],
        dtype=np.float64,
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ],
        dtype=np.float64,
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def compute_transform(src_points, dst_points, params0=None, variable_mask=None):
    """
    使用最小二乘法计算变换欧拉角和平移量
    :param src_points: 点云1，N x 3的numpy数组，每行代表一个点的XYZ坐标
    :param dst_points: 点云2，N x 3的numpy数组，每行代表一个点的XYZ坐标
    :return: 变换矩阵T，包括旋转矩阵R和平移向量t
    """
    # 初始参数值
    if params0 is None:
        params0 = np.zeros(6)

    # 定义误差函数，即计算点云2相对于点云1的残差
    def residuals(params, src, dst):
        if not variable_mask:
            for i, flag in enumerate(params):
                if not flag:
                    params[i] = params0[i]
        thetax, thetay, thetaz, tx, ty, tz = params

        # 构建旋转矩阵
        R = eulerAngles2rotationMat((thetax, thetay, thetaz))

        # 构建平移向量
        t = np.array([tx, ty, tz])

        # 计算点云2经过变换后的坐标
        transformed_src = R.dot(src.T).T + t

        # 计算残差（点云2与变换后的点云2之间的欧氏距离）
        error = np.linalg.norm(transformed_src - dst, axis=1)

        return error


    # 最小二乘法求解
    result = leastsq(residuals, params0, args=(src_points, dst_points))

    # 提取结果
    params_opt = result[0]
    roll, pitch, yaw, tx, ty, tz = params_opt

    # 构建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)

    # 构建平移向量
    t = np.array([tx, ty, tz])

    # 构建变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

# if __name__ == "__main__":
#     from line_profiler import LineProfiler
#
#     x = np.random.rand(100000, 3)
#     thetax = np.pi * 0.1
#     thetay = np.pi * 0.2
#     thetaz = np.pi * 0.3
#     t = eulerAngles2rotationMat(
#         np.array([thetax, thetay, thetaz], dtype=np.float64),
#     )
#     # print(t.flatten())
#     init_t = np.zeros((3,), dtype=np.float64)  # theta xyz
#     y = t.dot(x.T).T
#
#
#     t_ = compute_transform(x, y)
#
#     print(t)
#     print(t_)