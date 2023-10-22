import numpy as np
import copy
from scipy.optimize import leastsq
from typing import Optional, Sequence


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


class TransformEstimator:
    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        init_param: Optional[Sequence] = None,
        update_mask: Optional[Sequence] = None,
    ):
        self.src_points = np.array(src_points, dtype=np.float64)
        self.dst_points = np.array(dst_points, dtype=np.float64)
        if init_param is None:
            init_param = np.zeros((6,), dtype=np.float64)
        else:
            init_param = np.array(init_param, dtype=np.float64)
        assert init_param.ndim == 1 and init_param.shape[0] == 6
        self.init_param = init_param
        if update_mask is None:
            update_mask = np.ones_like(init_param, dtype=np.bool_)
        else:
            update_mask = np.array(update_mask, dtype=np.bool_)
        assert update_mask.shape == init_param.shape
        self.update_mask = update_mask

    def get_residuals(self):
        init_param = copy.deepcopy(self.init_param)
        update_mask = copy.deepcopy(self.update_mask)

        def residuals(params, src, dst):
            full_param = copy.deepcopy(init_param)
            full_param[update_mask] = params

            thetax, thetay, thetaz, tx, ty, tz = full_param

            # 构建旋转矩阵
            R = eulerAngles2rotationMat(thetax, thetay, thetaz)

            # 构建平移向量
            t = np.array([tx, ty, tz])

            # 计算点云2经过变换后的坐标
            transformed_src = R.dot(src.T).T + t

            # 计算残差（点云2与变换后的点云2之间的欧氏距离）
            error = np.linalg.norm(transformed_src - dst, axis=1)

            return error

        return residuals

    def get_trans_by_iter_least_square(self):
        # 利用最小二乘进行非线性参数优化（迭代求解）

        full_param = copy.deepcopy(self.init_param)
        params0 = full_param[self.update_mask]
        # 最小二乘法求解
        result = leastsq(
            self.get_residuals(),
            params0,
            args=(copy.deepcopy(self.src_points), copy.deepcopy(self.dst_points)),
        )

        # 提取结果
        full_param[self.update_mask] = result[0]
        thetax, thetay, thetaz, tx, ty, tz = full_param

        # 构建旋转矩阵
        R = eulerAngles2rotationMat(thetax, thetay, thetaz)

        # 构建平移向量
        t = np.array([tx, ty, tz])

        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def get_trans_by_bursa(self):
        # 基于bursa建模
        # 只能在欧拉角角度较小时使用
        src_points = copy.deepcopy(self.src_points)
        dst_points = copy.deepcopy(self.dst_points)
        n = self.src_points.shape[0]
        # 确定需要优化的参数
        param_ids = [i for i, flag in enumerate(self.update_mask) if flag]
        A = np.zeros((n * 3, len(param_ids)), dtype=np.float64)
        B = dst_points.flatten() - src_points.flatten()

        for i, param_id in enumerate(param_ids):
            if param_id == 0:
                # thetax
                A[1::3, i] = -src_points[:, 2]
                A[2::3, i] = src_points[:, 1]
            elif param_id == 1:
                # thetay
                A[0::3, i] = src_points[:, 2]
                A[2::3, i] = -src_points[:, 0]
            elif param_id == 2:
                A[0::3, i] = -src_points[:, 1]
                A[1::3, i] = src_points[:, 0]
            elif 3 <= param_id <= 5:
                # tx, ty, tz
                A[param_id - 3 :: 3, i] = 1
            else:
                raise ValueError

        w = np.zeros((6,), dtype=np.float64)
        w[self.update_mask] = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(B))
        thetax, thetay, thetaz, tx, ty, tz = w
        # todo: 如果计算出来的欧拉角角度较大，提供警告

        trans = np.eye(4, dtype=np.float64)
        trans[:3, :3] = eulerAngles2rotationMat(thetax, thetay, thetaz)
        trans[:3, 3] = (tx, ty, tz)

        return trans

    def get_trans_by_svd(self):
        src_points = copy.deepcopy(self.src_points)
        dst_points = copy.deepcopy(self.dst_points)

        # 1.计算质心
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)

        # 2.去中心化
        src_points = src_points - src_centroid
        dst_points = dst_points - dst_centroid

        # 构建协方差矩阵
        cov_matrix = src_points.T @ dst_points

        # 奇异值分解
        U, S, Vt = np.linalg.svd(cov_matrix)

        # 计算旋转矩阵
        R = Vt.T @ U.T
        # 处理反射变换
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        theta = np.array(rotationMat2eulerAngles(R))
        theta[~self.update_mask[:3]] = 0
        R = eulerAngles2rotationMat(*theta)


        # 计算平移向量，并将X和Y坐标设置为0
        t = np.zeros(3, dtype=np.float64)
        # print((dst_centroid - src_centroid)[self.update_mask[3:6]])
        t[self.update_mask[3:6]] = (dst_centroid - src_centroid)[self.update_mask[3:6]]
        # if self.update_mask[3]:
        # t[2] = -src_centroid[2] + dst_centroid[2]

        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T



# if __name__ == "__main__":
#     param = np.random.rand(6)
#     param[:3] *= 0.01
#     # param[3:] *= np.random.randint(-10, 10)
#     t = np.eye(4, dtype=np.float64)
#     t[:3, :3] = eulerAngles2rotationMat(*param[:3])
#     t[:3, 3] = param[3:]
#     print(param)
#
#     n = 1000000
#     x = np.random.rand(n, 3)
#     y = np.dot(t[:3, :3], x.T).T + t[:3, 3]
#
#     estimator = TransformEstimator(x, y)
#     # estimator = TransformEstimator(
#     #     x, y, update_mask=[False, False, True, True, True, True]
#     # )
#     # t_ = estimator.get_trans_by_iter_least_square()
#     t_ = estimator.get_trans_by_bursa()
#     print(t)
#     print(t_)
