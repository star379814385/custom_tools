from custom_tools.cvtools.transform3d import TransformEstimator, eulerAngles2rotationMat, rotationMat2eulerAngles
import numpy as np


if __name__ == "__main__":
    param = np.random.rand(6)
    param[:3] *= 0.01
    # param[[2, 3, 4]] = 0
    # param[3:] *= np.random.randint(-10, 10)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = eulerAngles2rotationMat(*param[:3])

    t[:3, 3] = param[3:]
    print(param)

    n = 1000000
    x = np.random.rand(n, 3)
    y = np.dot(t[:3, :3], x.T).T + t[:3, 3]

    # estimator = TransformEstimator(x, y)
    estimator = TransformEstimator(
        x, y, update_mask=[False, False, True, True, True, False]
    )
    # estimator = TransformEstimator(
    #     x, y, update_mask=[True, True, False, False, False, True]
    # )
    # t_ = estimator.get_trans_by_iter_least_square()
    # t_bursa = estimator.get_trans_by_bursa()
    t_svd = estimator.get_trans_by_svd()
    print(t)
    # print(t_bursa)
    print(t_svd)
    print(np.abs(t - t_svd))