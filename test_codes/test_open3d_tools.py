# import time
#
# from custom_tools.cvtools.open3d_tools import PointCloudWrapper
# from custom_tools.cvutils.vis import show_image
# import numpy as np
#
#
# if __name__ == "__main__":
#     pcd_path = r"/home/rainfylee/下载/(1)_20220330095203_pano.pcd_20312fab023adc638dc7df79.pcd"
#     pcd_wrapper = PointCloudWrapper(pcd_path)
#     panorama = pcd_wrapper.get_panorama()
#     # show_image(panorama)
#     normals = pcd_wrapper.get_normals(down_sample=8, return_img=True)
#     # show_image(normals)
#     pcd_points = np.array(pcd_wrapper.point_cloud.points, dtype=np.float32)
#
#
