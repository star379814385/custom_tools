from custom_tools.aitools.labelme_tools.urpc2labelme import urpc2labelme

if __name__ == "__main__":
    urpc2labelme(
        label_dir=r"D:\personal_dataset\URPC_opticalimage_dataset\testA\box",
        img_dir=r"D:\personal_dataset\URPC_opticalimage_dataset\testA\image",
        img_pattern="*.jpg",
    )