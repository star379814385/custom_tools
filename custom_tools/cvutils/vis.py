import cv2


def show_image(img, winname="img", win_flag=cv2.WINDOW_KEEPRATIO, wait_delay=0):
    cv2.namedWindow(winname, win_flag)
    cv2.imshow(winname, img)
    cv2.waitKey(wait_delay)
    try:
        cv2.destroyWindow(winname)
    except:
        pass
