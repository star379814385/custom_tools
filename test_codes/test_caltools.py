from custom_tools.aitools import cal_bbox_ious, cal_mask_ious
import numpy as np


def test_bbox_ious():
    bbox_mx4 = np.array([[0, 0, 512, 512]])
    bbox_nx4 = np.array(
        [
            [0, 0, 256, 256],
            [512, 512, 567, 678],
        ]
    )
    ious = cal_bbox_ious(bbox_mx4, bbox_nx4)
    assert ious.shape == (1, 2)
    assert ious[0, 0] == 0.25
    assert ious[0, 1] == 0

def test_mask_ious():
    mask1 = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.bool_)
    mask2 = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.bool_)
    ious = cal_mask_ious(mask1, mask2)
    assert ious.shape == (2, 2)
    results = [1, 1 / 3, 0, 1 / 3]
    for iou, result in zip(ious.flatten(), results):
        assert abs(iou - result) < 1e-4, (iou, result)

if __name__ == "__main__":
    # test_bbox_ious()
    # test_mask_ious()
    from pycocotools.mask import encode, decode

    import sys
    a = np.random.random((1024, 1024, 3))
    print(a.shape)
    print(sys.getsizeof(a)) # 8388736
    b = np.asfortranarray(a, dtype=np.bool_)
    print(sys.getsizeof(b)) # 1048704
    c = encode(b)
    print(sys.getsizeof(c)) # 232
    d = decode(c)
    print(d.shape)
    print(d.dtype)
    print(type(d))
    assert b.all() == d.all()
    print(sys.getsizeof(d.astype(np.bool_))) # 232
