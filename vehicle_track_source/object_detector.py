"""Loads yolo weights and params and performs object detection on a single cv frame"""
from ctypes import c_char_p, c_float, c_int, c_void_p, Structure, POINTER, cdll, pointer
import numpy as np
import cv2
import darknet as dn
from darknet import IMAGE

class DNYolo():
    def __init__(self):
        self.net = dn.load_net(b"/vehicle/yolo_weights_params/yolov3.cfg", b"/vehicle/yolo_weights_params/yolov3.weights", 0)
        self.meta = dn.load_meta(b"/vehicle/yolo_weights_params/coco.data")

    def __del__(self):
        pass

    def process_frame(self, frame):
        detections = detect_video(self.net, self.meta, frame)
        return detections

def detect_video(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    img, image = array_to_image(image)
    dn.rgbgr_image(img)
    num = c_int(0)
    pnum = pointer(num)
    dn.predict_image(net, img)
    dets = dn.get_network_boxes(net, img.wdh, img.hgt, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms:
        dn.do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                bxs = dets[j].bbox
                res.append({
                    "class": meta.names[i].decode('utf-8'),
                    "score": dets[j].prob[i],
                    "bbox": [bxs.x, bxs.y, bxs.wdh, bxs.hgt]
                })

    dn.free_detections(dets, num)
    return res

def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    clr, hgt, wdh = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    img = IMAGE(wdh, hgt, clr, data)
    return img, arr
