import argparse
import inspect
import this

import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
    parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
    parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
    parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
    parser.add_argument('--face_detection_model', '-fd', type=str, default='resources/models/face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
    parser.add_argument('--face_recognition_model', '-fr', type=str, default='resources/models/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
    return parser.parse_args()

def draw_rectangle(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            # cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            # cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            # cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            # cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            # cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    if fps is None:
        return
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

mydict = ['dat', 'loi']

if __name__ == '__main__':
    pass