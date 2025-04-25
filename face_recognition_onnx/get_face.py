import argparse
import numpy as np
import cv2 as cv
from utils import get_parser, draw_rectangle

if __name__ == '__main__':
    user = 'you'
    args = get_parser()
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,"")

    tm = cv.TickMeter()

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 1
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        key = cv.waitKey(1)
        if key == 27:
            break

        if key == ord('s') or key == ord('S'):
            if faces[1] is not None:
                face_align = recognizer.alignCrop(frame, faces[1][0])
                file_name = (f'resources/Images/faces/{user}/{user}_%04d.bmp') % dem
                cv.imwrite(file_name, face_align)
                dem = dem + 1
        # Draw results on the input image
        draw_rectangle(frame, faces, tm.getFPS())

        # Visualize results
        cv.imshow('Live', frame)
    cv.destroyAllWindows()
