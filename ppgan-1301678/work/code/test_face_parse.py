from ppgan.apps import FaceParsePredictor
import numpy as np
import cv2
from visualdl import LogWriter

with LogWriter(logdir='./log/', file_name='vdlrecords.1607330610.log') as writer:
    parser = FaceParsePredictor()
    origin_img = cv2.imread('../imgs/test_face_parse.png')
    writer.add_image(tag="test_face_parse/origin",
                            img=cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB),
                            step=0)
    writer.add_image(tag="test_face_parse/result",
                            img=np.asarray(parser.run('../imgs/test_face_parse.png')),
                            step=0)