from ppgan.apps import RealSRPredictor
import numpy as np
import cv2
from visualdl import LogWriter

with LogWriter(logdir='./log/', file_name='vdlrecords.1607330610.log') as writer:
    sr = RealSRPredictor()
    origin_img = cv2.imread('../imgs/test_sr.jpg')
    writer.add_image(tag="sr/origin",
                            img=cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB),
                            step=0)
    writer.add_image(tag="sr/result",
                            img=np.asarray(sr.run_image("../imgs/test_sr.jpg")),
                            step=0)