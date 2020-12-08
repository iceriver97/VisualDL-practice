import cv2
from visualdl import LogWriter

with LogWriter(logdir='./log/', file_name='vdlrecords.1607330610.log') as writer:
    origin_img = cv2.imread('../imgs/animeganv2_test.jpg')
    writer.add_image(tag="anime/origin",
                            img=cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB),
                            step=0)
    result = cv2.imread("../imgs/animeganv2_res.jpg")
    writer.add_image(tag="anime/result",
                            img=cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                            step=0)