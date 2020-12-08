#我们在上面代码的基础上添加几行就可以了
from ppgan.apps import DeOldifyPredictor
import numpy as np
import cv2
# 导入记录器
from visualdl import LogWriter
# 初始化一个记录器
with LogWriter(logdir="./log/", file_name='vdlrecords.1607330610.log') as writer:
    # 设置不同的参数组合
    render_params = [4, 8, 16, 32, 64]
    # 我们把原图记录进去，方便对比
    origin_img = cv2.imread('../imgs/test_old.jpg')
    writer.add_image(tag="deoldify/origin",
                            img=cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB),
                            step=0)
    for step in range(5):
        deoldify = DeOldifyPredictor(render_factor=render_params[step])
        # 添加图片数据
        writer.add_image(tag="deoldify/render",
                            img=np.asarray(deoldify.run_image("../imgs/test_old.jpg")),
                            step=step)