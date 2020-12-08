#测试 artistic 参数
from ppgan.apps import DeOldifyPredictor
import numpy as np
# 导入记录器
from visualdl import LogWriter
# 初始化一个记录器
with LogWriter(logdir='./log/', file_name='vdlrecords.1607330610.log') as writer:
    # 设置不同的参数组合
    render_params = [4, 8, 16, 32, 64]
    for step in range(5):
        #创建一个开启艺术性的预测器
        deoldify_artistic = DeOldifyPredictor(render_factor=render_params[step], artistic=True)
        # 添加图片数据, 创建不同的标签方便对比
        writer.add_image(tag="deoldify/artistic",
                            img=np.asarray(deoldify_artistic.run_image("../imgs/test_old.jpg")),
                            step=step)