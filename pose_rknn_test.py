from rknn.api import RKNN
import cv2
from utils import preprocess,postprocess

IMG_PATH = "YOUR OWN IMG PATH"
RKNN_PATH = "YOUR OWN MODEL PATH"

if __name__ == '__main__':
    rknn = RKNN()  # 创建RKNN对象

    # 加载rknn模型
    rknn.load_rknn(RKNN_PATH)

    # 初始化运行环境（指定运行环境）
    rknn.init_runtime(
        target='rk3588',
        device_id=None,
        perf_debug=False, # 设置为true可以打开性能评估的debug模式
        eval_mem=False, # 设置为True，表示打开性能内存评估模式
        async_mode=False, # 表示是否打开异步模式
        core_mask=RKNN.NPU_CORE_AUTO, # 设置运行的NPU核心
    )

    # 使用opencv打开要推理的图片
    img = cv2.imread(IMG_PATH)
    image_pad,ratio = preprocess.pose_preprocess(IMG_PATH,640)

    # 使用inference进行推理测试
    outputs=rknn.inference(
        inputs=[image_pad],
        data_format="nhwc",
    )
    print(len(outputs))

    # 推理预测得到结果进行显示
    box = postprocess.pose_postprocess(img, ratio, outputs, 640)
    print("!!",box)

    rknn.release()