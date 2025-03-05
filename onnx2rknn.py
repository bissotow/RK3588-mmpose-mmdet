from rknn.api import RKNN
import cv2
import numpy as np


if __name__ == '__main__':
    rknn = RKNN(verbose=True,verbose_file="log.txt")  # 创建RKNN对象，为了后面的操作。parm1表示打印详细日志;parm2表示保存到对应路径

    # 具体的应用：配置RKNN config 用于设置转换的参数设置（一般只修改mean_values和std_values和platform平台即可）
    rknn.config(
        mean_values= [[0,0,0]], # 表示预处理减去的均值化参数
        std_values= [[255,255,255]], # 表示预处理除的标准化参数
        quantized_dtype="asymmetric_quantized-8", # 表示量化类型
        quantized_algorithm='normal', # 表示量化算法
        quantized_method='channel', # 表示量化方式
        quant_img_RGB2BGR= False, # 是否转换格式
        target_platform="rk3588", # 运行的平台
        float_dtype="float16", # RKNN默认的副点数类型
        optimization_level=3, # 表示模型优化等级
        custom_string="this is my rknn model for 3588", # 添加的自定义信息
        remove_weight=False, # 去权重的从模型
        compress_weight=False, # 压缩模型权重，减小模型体积
        inputs_yuv_fmt=None, # 输入数据的YUV格式
        single_core_mode=False, # 表示构建RKNN模型在单核心模式，只用于RK3588
    )

    # 加载onnx模型
    rknn.load_onnx(
        model="YOUR OWN MODEL PATH", # 表示加载模型的路径
        input_size_list=[[1,3,512,512]], # 表示模型输入图片的个数、尺寸和通道数（分别是batch、channel、img_size）
    )

    # 使用build接口构建rknn模型
    rknn.build(
        do_quantization=False,    # 是否做量化操作（量化——降低模型复杂性）
        #dataset="./res/dataset.txt",     # dataset 表示要量化的图片集（去找对应输入图片的）
    )

    # 导出rknn模型
    rknn.export_rknn(
        export_path="YOUR OWN MODEL PATH" # 表示导出rknn的保存路径
    )

    # 初始化运行环境（指定运行环境）
    rknn.init_runtime(
        target=None,
        device_id=None,
        perf_debug=False, # 设置为true可以打开性能评估的debug模式
        eval_mem=False, # 设置为True，表示打开性能内存评估模式
        async_mode=False, # 表示是否打开异步模式
        core_mask=RKNN.NPU_CORE_AUTO, # 设置运行的NPU核心
    )

    # 使用opencv打开要推理的图片
    img_2 = cv2.imread("YOUR OWN IMG PATH")
    # 转换编码格式。由于cv2读图片会转换成rgb，所以需要转化
    cv2.cvtColor(src=img_2, code=cv2.COLOR_BGR2RGB)
    # resize
    img_2 = cv2.resize(img_2,(512,512))
    # 使用inference进行推理测试
    outputs=rknn.inference(
        inputs=[img_2],
        data_format="nhwc",
    )
    print(len(outputs))
    print(outputs[0])


    rknn.release()