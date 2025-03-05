# RK3588-mmpose-mmdet
使用RK3588开发板对mmpose、mmdet模型进行部署。包含了模型转换和算子修改等内容

# RK3588开发板部署mm系列模型

​	使用RK3588开发板对mmpose、mmdet框架训练的代码进行部署。具体的代码解释和流程描述写于博客：[RK3588部署MM系列模型全流程-CSDN博客](https://blog.csdn.net/qq_50991821/article/details/146021703?spm=1001.2014.3001.5501)

​	项目结构为：

**——ONNX**：用来存放pth转换得到的onnx模型

**——res**：用来测试和量化所使用的图像资源

**——RKNN**：用来存放onnx转换得到的rknn模型

**——utils**：存放预处理和后处理的具体代码

​		**——onnx2rknn.py**：onnx模型转换rknn模型代码

​		**——det_rknn_test.py**：测试mmdet的rknn模型效果

​		**——pose_rknn_test.py**： 测试mmpose的rknn效果

​		**——predict.py**：结合项目代码最终对整体模型进行效果测试，包含了可视化

​		**——clip_onnx.py**：进行模型算子修改代码

​		**——deploy.py**：最终进行开发板部署所使用的代码

​	由于是实验室的产出，所以模型涉密，**在此不把模型和图片进行上传**。但是只需要按照博客中的步骤就可以获得onnx模型，进而完善代码。

​	具体使用过程中请把源码中涉及到的路径修改成自己模型和图片的路径，然后对自己的具体任务，可以修改predict中的可视化。

