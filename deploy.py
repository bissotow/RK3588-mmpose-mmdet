from rknnlite.api import RKNNLite
import cv2

from utils import preprocess,postprocess

IMG_PATH = "YOUR OWN IMG PATH"
BOX_PATH = "YOUR OWN MODEL PATH"
POSE_PATH = "YOUR OWN MODEL PATH"

def load_model(rknn,Model_Path,IMAGE):
    # 加载rknn模型
    rknn.load_rknn(Model_Path)

    # 初始化运行环境（指定运行环境）
    rknn.init_runtime()

    # 使用inference进行推理测试
    outputs = rknn.inference(
        inputs=[IMAGE],
        data_format="nhwc",
    )
    print(len(outputs))

    return outputs




if __name__ == '__main__':
    box_rknn = RKNNLite()  # 创建RKNN对象
    pose_rknn = RKNNLite()  # 创建RKNN对象

    # 使用opencv打开要推理的图片
    img = cv2.imread(IMG_PATH)
    image_pad, ratio = preprocess.bbox_preprocess(IMG_PATH, 640)

    box_result = load_model(box_rknn,BOX_PATH,image_pad)
    # 推理预测得到结果进行显示
    box = postprocess.bbox_postprocess(img, ratio, box_result, 640)

    x, y = [], []  # 记录点坐标的列表

    for bbox in box:
        x1,y1,x2,y2 = bbox
        det_image = img[int(y1):int(y2),int(x1):int(x2)]
        #cv2.imwrite("box.jpg", det_image)

        pose_image, center, scale = preprocess.pose_preprocess(det_image,(512,512))
        #pose_image = [pose_image.transpose(2, 0, 1)]

        pose_result = load_model(pose_rknn, POSE_PATH, pose_image)

        kpts,score  = postprocess.pose_postprocess(pose_result,(512,512),center,scale)
        print(kpts)

        #cv2.imshow("Result", det_image)
        #if cv2.waitKey(0) == 0x1B:
        #    cv2.destroyAllWindows()


    # 推理预测得到结果进行显示
        # 画点操作(找5个点),只有5个点才可以处理
        if len(kpts[0]) == 5:
            for i in range(len(kpts[0])):
                px = kpts[0][i][0]
                py = kpts[0][i][1]
                print(px)
                cv2.circle(img, (int(px + x1), int(py + y1)), 2, (0, 0, 255), cv2.FILLED)
                x.append(int(px + x1))
                y.append(int(py + y1))
        else:  # 一个框里并没有五个点,证明该检测框识别失败
            print("error!!")
            break
    cv2.imwrite("YOUR OWN IMG PATH", img)

    box_rknn.release()
    pose_rknn.release()
