from typing import List, Tuple
import numpy as np
import cv2
import os


label_names = ["skeleton",    ]

colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(label_names))]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_multi_level_priors(featmap_sizes, strides, offset=0):
    multi_level_priors = []
    for level_idx in range(len(featmap_sizes)):
        feat_h, feat_w = featmap_sizes[level_idx]
        stride_w, stride_h = strides[level_idx], strides[level_idx]
        shift_x = (np.arange(0, feat_w) + offset) * stride_w
        shift_y = (np.arange(0, feat_h) + offset) * stride_h
        shift_xx, shift_yy = np.meshgrid(shift_x, shift_y)
        shift_xx, shift_yy = shift_xx.reshape(-1), shift_yy.reshape(-1)
        stride_w = np.full((shift_xx.shape[0], ), stride_w)
        stride_h = np.full((shift_yy.shape[0], ), stride_h)
        priors = np.stack([shift_xx, shift_yy, stride_w, stride_h],
                                axis=-1)
        multi_level_priors.append(priors)
    return multi_level_priors


def decoded_bbox(bbox_preds, priors):
    # 确保输入为二维数组
    bbox_preds = bbox_preds.reshape(-1, 4)
    priors = priors.reshape(-1, 4)

    # 分离先验框的中心点和宽高
    prior_centers = priors[:, :2]  # (cx, cy)
    # 分离预测的距离值
    l, r, t, b = bbox_preds[:, 0], bbox_preds[:, 1], bbox_preds[:, 2], bbox_preds[:, 3]

    # 计算左上角和右下角坐标
    x1 = prior_centers[:, 0] - l
    y1 = prior_centers[:, 1] - t
    x2 = prior_centers[:, 0] + r
    y2 = prior_centers[:, 1] + b

    # 拼接解码后的边界框
    decoded_bboxes = np.stack([x1, y1, x2, y2], axis=-1)

    return decoded_bboxes


def nms(boxes, scores, iou_threshold):
    # 如果没有检测到任何框，直接返回空
    if len(boxes) == 0:
        return []

    # 将框的坐标提取出来
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按照得分从高到低排序，并得到排序后的索引
    order = scores.argsort()[::-1]

    keep = []  # 用于存储保留下来的框的索引

    while order.size > 0:
        # 当前得分最高的框的索引
        i = order[0]
        keep.append(i)

        # 计算当前框与其他所有框的交集区域的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算交并比 (IoU)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的框的索引
        inds = np.where(iou <= iou_threshold)[0]

        # 更新排序后的索引，保留那些 IoU 小于阈值的框
        order = order[inds + 1]

    return keep


def per_class_nms(boxes, scores, labels, iou_threshold=0.5):
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []

    for label in unique_labels:
        # 获取当前类别的所有框、得分和标签
        label_mask = labels == label
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]

        # 对当前类别执行 NMS
        keep = nms(label_boxes, label_scores, iou_threshold)

        # 保留 NMS 后的框、得分和标签
        final_boxes.append(label_boxes[keep])
        final_scores.append(label_scores[keep])
        final_labels.extend([label] * len(keep))

    # 将结果合并成单个数组
    final_boxes = np.concatenate(final_boxes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_labels = np.array(final_labels)

    return final_boxes, final_scores, final_labels


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    # mask = max_val_x > max_val_y
    # max_val_x[mask] = max_val_y[mask]
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def convert_coco_to_openpose(keypoints, scores):
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)

    # neck score when visualizing pred
    neck[:,
         2:3] = np.where(keypoints_info[:, 5, 2:3] > keypoints_info[:, 6, 2:3],
                         keypoints_info[:, 6, 2:3], keypoints_info[:, 5, 2:3])
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
    return keypoints, scores

def bbox_postprocess(img,ratio,outputs,INPUT_SIZE=640):
    # 参数设置
    box = []
    input_size = (INPUT_SIZE, INPUT_SIZE)
    strides = (8, 16, 32)
    score_thr = 0.5  # 置信度分数阈值，越高越严格
    iou_threshold = 0.28  # IoU阈值，用于NMS抑制重叠框，越低越严格

    flatten_outputs = []
    for out in outputs:
        # 检查形状，防止出错
        if len(out.shape) == 4:  # 确保有 4 维
            reshaped = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3]).transpose(0, 2, 1)
            flatten_outputs.append(reshaped)
        else:
            print(f"Skipping element with unexpected shape: {out.shape}")
    print(len(flatten_outputs))  # 检查结果列表长度

    # 处理输出结果
    # flatten_outputs = [outputs.reshape((out.shape[0], out.shape[1], out.shape[2] * out.shape[3])).transpose(0, 2, 1) for
    #                    out in outputs]

    flatten_cls_scores = np.concatenate(flatten_outputs[0:3], axis=1)
    flatten_cls_scores = sigmoid(flatten_cls_scores)
    flatten_bbox_preds = np.concatenate(flatten_outputs[3:6], axis=1)

    # 生成先验网格
    featmap_sizes = [(input_size[0] // stride, input_size[1] // stride) for stride in strides]
    multi_level_priors = get_multi_level_priors(featmap_sizes, strides)  # 生成每个尺度层级上的先验网格
    flatten_priors = np.concatenate(multi_level_priors)

    # 解码bbox + xywh2xyxy
    decoded_bboxes = decoded_bbox(flatten_bbox_preds,
                                  flatten_priors)  # [1, N, 4], each representing [x1, y1, x2, y2]

    # 对 scores 进行阈值过滤
    flatten_cls_scores = flatten_cls_scores[0]

    labels = np.argmax(flatten_cls_scores, 1)
    max_scores = flatten_cls_scores[np.arange(labels.shape[0]), labels]
    scores = max_scores
    valid_mask = scores >= score_thr

    print(f"decoded_bboxes shape: {decoded_bboxes.shape}")  # 应为 (N, 4)
    print(f"valid_mask shape: {valid_mask.shape}")  # 应为 (N,)

    filtered_bboxes = decoded_bboxes[valid_mask]  # [N', 4]
    filtered_scores = scores[valid_mask]  # [N']
    filtered_labels = labels[valid_mask]  # [N']

    # rescale bboxes
    filtered_bboxes = filtered_bboxes / ratio

    # 非极大值抑制 (NMS)
    final_boxes, final_scores, final_labels = per_class_nms(filtered_bboxes, filtered_scores, filtered_labels,
                                                            iou_threshold=iou_threshold)

    #print(f'过滤前：{len(scores)}, 阈值过滤：{len(filtered_scores)}, NMS过滤：{len(final_scores)}')



    # 显示结果
    for bbox, score, label in zip(final_boxes, final_scores, final_labels):
        # bbox[0::2], bbox[1::2] = np.clip(bbox[0::2], 0, input_size[1]), np.clip(bbox[1::2], 0, input_size[0])   # 防止检测框超出边界
        #x1, y1, x2, y2 = bbox
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[label], 2)
        #cv2.putText(img, f"{label_names[label]}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 0, 255), 2)
        #print(bbox)
        box.append(bbox)
        #print(box)
    return box


def pose_postprocess(
        outputs: List[np.ndarray],
        model_input_size:(512,512),
        center: Tuple[int, int],
        scale: Tuple[int, int],
        simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores




