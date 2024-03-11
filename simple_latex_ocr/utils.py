# -*- encoding: utf-8 -*-
# @Author: chaodreaming
# @Contact: chaodreaming@gmail.com
from io import BytesIO
from typing import List
from typing import Union

import cv2
import numpy as np
# import torch
# from albumentations.pytorch import ToTensorV2
from PIL import Image

from .utils_load import OrtInferSession


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok,clean_up_tokenization_spaces=False) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('Ċ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]',
                                                                                                          '').strip()
            for detok in dec]
def determine_alignment(boxes, threshold=5):
    """
    判断矩形框是居中对齐、左对齐还是右对齐。

    :param boxes: 矩形框列表，每个元素为[x, y, w, h]
    :param threshold: 对齐判定的阈值
    :return: 返回矩形框的对齐方式：'左对齐'、'右对齐'或'居中对齐'
    """
    # 提取所有框的左边界（x坐标）和右边界（x + w坐标）
    left_edges = [x for x, _, w, _ in boxes]
    right_edges = [x + w for x, _, w, _ in boxes]

    try:
        # 计算左边界和右边界的最大和最小值，判断是否在阈值范围内
        left_range = max(left_edges) - min(left_edges)
        right_range = max(right_edges) - min(right_edges)

        if left_range <= threshold:
            return 'l'
        elif right_range <= threshold:
            return 'r'
        else:
            return 'c'
    except:
        return "c"
class PreProcess:
    def __init__(self,detect_path, max_dims: List[int], min_dims: List[int]):
        self.detecter=OrtInferSession(detect_path)
        self.max_dims, self.min_dims = max_dims, min_dims
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)


    def letterbox(self, im, new_shape=(128, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        # if shape[0] > shape[1]:
        #     max_wh = max(new_shape[0], new_shape[1])
        #     new_shape = (max_wh, max_wh)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        # return im, ratio, (dw, dh)

        return im

    def pad(self, img: Image.Image, divable: int = 32) -> Image.Image:
        """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

        Args:
            img (PIL.Image): input image
            divable (int, optional): . Defaults to 32.

        Returns:
            PIL.Image
        """
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)

        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b: b + h, a: a + w]
        im = Image.fromarray(rect).convert("L")
        dims: List[Union[int, int]] = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))

        padded = Image.new("L", tuple(dims), 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def detect_image(self, input_image):
        # original_image: np.ndarray = cv2.imread(input_image)
        if type(input_image) == str:
            source_image=Image.open(input_image)
        elif type(input_image)==bytes:
            source_image=Image.open(BytesIO(input_image))
        else:
            source_image =Image.open(BytesIO(input_image.read()))
        original_image: np.ndarray = cv2.cvtColor(np.array(source_image),
                                                  cv2.COLOR_RGB2BGR)

        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640
        ##################################################
        # 图像预处理，和cv2.dnn.blobFromImage近似操作
        image = cv2.resize(image, (640, 640))  # 调整图像大小
        image = image.astype(np.float32) / 255.0  # 归一化
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR到RGB
        image = image.transpose(2, 0, 1)  # 重排数组顺序 HWC到CHW
        image = np.expand_dims(image, axis=0)  # 添加维度，以符合模型的输入
        # image=image.astype(np.float16)

        # model_inputs = onnx_model.get_inputs()
        # outputs = onnx_model.run(None, {model_inputs[0].name: image})[0]
        outputs = self.detecter([image])[0].astype(np.float32)

        # blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        # onnx_model.setInput(blob)
        # outputs = onnx_model.forward()
        ##########################################################
        outputs = np.array([cv2.transpose(outputs[0])])  # 1 8400 6
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.45:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.45, 0.5)
        sorted_result_boxes = sorted(result_boxes, key=lambda i: boxes[i][1] * scale)  # 根据NMS结果中的框的y坐标进行排序
        images = []
        ########################################################显示
        # detections = []
        # for i in range(len(result_boxes)):
        #     index = result_boxes[i]
        #     box = boxes[index]
        #     detection = {
        #         'class_id': class_ids[index],
        #         'class_name': CLASSES[class_ids[index]],
        #         'confidence': scores[index],
        #         'box': box,
        #         'scale': scale}
        #     detections.append(detection)
        #     draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale),
        #                       round(box[1] * scale),
        #                       round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        #
        #     print(detections)
        #     original_image = cv2.resize(original_image, (320, 420))
        # cv2.imshow('image', original_image)
        # cv2.waitKey(0)
        #########################################################
        for i in range(len(sorted_result_boxes)):
            index = sorted_result_boxes[i]
            box = boxes[index]
            # detection = {
            #     'class_id': class_ids[index],
            #     'class_name': CLASSES[class_ids[index]],
            #     'confidence': scores[index],
            #     'box': box,
            #     'scale': scale}
            # detections.append(detection)
            x, y, w, h = round(box[0] * scale), round(box[1] * scale), round(box[2] * scale), round(box[3] * scale)

            # boxes = [[100, 120, 50, 80], [102, 115, 48, 78], [104, 118, 52, 82]]

            cropped_image = source_image.crop((x, y, x + w, y + h))
            images.append(cropped_image)

        alignment = determine_alignment(boxes, threshold=5)

        return images, alignment

    def minmax_size(
        self,
        img: Image.Image,
    ) -> Image.Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.

        Returns:
            Image: Image with correct dimensionality
        """
        if self.max_dims is not None:
            ratios = [a / b for a, b in zip(img.size, self.max_dims)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(size.astype(int), Image.BILINEAR)

        if self.min_dims is not None:
            padded_size: List[Union[int, int]] = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, self.min_dims)
            ]

            new_pad_size = tuple(padded_size)
            if new_pad_size != img.size:  # assert hypothesis
                padded_im = Image.new("L", new_pad_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def normalize(self, img: np.ndarray, max_pixel_value=255.0) -> np.ndarray:
        mean = self.mean * max_pixel_value
        std = self.std * max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    @staticmethod
    def to_gray(img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def transpose_and_four_dim(img: np.ndarray) -> np.ndarray:
        return img.transpose(2, 0, 1)[:1][None, ...]