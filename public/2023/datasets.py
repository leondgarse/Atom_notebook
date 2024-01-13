import os
import math
import random
import cv2
import numpy as np


def load_from_custom_json(data_path, with_info=False):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)
    test_key = "validation" if "validation" in aa else "test"
    train, test, info = aa["train"], aa[test_key], aa.get("info", {})
    total_images, num_classes = len(train), info.get("num_classes", 0)
    if num_classes <= 0:
        num_classes = max([max([int(jj) for jj in ii["objects"]["label"]]) for ii in train]) + 1
        print(">>>> Using max value from train as num_classes:", num_classes)

    if "base_path" in info and len(info["base_path"]) > 0:
        base_path = os.path.expanduser(info["base_path"])
        for ii in train:
            ii["image"] = os.path.join(base_path, ii["image"])
        for ii in test:
            ii["image"] = os.path.join(base_path, ii["image"])
    return train, test


def load_image(image_file, target_size=640, use_augment=True):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = cv2.imread(image_file)  # BGR

    h0, w0 = img.shape[:2]  # orig hw
    r = target_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not use_augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, h0, w0  # img, hw_original

def imread(image_path, image_size):
    from PIL import Image, ExifTags
    return Image.open(image_path).convert("RGB").resize([image_size, image_size], resample=Image.Resampling.BICUBIC)

def combine_mosaic(images, bboxes, labels, target_size=640):
    # loads images in a mosaic, bboxes: [top, left, bottom, right]
    mosaic_border = target_size // 2
    hh_center = int(random.uniform(mosaic_border, 2 * target_size - mosaic_border))
    ww_center = int(random.uniform(mosaic_border, 2 * target_size - mosaic_border))

    paste_border = target_size * 2
    mosaic_image = np.full((paste_border, paste_border, 3), 114, dtype=np.uint8)  # base image with 4 tiles
    mosaic_bboxes, mosaic_labels = [], []

    for order in range(4):
        cur_image = images[order]
        height, width = cur_image.shape[0], cur_image.shape[1]
        if order == 0:
            paste_top, paste_left, paste_bottom, paste_right = max(hh_center - height, 0), max(ww_center - width, 0), hh_center, ww_center
            cut_top, cut_left, cut_bottom, cut_right = height - (hh_center - paste_top), width - (ww_center - paste_left), height, width
        elif order == 1:
            paste_top, paste_left, paste_bottom, paste_right = max(hh_center - height, 0), ww_center, hh_center, min(ww_center + width, paste_border)
            cut_top, cut_left, cut_bottom, cut_right = height - (hh_center - paste_top), 0, height, min(paste_right - ww_center, width)
        elif order == 2:
            paste_top, paste_left, paste_bottom, paste_right = hh_center, max(ww_center - width, 0), min(hh_center + height, paste_border), ww_center
            cut_top, cut_left, cut_bottom, cut_right = 0, width - (ww_center - paste_left), min(paste_bottom - hh_center, height), width
        elif order == 3:
            paste_top, paste_left, paste_bottom, paste_right = hh_center, ww_center, min(hh_center + height, paste_border), min(ww_center + width, paste_border)
            cut_top, cut_left, cut_bottom, cut_right = 0, 0, min(paste_bottom - hh_center, height), min(paste_right - ww_center, width)

        mosaic_image[paste_top:paste_bottom, paste_left:paste_right] = cur_image[cut_top:cut_bottom, cut_left:cut_right]
        hh_offset, ww_offset = paste_top - cut_top, paste_left - cut_left
        mosaic_bboxes.append(bboxes[order] * [height, width, height, width] + [hh_offset, ww_offset, hh_offset, ww_offset])
        mosaic_labels.append(labels[order])
    mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
    mosaic_labels = np.concatenate(mosaic_labels, axis=0)

    mosaic_bboxes = np.clip(mosaic_bboxes, 0, paste_border)
    return mosaic_image, mosaic_bboxes, mosaic_labels

class DetectionDataset(Dataset):  # for training/testing
    def __init__(self, data, image_size=640, batch_size=16, is_train=True, rect=False, pad=0.0, mosaic=1.0):
        self.data = data
        self.image_size = image_size
        self.rect = False if is_train else rect
        self.mosaic = mosaic
        self.degrees, self.translate, self.scale, self.shear = 10, 0.1, 0.1, 10
        self.hgain, self.sgain, self.vgain = 0.5, 0.5, 0.5
        self.fliplr = 0.5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        image_path, objects = datapoint["image"], datapoint["objects"]
        bbox, label = np.array(objects["bbox"], dtype='float32'), np.array(objects["label"], dtype="int64")

        image = cv2.imread(image_path)
        orign_height, orign_width = image.shape[:2]

        if self.is_train and random.random() < self.mosaic:
            images, bboxes, labels = [image], [bbox], [label]
            for ii in [random.randint(0, len(self.data) - 1) for _ in range(3)]  # 3 additional image indices
                datapoint = self.data[ii]
                images.append(load_image(datapoint["image"])[0])
                bboxes.append(datapoint["objects"]["bbox"])
                labels.append(datapoint["objects"]["label"])
            image, bbox, label = combine_mosaic(images, bboxes, labels)
        else:
            bbox *= [orign_height, orign_width, orign_height, orign_width]

        if self.is_train:
            image, bbox, label = random_perspective(
                image, bbox, label, target_shape=self.image_size, degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear
            )
            augment_hsv(image, hgain=self.hgain, sgain=self.sgain, vgain=self.vgain)
        bbox /= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]  # normalized 0-1

        if self.is_train and random.random() < self.fliplr:
            image = np.fliplr(image)
            labels[:, [1, 3]] = 1 - labels[:, [1, 3]]
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB -> channels first
        image = np.ascontiguousarray(image)
        return image, bbox, label
        # return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, "Image Not Found " + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(image, bbox, label, target_shape=(640, 640), degrees=10, translate=0.1, scale=0.1, shear=10):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    height, width = target_shape[:2] if isinstance(target_shape, (list, tuple)) else (target_shape, target_shape)

    # Center
    cc = np.eye(3)
    cc[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    cc[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    rr = np.eye(3)
    rr[:2] = cv2.getRotationMatrix2D(angle=random.uniform(-degrees, degrees), center=(0, 0), scale=random.uniform(1 - scale, 1 + scale))

    # Shear
    ss = np.eye(3)
    ss[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    ss[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    tt = np.eye(3)
    tt[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    tt[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    mm = tt @ ss @ rr @ cc  # order of operations (right to left) is IMPORTANT
    image = cv2.warpAffine(image, mm[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # warp points
    bbox = bbox[:, [1, 0, 3, 2, 3, 0, 1, 2]].reshape(-1, 2)  # x1y1, x2y2, x1y2, x2y1
    bbox = np.pad(bbox, [[0, 0], [0, 1]], constant_values=1)
    bbox = bbox @ mm.T  # transform
    bbox = bbox[:, :2].reshape(-1, 8)

    # create new boxes
    xx, yy = bbox[:, [0, 2, 4, 6]], bbox[:, [1, 3, 5, 7]]
    bbox = np.concatenate((yy.min(1), xx.min(1), yy.max(1), xx.max(1))).reshape(4, -1).T

    # clip boxes
    bbox[:, [0, 2]] = bbox[:, [0, 2]].clip(0, height)
    bbox[:, [1, 3]] = bbox[:, [1, 3]].clip(0, width)

    # filter candidates
    valid_candidates = np.logical_and(bbox[:, 2] - bbox[:, 0] > 2, bbox[:, 3] - bbox[:, 1] > 2)
    return image, bbox[valid_candidates], label[valid_candidates]


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
