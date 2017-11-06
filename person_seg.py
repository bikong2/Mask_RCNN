# @author: lixihua9@126.com

import sys, os, time
import random, math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2
import coco
import utils
import model as modellib
import visualize

class PersonSegment(object):

    def __init__(self, MODEL_DIR, COCO_MODEL_PATH):
        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.print_py2()
        self._model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self._model.load_weights(COCO_MODEL_PATH, by_name=True)
        """
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
        """
    
    def seg(self, img_path):
        image = scipy.misc.imread(img_path)
        results = self._model.detect([image], verbose=1)
        r = results[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        return r['masks']

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
person_segment = PersonSegment(os.path.join(ROOT_DIR, "logs"), os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5"))


#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
#file_names = next(os.walk(IMAGE_DIR))[2]
img_path = "../kuaibao_project/images/59db7e9cN4ba43971.jpg"
masks = person_segment.seg(img_path)
image = scipy.misc.imread(img_path)

for i in range(masks.shape[2]):
    cv2.imshow("image", image)
    print masks[:,:,i].shape
    cv2.imshow("masks", 255*masks[:,:,i])
    cv2.waitKey()

# The END!
