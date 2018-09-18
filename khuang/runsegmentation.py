import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
from samples.coco import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "/local/MT/Datasets/living_room_traj1n_frei_png/rgb"


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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




from PIL import Image

imgs = os.listdir(IMAGE_DIR)
imgs.sort()

im = Image.open(os.path.join(IMAGE_DIR, imgs[1]))


print(len(imgs))



for img in range(0,len(imgs)):
#     img = '{:d}'.format(img).zfill(4)
#     file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgs[img]))
    print(img)
    print(imgs[img])
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
#    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],  class_names, r['scores'])
    
    shape = r['masks'].shape
    num_obj = shape[2]
    print(num_obj)

    masks=r['masks']
    mask = np.zeros([480,640])
    
    classid=r['class_ids']
    
    
    for i in range(1,num_obj+1):
        print(str(img)+'.'+str(i))
        print('Class_ID')
        print(classid[i-1])
        print(class_names[classid[i-1]])
        mask_tmp = np.array(masks[:,:,i-1]*classid[i-1])
        mask = np.maximum(mask,mask_tmp)
        mask=masks[:,:,i-1]*classid[i-1]
        np.savetxt(os.path.join(IMAGE_DIR,'../pixel_label/'+str(imgs[img])+'.'+str(i)+".txt"),mask,fmt='%i')
