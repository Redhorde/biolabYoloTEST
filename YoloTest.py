from fastai1.old.fastai.conv_learner import *
from fastai1.old.fastai.model import *

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from YoloFunctionality import yolo_cut


files_path = "zbiory/valid/fist/emg-03_PK-sequential-2018-03-27-12-12-46-342.mov-8907-fist.png"
vgg_arch = resnet34
vgg_sz = 300
vgg_data = ImageClassifierData.from_paths('./vgg', tfms=tfms_from_model(vgg_arch, vgg_sz), bs=128)
vgg_learn = ConvLearner.pretrained(vgg_arch, vgg_data, precompute=False)
vgg_learn.load('best_sgdr_v3')
vgg_result = pd.Series()

image = cv2.imread(files_path)
img_crop = yolo_cut(image)
#img_crop = image
#y0 = 200
#x0 = 500
#height = 600
#width = 600
#image = image[y0:y0 + height, x0:x0 + width]  # crop
img_crop = img_crop.astype(np.float32) / 255
img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
img_crop = cv2.resize(img_crop, (600, 600))


trn_tfms, val_tfms = tfms_from_model(vgg_arch, vgg_sz)
im = val_tfms(img_crop)

predictions = vgg_learn.predict_array(im[None])
res = np.argmax(predictions)
print("predictions:")
print(predictions)
print("argmax(predictions):")
print(res)
plt.imshow(img_crop)
plt.show()