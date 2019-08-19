import time
from model_zoo import load_model, resnet, googLeNet
import ensembel_model
import utils
import cv2
import numpy as np
import torch
from torch import nn
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

# test resnet
version = 'resnet-tiny-n7'
new_model = resnet.my_resnet

mean = np.array([[[[113.91022]],
                  [[123.0098]],
                  [[125.40064]]]], dtype=np.float32)


# model, create_new_model = load_model.load_model(
#     version=version,
#     new_model=new_model,
#     just_weights=False,
#     retrain=False,
#     to_cuda=False
# )
model = ensembel_model.my_ensembel_model(False)
model.eval()

test_size = 20
time_cost = []
for i in range(test_size):
    img = cv2.imread("get_data/data_sample/{}.png".format(i))
    img = img.transpose([2, 0, 1]).reshape([1, 3, 32, 32]).astype(np.float32)
    img = (img - mean)/64.15484306
    time_start = time.time()
    x = torch.Tensor(img)
    y = model(x).detach()
    y = nn.Softmax(dim=1)(y).numpy()
    time_end = time.time()
    time_cost.append(time_end - time_start)

time_cost = np.array(time_cost)
print(time_cost.mean()*1000, time_cost.std()*1000)
