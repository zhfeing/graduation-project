import torch
from torch import nn
from model_zoo import googLeNet, load_model
from get_data import import_data
import numpy as np
import cv2


def soft_max(x):
    num = np.exp(x)
    den = np.sum(num, axis=1, keepdims=True)
    return num/den


model, create_new = load_model.load_model(
    version="googlenet-1.0",
    new_model=googLeNet.my_googLeNet,
    retrain=False,
    to_cuda=False
)
if create_new:
    print("[info]: try to test a non-trained model")
    exit(-1)

data_dict = import_data.import_numpy_data(
    cifar_10_dir=None,
    load_dir="/media/Data/datasets/cifar/cifar-10-python/data",
    reload=False
)

train_x = data_dict['train_x']
train_y = data_dict['train_y']
valid_x = data_dict['valid_x']
valid_y = data_dict['valid_y']
test_x = data_dict['test_x']
test_y = data_dict['test_y']
mean = data_dict['mean']
std = data_dict['std']
label_names = data_dict['label_names']

model.eval()
cv2.namedWindow("fail case", cv2.WINDOW_NORMAL)
lossness = nn.CrossEntropyLoss(size_average=True)

for i in range(test_x.shape[0]):
    x = test_x[i].reshape([1, *test_x[i].shape])
    y = test_y[i].reshape([1, *test_y[i].shape])

    x_tensor = torch.Tensor(x)
    pred_tensor, _, _ = model(x_tensor)
    pred = pred_tensor.detach().numpy()

    if pred.argmax() != y.argmax():
        pred = soft_max(pred)
        x = x*std + mean
        x = x.astype(np.uint8)

        x = x.squeeze()
        x = np.transpose(x, axes=[1, 2, 0])

        cv2.imshow("fail case", x)

        print("predit label: {}, true label: {}".format(
            label_names[pred.argmax()],
            label_names[y.argmax()]
        ))

        print(i)
        print(pred, y)
        cv2.waitKey()

