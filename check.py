import torch
from torch import nn
from model_zoo import googLeNet, load_model, resnet
from get_data import import_data
import numpy as np
import cv2
import os


def soft_max(x):
    num = np.exp(x)
    den = np.sum(num, axis=1, keepdims=True)
    return num/den


def check_failed_example():
    model, create_new = load_model.load_model(
        version="googlenet-2.0",
        new_model=googLeNet.my_googLeNet,
        just_weights=False,
        retrain=False,
        to_cuda=True
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


class EnsembleModel(nn.Module):
    def __init__(self, googlenet_version_1, googlenet_version_2, resnet_version):
        super(EnsembleModel, self).__init__()
        self._googlenet_version_1 = googlenet_version_1
        self._googlenet_version_2 = googlenet_version_2
        # self._googlenet_version_3 = googlenet_version_3
        self._resnet_version = resnet_version
        self._google_model_1 = None
        self._google_model_2 = None
        self._resnet_model = None
        self.build()

    def build(self):
        self._google_model_1, create_new = load_model.load_model(
            version=self._googlenet_version_1,
            new_model=googLeNet.my_googLeNet,
            just_weights=False,
            retrain=False,
            to_cuda=True
        )
        if create_new:
            print("[info]: load googlenet failed")
            exit(-1)

        self._google_model_2, create_new = load_model.load_model(
            version=self._googlenet_version_2,
            new_model=googLeNet.my_googLeNet,
            just_weights=False,
            retrain=False,
            to_cuda=True
        )
        if create_new:
            print("[info]: load googlenet failed")
            exit(-1)
        #
        # self._google_model_3, create_new = load_model.load_model(
        #     version=self._googlenet_version_3,
        #     new_model=googLeNet.my_googLeNet,
        #     just_weights=False,
        #     retrain=False,
        #     to_cuda=True
        # )
        # if create_new:
        #     print("[info]: load googlenet failed")
        #     exit(-1)

        self._resnet_model, create_new = load_model.load_model(
            version=self._resnet_version,
            new_model=resnet.my_resnet,
            just_weights=False,
            retrain=False,
            to_cuda=True
        )
        if create_new:
            print("[info]: load resnet failed")
            exit(-1)

    def forward(self, x):
        x1, _, _ = self._google_model_1(x)
        x2, _, _ = self._google_model_2(x)
        # x3, _, _ = self._google_model_3(x)
        x4 = self._resnet_model(x)
        x = (x1 + x2 + x4)/3.0
        return x


def check_ensemble():
    import train
    from torch.utils import data

    model = EnsembleModel("googlenet-1.0", "googlenet-2.0", "resnet-2.0")

    default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    train_set, valid_set, test_set = import_data.import_dataset(load_dir=default_load_data_dir)

    test_loader = data.DataLoader(test_set, batch_size=128)

    loss, acc = train.eval_model(model, test_loader, nn.CrossEntropyLoss(), lambda x: x, lambda x: x.detach())
    print("loss: {}, acc: {}".format(loss, acc))


def get_test_acc():
    import test
    from model_zoo import resnet
    import utils

    new_model = resnet.my_resnet
    new_util = utils.ResNetUtils()

    default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    train_set, valid_set, test_set = import_data.import_dataset(load_dir=default_load_data_dir)
    test.test(
        "resnet-1.0", test_set, new_model, new_util.loss_for_eval,
        new_util.get_true_pred, new_util.detach_pred, 128
    )


def get_acc_by_label():
    from model_zoo import googLeNet
    from model_zoo import load_model
    from torch.utils import data

    model, create_new = load_model.load_model(
        version="resnet-2.0",
        new_model=googLeNet.my_googLeNet,
        just_weights=False,
        retrain=False,
        to_cuda=True
    )
    if create_new:
        print("[info]: load googlenet failed")
        exit(-1)

    default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    train_set, valid_set, test_set = import_data.import_dataset(default_load_data_dir, train_to_cuda=False)
    loader = data.DataLoader(test_set, batch_size=128)

    acc_tabel = np.zeros([10, 10])

    for step, (x, y) in enumerate(loader):
        batch_size = x.size()[0]
        pred = model(x)

        pred_label = pred.cpu().detach().numpy().argmax(axis=1)
        true_label = y.cpu().detach().numpy()
        for i in range(batch_size):
            acc_tabel[true_label[i], pred_label[i]] += 1

    acc_tabel /= len(loader.dataset)/10.0
    print(acc_tabel)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)
    check_ensemble()

