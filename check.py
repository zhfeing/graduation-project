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
    def __init__(self, googlenet_list, resnet_list):
        super(EnsembleModel, self).__init__()
        self._googlenet_list = googlenet_list
        self._resnet_list = resnet_list
        self._google_model = list()
        self._resnet_model = list()
        self.build()

    def build(self):
        for version in self._googlenet_list:
            model, create_new = load_model.load_model(
                version=version,
                new_model=googLeNet.my_googLeNet,
                just_weights=False,
                retrain=False,
                to_cuda=True
            )
            if create_new:
                print("[info]: load googlenet failed")
                exit(-1)
            self._google_model.append(model)

        for version in self._resnet_list:
            model, create_new = load_model.load_model(
                version=version,
                new_model=resnet.my_resnet,
                just_weights=False,
                retrain=False,
                to_cuda=True
            )
            if create_new:
                print("[info]: load resnet failed")
                exit(-1)
            self._resnet_model.append(model)

    def forward(self, x):
        y = 0
        for model in self._google_model:
            y += model(x)[0]

        for model in self._resnet_model:
            y += model(x)

        y /= len(self._google_model) + len(self._resnet_model)
        return y


def check_ensemble():
    import train
    from torch.utils import data

    model = EnsembleModel(["googlenet-4.0", "googlenet-3.0", "googlenet-1.0"], ["resnet-3.0"])
    model.eval()
    default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    train_set, valid_set, test_set = import_data.import_dataset(load_dir=default_load_data_dir)

    test_loader = data.DataLoader(test_set, batch_size=32)

    loss, acc = train.eval_model(
        model, test_loader,
        lambda pred, y, x: nn.CrossEntropyLoss()(pred, y),
        lambda x: x, lambda x: x.detach()
    )
    print("loss: {}, acc: {}".format(loss, acc))

    # load_model.print_parameters(model)


def get_test_acc(version):
    import test
    from model_zoo import resnet, googLeNet
    import utils

    new_model = resnet.my_resnet
    new_util = utils.ResNetUtils()

    default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    train_set, valid_set, test_set = import_data.import_dataset(load_dir=default_load_data_dir, )
    test.test(
        version, test_set, new_model, new_util.loss_for_eval,
        new_util.get_true_pred, new_util.detach_pred, 128, just_weights=False
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


def get_params(version):
    model, create_new = load_model.load_model(
        version=version,
        new_model=resnet.my_resnet,
        just_weights=False,
        retrain=False,
        to_cuda=False
    )
    if create_new:
        print("[info]: load resnet failed")
        exit(-1)
    load_model.print_parameters(model)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)
    version = "ensemble-3.0"
    get_test_acc(version)
    # get_params(version)
    # check_ensemble()

