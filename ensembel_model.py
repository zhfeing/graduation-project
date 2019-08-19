from torch import nn
from model_zoo import resnet, googLeNet, load_model


class EnsembleModel(nn.Module):
    def __init__(self, googlenet_list, resnet_list, to_cuda=True):
        super(EnsembleModel, self).__init__()
        self._googlenet_list = googlenet_list
        self._resnet_list = resnet_list
        self._to_cuda = to_cuda
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
                to_cuda=self._to_cuda
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
                to_cuda=self._to_cuda
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


def my_ensembel_model(*args):
    model = EnsembleModel(["googlenet-4.0", "googlenet-3.0", "googlenet-1.0"], ["resnet-3.0"], *args)
    return model
