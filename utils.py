from torch import nn


class GoogLeNetUtils:
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

    def get_true_pred(self, pred):
        return pred[0]

    def loss_for_train(self, pred, y):
        loss = self._loss(pred[0], y) + 0.3*self._loss(pred[1], y) + 0.3*self._loss(pred[2], y)
        return loss

    def loss_for_eval(self, pred, y):
        loss = self._loss(pred[0], y)
        return loss

    def detach_pred(self, pred):
        for i in pred:
            i = i.detach()
        return pred
