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
        p = list()
        for i in range(len(pred)):
            p.append(pred[i].detach())
        return p


class ResNetUtils:
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

    def get_true_pred(self, pred):
        return pred

    def loss_for_train(self, pred, y):
        loss = self._loss(pred, y)
        return loss

    def loss_for_eval(self, pred, y):
        loss = self._loss(pred, y)
        return loss

    def detach_pred(self, pred):
        return pred.detach()


def google_learn_rate_schedule(epoch, optimizer):
    if epoch == 50 or epoch == 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print("\n[info]: lr changed")


def resnet_learn_rate_schedule(epoch, optimizer):
    if epoch == 50 or epoch == 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print("\n[info]: lr divided by 10")
