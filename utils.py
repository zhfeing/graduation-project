from torch import nn
import torch


class GoogLeNetUtils:
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

    def get_true_pred(self, pred):
        return pred[0]

    def loss_for_train(self, pred, y, x=None):
        loss = self._loss(pred[0], y) + 0.3*self._loss(pred[1], y) + 0.3*self._loss(pred[2], y)
        return loss

    def loss_for_eval(self, pred, y, x=None):
        loss = self._loss(pred[0], y)
        return loss

    def detach_pred(self, pred):
        p = list()
        for i in range(len(pred)):
            p.append(pred[i].detach())
        return p

    def learn_rate_schedule(self, epoch, optimizer):
        if epoch == 50 or epoch == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("\n[info]: lr changed")


class ResNetUtils:
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

    def get_true_pred(self, pred):
        return pred

    def loss_for_train(self, pred, y, x=None):
        loss = self._loss(pred, y)
        return loss

    def loss_for_eval(self, pred, y, x=None):
        loss = self._loss(pred, y)
        return loss

    def detach_pred(self, pred):
        return pred.detach()

    def learn_rate_schedule(self, epoch, optimizer):
        if epoch == 40 or epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("\n[info]: lr divided by 10")


class DistillModelUtils:
    def __init__(self, cumbersome_model, T, alpha):
        self._cumbersome_model = cumbersome_model
        self._T = T
        self._alpha = alpha

    def my_KLDiv(self, pred, target):
        R = nn.Softmax(dim=1)(target/self._T)
        Q = nn.Softmax(dim=1)(pred/self._T)
        loss = R*(R.log() - Q.log())
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

    def get_loss(self, pred, y, x):
        target = self._cumbersome_model(x).detach()
        hard_loss = nn.CrossEntropyLoss(reduction='mean')(pred, y)
        soft_loss = self._T * self._T * self.my_KLDiv(pred, target)
        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss

    def get_true_pred(self, pred):
        return pred

    def loss_for_train(self, pred, y, x):
        loss = self.get_loss(pred, y, x)
        return loss

    def loss_for_eval(self, pred, y, x):
        loss = self.get_loss(pred, y, x)
        return loss

    def detach_pred(self, pred):
        return pred.detach()

    def learn_rate_schedule(self, epoch, optimizer):
        if epoch == 40 or epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("\n[info]: lr changed")

