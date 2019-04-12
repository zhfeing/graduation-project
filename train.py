import torch
from torch.utils import data
from torch import nn
from torch import optim
import numpy as np
import os


lossness = nn.CrossEntropyLoss().cuda()


def calculate_loss(pred, y):
    return lossness(pred, y)


def eval_model(model, data_loader):
    model.eval()
    acc = 0
    loss = 0
    num = len(data_loader)
    for step, (x, y) in enumerate(data_loader):
        batch_size = x.size()[0]
        pred, _, _ = model(x)
        pred = pred.detach()
        loss += calculate_loss(pred, y)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.item(), acc.item()


def fit(model, epoch, optimizer, train_loader, valid_loader, check_freq, train_version):
    # train history
    loss_his = list()
    acc_his = list()
    loss_val_his = list()
    acc_val_his = list()

    for ep in range(epoch):
        epoch_best_loss = np.inf
        epoch_best_acc = 0
        model.train()

        if ep % 10 == 0 and ep > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
            print("\n[info]: lr halved")

        for step, (x, y) in enumerate(train_loader):
            batch_size = x.size()[0]
            pred_1, pred_2, pred_3 = model(x)
            loss = calculate_loss(pred_1, y) + 0.3*calculate_loss(pred_2, y) + 0.3*calculate_loss(pred_3, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save check point
            if step % check_freq == 0:
                pred = torch.max(pred_1, 1)[1]
                acc = (pred == y).sum().float() / batch_size

                loss_his.append(loss.item())
                acc_his.append(acc.item())

                print("\r[info]: epoch: {}/{}, step: {}/{}, loss: {:5f}, acc: {:4f}".
                      format(ep + 1, epoch, step, len(train_loader), loss, acc), end="")

                if loss.item() < epoch_best_loss and acc.item() > epoch_best_acc:
                    epoch_best_loss = loss.item()
                    epoch_best_acc = acc.item()
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            "./model_zoo/model",
                            "model_weights_{}_epoch_{}.pth".format(train_version, ep)
                        )
                    )
                    print("\n[info]: save model with loss: {:.5f}, acc: {:.4f}".format(epoch_best_loss, epoch_best_acc))

        # eval
        print("\n\n have an evaluation...")
        val_loss, val_acc = eval_model(model, valid_loader)

        loss_val_his.append(val_loss)
        acc_val_his.append(val_acc)

        print("[info]: val loss: {:5f}, val acc: {:4f}\n".
              format(val_loss, val_acc))

        torch.save(
            model.state_dict(),
            os.path.join(
                "./model_zoo/model",
                "model_weights_{}_epoch_{}_final.pth".format(train_version, ep)
            )
        )

    # after training
    loss_his = np.array(loss_his)
    acc_his = np.array(acc_his)
    loss_val_his = np.array(loss_val_his)
    acc_val_his = np.array(acc_val_his)
    np.save(os.path.join("./logs", "loss_his_{}".format(train_version)), loss_his)
    np.save(os.path.join("./logs", "acc_his_{}".format(train_version)), acc_his)
    np.save(os.path.join("./logs", "loss_val_his_{}".format(train_version)), loss_val_his)
    np.save(os.path.join("./logs", "acc_val_his_{}".format(train_version)), acc_val_his)
    return loss_val_his, acc_val_his


def train(model, train_set, valid_set, lr, epoch, train_version, batch_size, regularize, check_freq=5):
    """
    :param model: cuda model
    :param train_set:  cuda dataset
    :param valid_set:  cuda valid set
    :param lr:
    :return:
    """
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = data.DataLoader(
        dataset=valid_set,
        batch_size=batch_size
    )

    weight_decay = 0
    if regularize:
        weight_decay = 1e-4

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_val_his, acc_val_his = fit(
        model=model,
        epoch=epoch,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        check_freq=check_freq,
        train_version=train_version
    )

    # get best model
    print("[info]: getting best model...")
    # best final acc
    best_final_id = np.argmax(acc_val_his)
    print("[info]: best final id: {}, val acc: {:.4f}".format(best_final_id, acc_val_his[best_final_id]))

    # get best epoch model
    best_epoch_acc_val = np.array(list())
    best_epoch_loss_val = np.array(list())

    for ep in range(epoch):
        model.load_state_dict(
            torch.load(
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}.pth".
                             format(train_version, ep))
            )
        )
        loss, acc = eval_model(model, valid_loader)
        print("[info]: epoch {}: val acc: {:.4f}".format(ep, acc))
        best_epoch_loss_val = np.append(best_epoch_loss_val, loss)
        best_epoch_acc_val = np.append(best_epoch_acc_val, acc)

    best_epoch_id = np.argmax(best_epoch_acc_val)
    print("[info]: best epoch val acc: {:.4f}".format(best_epoch_acc_val[best_epoch_id]))
    if acc_val_his[best_final_id] > best_epoch_acc_val[best_epoch_id]:
        print("[info]: choose batch final module")
        model.load_state_dict(
            torch.load(
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}_final.pth".
                             format(train_version, best_final_id))
            )
        )
    else:
        print("[info]: choose batch best module")
        model.load_state_dict(
            torch.load(
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}.pth".
                             format(train_version, best_epoch_id))
            )
        )

    torch.save(
        model.state_dict(),
        os.path.join(
            "./model_zoo/model",
            "model_weights_{}.pth".format(train_version)
        )
    )



