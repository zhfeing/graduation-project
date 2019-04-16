import torch
from torch.utils import data
from torch import optim
import numpy as np
import os
from get_data import data_augmentation


def eval_model(model, data_loader, eval_loss_function, get_true_pred, detach_pred):
    """
    eval_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        eval_loss_function(pred, label), which pred comes from model output
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    detach_pred: detach pred from output, useful when model output a tuple called as detach_pred(module_output)
    """
    model.eval()
    acc = 0
    loss = 0
    num = len(data_loader)
    for step, (x, y) in enumerate(data_loader):
        batch_size = x.size()[0]
        pred = model(x)
        pred = detach_pred(pred)
        loss += eval_loss_function(pred, y)
        pred = get_true_pred(pred)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.item(), acc.item()


def fit(
        model, epoch, optimizer,
        train_loader, valid_loader,
        check_freq, train_version,
        train_loss_function, get_true_pred, eval_loss_function, detach_pred,
        learn_rate_schedule
):
    """
    check_freq: check training result in step
    train_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        train_loss_function(pred, label)
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    eval_loss_function: called by function 'eval_model'
    detach_pred: called by function 'eval_model'
    learn_rate_schedule: called as learn_rate_schedule(epoch, optimizer)
    return: loss and acc history
    """

    # train history
    loss_his = list()
    acc_his = list()
    loss_val_his = list()
    acc_val_his = list()

    for ep in range(epoch):
        epoch_best_loss = np.inf
        epoch_best_acc = 0
        model.train()

        learn_rate_schedule(ep, optimizer)

        for step, (x, y) in enumerate(train_loader):
            batch_size = x.size()[0]
            x = data_augmentation.tensor_data_argumentation(x)
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = train_loss_function(pred, y)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save check point
            if step % check_freq == 0:
                pred = torch.max(get_true_pred(pred), 1)[1]
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
                            "model_weights_{}_epoch_{}.pkl".format(train_version, ep)
                        )
                    )
                    print("\n[info]: save model with loss: {:.5f}, acc: {:.4f}".format(epoch_best_loss, epoch_best_acc))

        # eval
        print("\n\n have an evaluation...")
        val_loss, val_acc = eval_model(
            model=model,
            data_loader=valid_loader,
            eval_loss_function=eval_loss_function, get_true_pred=get_true_pred,
            detach_pred=detach_pred
        )

        loss_val_his.append(val_loss)
        acc_val_his.append(val_acc)

        print("[info]: val loss: {:5f}, val acc: {:4f}\n".
              format(val_loss, val_acc))

        torch.save(
            model.state_dict(),
            os.path.join(
                "./model_zoo/model",
                "model_weights_{}_epoch_{}_final.pkl".format(train_version, ep)
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


def train(
        model, train_set, valid_set, lr, epoch, train_version, batch_size, regularize,
        train_loss_function, get_true_pred, eval_loss_function, detach_pred,
        learn_rate_schedule, check_freq=5
):
    """
    model: cuda model
    train_set:  cuda dataset
    valid_set:  cuda valid set
    train_loss_function: called by function 'fit'
    get_true_pred:  called by function 'fit'
    eval_loss_function: called by function 'fit'
    learn_rate_schedule: called by function 'fit'
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

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    loss_val_his, acc_val_his = fit(
        model=model,
        epoch=epoch,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        check_freq=check_freq,
        train_version=train_version,
        train_loss_function=train_loss_function,
        get_true_pred=get_true_pred,
        eval_loss_function=eval_loss_function,
        detach_pred=detach_pred,
        learn_rate_schedule=learn_rate_schedule
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
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}.pkl".
                             format(train_version, ep))
            )
        )
        loss, acc = eval_model(
            model=model,
            data_loader=valid_loader,
            eval_loss_function=eval_loss_function,
            get_true_pred=get_true_pred,
            detach_pred=detach_pred
        )
        print("[info]: epoch {}: val acc: {:.4f}".format(ep, acc))
        best_epoch_loss_val = np.append(best_epoch_loss_val, loss)
        best_epoch_acc_val = np.append(best_epoch_acc_val, acc)

    best_epoch_id = np.argmax(best_epoch_acc_val)
    print("[info]: best epoch val acc: {:.4f}".format(best_epoch_acc_val[best_epoch_id]))
    if acc_val_his[best_final_id] > best_epoch_acc_val[best_epoch_id]:
        print("[info]: choose batch final module")
        model.load_state_dict(
            torch.load(
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}_final.pkl".
                             format(train_version, best_final_id))
            )
        )
    else:
        print("[info]: choose batch best module")
        model.load_state_dict(
            torch.load(
                os.path.join("model_zoo/model", "model_weights_{}_epoch_{}.pkl".
                             format(train_version, best_epoch_id))
            )
        )

    torch.save(
        model.state_dict(),
        os.path.join(
            "./model_zoo/model",
            "model_weights_{}.pkl".format(train_version)
        )
    )



