from torch.utils import data
from model_zoo import load_model
import train


def test(test_version, test_set, new_model, eval_loss_function, get_true_pred, detach_pred, batch_size=32):
    print("[info]: testing model...")
    # load model
    model, create_new = load_model.load_model(
        version=test_version,
        new_model=new_model,
        just_weights=True,
        retrain=False,
        to_cuda=True
    )
    if create_new:
        print("[info]: try to test a non-trained model")
        exit(-1)

    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size
    )
    loss, acc = train.eval_model(
        model=model,
        data_loader=test_loader,
        eval_loss_function=eval_loss_function,
        get_true_pred=get_true_pred,
        detach_pred=detach_pred
    )
    print("[info]: test loss: {:5f}, test acc: {:4f}".format(loss, acc))
    return loss, acc

