from torch.utils import data
from model_zoo import load_model
from model_zoo import googLeNet
import train


def test(test_version, test_set, batch_size=32):
    print("[info]: testing model...")
    # load model
    model, create_new = load_model.load_model(
        version=test_version,
        new_model=googLeNet.my_googLeNet,
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
    loss, acc = train.eval_model(model, test_loader)
    print("[info]: test loss: {:5f}, test acc: {:4f}".format(loss, acc))

