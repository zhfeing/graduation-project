import argparse
import os
import draw_his
import train
import test
from get_data import import_data
from model_zoo import googLeNet
from model_zoo import load_model


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store', type=str, default="0")
parser.add_argument('--lr', action='store', type=float, default=0.001)
parser.add_argument('--epochs', action='store', type=int, default=10)
parser.add_argument('--train_v', action='store', type=str, default="1.0")
parser.add_argument('--load_v', action='store', type=str, default="1.0")

default_load_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
# default_load_data_dir = "/media/Data/datasets/cifar/cifar-10-python/data"
parser.add_argument('--load_data_dir', action='store', type=str, default=default_load_data_dir)
parser.add_argument('--retrain', type=lambda x: bool(str2bool(x)), default=False)
parser.add_argument('--regularize', type=lambda x: bool(str2bool(x)), default=False)
parser.add_argument('--batch_size', action='store', type=int, default=32)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

print("[info]: use gpu: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
print("[info]: set learning rate: {}".format(args.lr))
print("[info]: epochs: {}".format(args.epochs))
print("[info]: train_version: {}".format(args.train_v))
print("[info]: load_version: {}".format(args.load_v))
print("[info]: retrain: {}".format(args.retrain))
print("[info]: regularize: {}".format(args.regularize))
print("[info]: batch_size: {}".format(args.batch_size))


model, create_new = load_model.load_model(
    version=args.load_v,
    new_model=googLeNet.my_googLeNet,
    retrain=args.retrain,
    to_cuda=True
)

train_set, valid_set, test_set = import_data.import_dataset(
    load_dir=args.load_data_dir
)

train.train(
    model=model,
    train_set=train_set,
    valid_set=valid_set,
    lr=args.lr,
    epoch=args.epochs,
    batch_size=args.batch_size,
    regularize=args.regularize,
    train_version=args.train_v
)

test.test(
    test_version=args.train_v,
    test_set=test_set,
    batch_size=args.batch_size
)

draw_his.draw_his(version=args.train_v, show=False)
