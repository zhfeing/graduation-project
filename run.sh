#!/bin/bash

if [[ $HOSTNAME == "vipa210" ]]; then
    echo "run on vipa210"
    python run.py \
    --gpu="2" --lr=0.1 --epochs=100 \
    --train_v="resnet-1.0" --load_v="resnet-1.0" \
    --regularize=0 --retrain=1 --batch_size=128

elif [[ $HOSTNAME == "vipa-Precision-Tower-7910" ]]; then
    echo "run on vipa-Precision-Tower-7910"
    python run.py \
    --gpu="0" --lr=0.1 --epochs=100 \
    --train_v="resnet-2.0" --load_v="resnet-2.0" \
    --regularize=1 --retrain=1 --batch_size=128
else
    echo "unknown host name"
fi

rm model_zoo/model/*epoch*
