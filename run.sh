#!/bin/bash

if [[ $HOSTNAME == "vipa210" ]]; then
    echo "run on vipa210"
    python run.py \
    --gpu="1" --lr=0.01 --epochs=50 \
    --train_v="resnet-1.1" --load_v="resnet-1.0" \
    --regularize=0 --retrain=0 --batch_size=256

elif [[ $HOSTNAME == "vipa-Precision-Tower-7910" ]]; then
    echo "run on vipa-Precision-Tower-7910"
    python run.py \
    --gpu="1" --lr=0.01 --epochs=50 \
    --train_v="resnet-2.1" --load_v="resnet-2.0" \
    --regularize=0 --retrain=0 --batch_size=256
else
    echo "unknown host name"
fi

rm model_zoo/model/*epoch*
