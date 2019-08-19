#!/bin/bash

if [[ $HOSTNAME == "vipa210" ]]; then
    echo "run on vipa210"
    python run.py \
    --gpu="2" --lr=0.1 --epochs=80 \
    --train_v="ensemble-3.0" --load_v="ensemble-3.0" \
    --regularize=0 --retrain=1 --batch_size=128 \
    --T=4 --alpha=0.1

elif [[ $HOSTNAME == "vipa-Precision-Tower-7910" ]]; then
    echo "run on vipa-Precision-Tower-7910"
    python run.py \
    --gpu="0" --lr=0.1 --epochs=80 \
    --train_v="resnet-tiny-n5" --load_v="resnet-tiny-n5" \
    --regularize=0 --retrain=0 --batch_size=128 \
    --T=4 --alpha=0.1
else
    echo "unknown host name"
fi

rm model_zoo/model/*epoch*
