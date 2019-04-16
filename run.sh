#!/bin/bash

if [[ $HOSTNAME == "vipa210" ]]; then
    echo "run on vipa210"
    python run.py \
    --gpu="3" --lr=0.01 --epochs=100 \
    --train_v="googlenet-1.0" --load_v="googlenet-1.0" \
    --regularize=0 --retrain=1 --batch_size=256

elif [[ $HOSTNAME == "vipa-Precision-Tower-7910" ]]; then
    echo "run on vipa-Precision-Tower-7910"
    python run.py \
    --gpu="1" --lr=0.01 --epochs=100 \
    --train_v="googlenet-2.0" --load_v="googlenet-2.0" \
    --regularize=0 --retrain=0 --batch_size=256
else
    echo "unknown host name"
fi

rm model_zoo/model/*epoch*
